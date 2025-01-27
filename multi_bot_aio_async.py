import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List
import ccxt
import ccxt.pro as ccxtpro
import colorama
import pandas as pd
import signal
import state
import sys
import time
import traceback
from datetime import datetime, timedelta

from colorama import Fore, Style
from config import load_config, Config, VERSION
from api.manager_async import ManagerAsync
from directionalscalper.core.exchanges import BybitExchangeAsync
from directionalscalper.core.strategies.logger import Logger
from directionalscalper.core.strategies.bybit.gridbased.lineargrid_base_async import LinearGridBaseFuturesAsync
from live_table_manager_with_state import LiveTableManager
from pathlib import Path
from rate_limit import RateLimitAsync

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)


logging = Logger(logger_name="SingleBot", filename="SingleBot.log", stream=True)

colorama.init()

general_rate_limiter_async = RateLimitAsync(50, 1)

class SingleBot:
    stop_ws = False
    ohlcvcs = {
        "1m": {},
        "3m": {}
    }
    open_position_data = {}
    active_long_symbols = set()
    active_short_symbols = set()
    unique_active_symbols = set()
    trading_symbols = set()
    subscribed_symbols = set()
    rotator_symbols_cache = {"timestamp": None, "symbols": []}
    long_threads = {}
    short_threads = {}
    traders = {
        "long": {},
        "short": {}
    }
    finished_trading_symbols = {
        "long": set(),
        "short": set()
    }

    start_time = time.time()
    last_positions_watched = int(time.time() * 1000)
    stop = False
    sigint_count = 0  # Counter for SIGINT signals

    CACHE_DURATION = 50  # Cache duration in seconds
    MAX_CANDLES = 1000
    OHLCV_TIMEFRAMES = ["1m", "3m"]

    def __init__(self, config: Config, exchange_name: str, account_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self.account_name = account_name

        self.exchange_config = next(
            (
                exch
                for exch in config.exchanges
                if exch.name == exchange_name and exch.account_name == account_name
            ),
            {},
        )

        if not self.exchange_config:
            raise ValueError(
                f"Exchange {exchange_name} with account {account_name} not found in the configuration file."
            )

        self.symbols_allowed = self.exchange_config.symbols_allowed
        self.long_mode = self.config.bot.linear_grid["long_mode"]
        self.short_mode = self.config.bot.linear_grid["short_mode"]

        self.config_graceful_stop_long = self.graceful_stop_long = self.config.bot.linear_grid.get("graceful_stop_long", False)
        self.config_graceful_stop_short = self.graceful_stop_short = self.config.bot.linear_grid.get("graceful_stop_short", False)
        self.config_auto_graceful_stop = self.config.bot.linear_grid.get("auto_graceful_stop", False)
        self.target_coins_mode = self.config.bot.linear_grid.get("target_coins_mode", False)
        self.whitelist = (
            set(self.config.bot.whitelist) if self.target_coins_mode else []
        )

        logging.info(
            f"Target coins mode is {'enabled' if self.target_coins_mode else 'disabled'}"
        )

        self.exchange = BybitExchangeAsync(
            self.exchange_config.api_key,
            self.exchange_config.api_secret,
            collateral_currency=self.exchange_config.collateral_currency or "USDT",
        )

        self.ws_exchange = ccxtpro.bybit(
            {
                "apiKey": self.exchange.api_key,
                "secret": self.exchange.secret_key,
                "newUpdates": False,
            }
        )

        self.manager = ManagerAsync(
            self.exchange,
            exchange_name=exchange_name,
            data_source_exchange=config.api.data_source_exchange,
            api=config.api.mode,
            path=Path("data", config.api.filename),
            url=f"{config.api.url}{config.api.filename}",
        )

        signal.signal(signal.SIGINT, self._signal_handler)

        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=max(self.symbols_allowed * 2, 100)))

    async def run(self):
        # Start the live table in a separate task
        asyncio.create_task(self._run_live_table())

        # Start periodic position updates
        asyncio.create_task(self._periodic_position_update())

        # load positions data, clear open orders etc
        await self._prepare()

        # subscribe to ws
        asyncio.create_task(self._subscribe_to_ws())

        # rotation loop
        await self.main_loop()

    async def _prepare(self):
        await self._update_position_data()

        symbols_to_load = self.open_position_symbols.union(self.whitelist)

        # preloading OHLCVCS for open positions and whitelisted symbols to generate signals
        await self._fetch_ohlcvcs(symbols_to_load)

        # try:
        #     self.exchange.cancel_all_open_orders_bybit()
        #     logging.info("Cleared all open orders on the exchange upon initialization.")
        # except Exception as e:
        #     logging.error(f"Exception caught while cancelling orders: {e}")

    async def main_loop(self):
        while not self.stop:
            try:
                current_time = time.time()

                logging.info("Main loop started at %s", current_time)

                await self._update_balance()

                await self._update_position_data()

                trading_symbols = self.open_position_symbols

                if self.target_coins_mode and self.whitelist:
                    trading_symbols.update(self.whitelist)

                new_symbols = {}

                # it might open more than the allowed number of positions if there are open orders
                # if len(trading_symbols) < self.symbols_allowed:
                if (
                    not self.rotator_symbols_cache["timestamp"]
                    or current_time - self.rotator_symbols_cache["timestamp"] >= self.CACHE_DURATION
                ):
                    logging.info("Fetching updated rotator symbols")

                    async with general_rate_limiter_async:
                        await self._fetch_updated_symbols()

                    logging.info(
                        f"Refreshed latest rotator symbols: {self.rotator_symbols_cache['symbols']}"
                    )
                else:
                    logging.info(
                        f"No refresh needed yet. Last update was at {self.rotator_symbols_cache['timestamp']}, less than 60 seconds ago."
                    )

                # should be allowed_symbols count - trading_symbols count
                new_symbols = [
                    symbol for symbol in self.rotator_symbols_cache['symbols']
                    if symbol not in trading_symbols
                ]
                # new_symbols = set(new_symbols[:self.exchange_config.symbols_allowed - len(trading_symbols)])

                if new_symbols:
                    logging.info(f"Adding new symbols: {new_symbols}")

                self.trading_symbols = trading_symbols.union(new_symbols)

                for symbol in list(self.trading_symbols):
                    if symbol not in self.ohlcvcs["1m"]:
                        await self._fetch_ohlcvcs({symbol})

                    if symbol not in self.ohlcvcs["1m"]:
                        logging.info(f"Symbol {symbol} not in 1m ohlcvcs. Removing from trading symbols until OHLCVC is loaded.")
                        self.trading_symbols.remove(symbol)

                logging.info(f"Trading symbols: {self.trading_symbols}")

                self.finished_trading_symbols["long"].clear()
                self.finished_trading_symbols["short"].clear()

                self._cleanup_threads()

                await self._remove_dangling_orders()

                logging.info("Main loop finished at %s after %s seconds", time.time(), time.time() - current_time)

                # sleep for 3 minutes to wait for potential orders to be executed
                await asyncio.sleep(180)
            except Exception as e:
                logging.exception(e)

        await self.exchange.exchange_async.close()

    async def _subscribe_to_ws(self):
        await asyncio.gather(
            self._subscribe_to_balance_ws(),
            self._subscribe_to_trades_ws(),
            # it returns invalid position data
            # self._subscribe_to_positions_ws(),
        )

        logging.info("Closing WebSocket connection...")

        await self.ws_exchange.close()

    async def _subscribe_to_balance_ws(self):
        while not self.stop:
            try:
                ws_balance = await self.ws_exchange.watch_balance()
                casted_balance = {
                    **ws_balance,
                    "info": {
                        'result': {
                            'list': ws_balance['info']
                        }
                    }
                }
                state.balance["available"], state.balance["total"] = self.exchange.parse_balance(casted_balance)
                state.balance["updated_at"] = self.ws_exchange.milliseconds()
            except Exception as e:
                logging.exception(e)

    async def _subscribe_to_trades_ws(self):
        while not self.stop:
            try:
                if not self.trading_symbols:
                    logging.info("No trading symbols to subscribe to trades for. Sleeping for 5 seconds.")
                    await self.ws_exchange.sleep(5000)
                    continue

                updated_trades = []

                try:
                    updated_trades = await self.ws_exchange.watch_trades_for_symbols(sorted(self.trading_symbols))
                except ccxt.errors.ExchangeError as e:
                    if "bybit error:already subscribe" in str(e):
                        logging.warning("Already subscribed error encountered. Retrying...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise

                await self._process_new_trades([trade for trade in updated_trades if self._bybit_symbol_reverse(trade["symbol"]) in self.trading_symbols])

            except Exception as e:
                logging.warning(e)

    async def _process_new_trades(self, trades):
        try:
            bybit_symbol_by_trades = trades[0]['symbol']
            symbol = self._bybit_symbol_reverse(bybit_symbol_by_trades)

            if symbol not in self.open_position_symbols:
                # if self.config_auto_graceful_stop:
                #     logging.info(f"Symbol {symbol} not in open position symbols and auto graceful stop is enabled. Skipping trades.")
                #     return

                if len(self.open_position_symbols) >= self.symbols_allowed:
                    logging.info(f"[{symbol}] Not in open position symbols and open position symbols limit reached. Skipping trades.")
                    return

            # if both runners are running, no need to process trades
            if (symbol in self.long_threads and not self.long_threads[symbol].done()) and (symbol in self.short_threads and not self.short_threads[symbol].done()):
                logging.info(f"[{symbol}] Both runners are running. Skipping trades.")
                return

            # Process 1m data
            ohlcvc_by_trades = self.ws_exchange.build_ohlcvc(trades, "1m")
            ohlcv_data_sliced = [entry[:6] for entry in ohlcvc_by_trades]
            ohlcvc_pd = self.exchange.convert_ohlcv_to_df(ohlcv_data_sliced)

            if symbol in self.ohlcvcs["1m"]:
                df = pd.concat(
                    [
                        self.ohlcvcs["1m"][symbol],
                        ohlcvc_pd,
                    ]
                )
                self.ohlcvcs["1m"][symbol] = df[~df.index.duplicated(keep="last")].tail(self.MAX_CANDLES)

            # Process 3m data - we'll update this when we get 3 1m candles
            if symbol in self.ohlcvcs["3m"]:
                last_3m_candle_time = self.ohlcvcs["3m"][symbol].index[-1]
                current_time = pd.Timestamp.now(tz="UTC").tz_localize(None)  # Get current time in UTC and make it naive

                # If more than 3 minutes have passed since last 3m candle
                if (current_time - last_3m_candle_time).total_seconds() >= 120:
                    # Fetch new 3m data
                    logging.info(f"[{symbol}] Fetching new 3m data")
                    new_3m_data = await self._fetch_ohlcvc(symbol, "3m", limit=5)
                    if new_3m_data is not None:
                        df = pd.concat(
                                [
                                    self.ohlcvcs["3m"][symbol],
                                    new_3m_data,
                                ]
                            )
                        self.ohlcvcs["3m"][symbol] = df[~df.index.duplicated(keep="last")].tail(self.MAX_CANDLES)

            # Generate signal using both timeframes
            if symbol in self.ohlcvcs["1m"] and symbol in self.ohlcvcs["3m"]:
                signal = self.exchange.generate_l_signals_from_data(
                    self.ohlcvcs["1m"][symbol],
                    self.ohlcvcs["3m"][symbol],
                    symbol
                )
                self._process_signal(symbol=symbol, signal=signal)
            else:
                logging.warning(f"Missing data for symbol {symbol} in one of the timeframes")

        except Exception as e:
            logging.warning(f"Error processing trades: {e}")

    async def _subscribe_to_positions_ws(self):
        while not self.stop:
            try:
                # print(f"Subscribing to positions ws. Last positions watched: {self.last_positions_watched}")

                # watching only position changes
                position_data = await self.ws_exchange.watch_positions()

                for position in position_data:
                    side = position['side']

                    if side is None:
                        continue

                    symbol = self._bybit_symbol_reverse(position["symbol"])

                    if position['info']['side'] == '' and symbol in self.open_position_data:
                        print(f"[{symbol}] EMPTY POSITION: {position}")

                        # logging.info(f"[{symbol}] {side} position closed. Removing from open position data.")

                        # existing_pos_idx = next(
                        #     (
                        #         idx
                        #         for idx, p in enumerate(self.open_position_data[symbol])
                        #         if p["side"].lower() == side
                        #     ),
                        #     None,
                        # )

                        # if existing_pos_idx is not None:
                        #     self.open_position_data[symbol].pop(existing_pos_idx)

                        # continue

                    self.trading_symbols.add(symbol)

                    if symbol not in self.open_position_data:
                        self.open_position_data[symbol] = [position]
                    else:
                        print(f"symbol: {symbol} in open_position_data")

                        existing_pos_idx = next(
                            (
                                idx
                                for idx, p in enumerate(self.open_position_data[symbol])
                                if p["side"].lower() == side
                            ),
                            None,
                        )

                        print(f"{symbol} positions: ", self.open_position_data[symbol])

                        print(f"existing_pos_idx: {existing_pos_idx}")

                        if existing_pos_idx is not None:
                            self.open_position_data[symbol][existing_pos_idx] = position
                        else:
                            print(f"{symbol} positions: ", self.open_position_data[symbol])
                            self.open_position_data[symbol].append(position)

                self._process_positions_update()
            except Exception as e:
                logging.exception(e)

    async def _cleanup(self):
        logging.info("Waiting for tasks to finish...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("Tasks finished. Exiting...")

    async def _run_live_table(self):
        table_manager = LiveTableManager()
        await asyncio.to_thread(table_manager.display_table)

    def _process_positions_update(self):
        try:
            self.open_position_symbols = set(self.open_position_data.keys())
            logging.info(f"Open position symbols: {self.open_position_symbols}")

            current_long_positions = sum(
                1
                for symbol, positions in self.open_position_data.items()
                for pos in positions
                if pos["side"] and pos["side"].lower() == "long"
            )
            current_short_positions = sum(
                1
                for symbol, positions in self.open_position_data.items()
                for pos in positions
                if pos["side"] and pos["side"].lower() == "short"
            )
            logging.info(
                f"Current long positions: {current_long_positions}, Current short positions: {current_short_positions}"
            )

            self._update_active_symbols()

            if self.config_auto_graceful_stop:
                if (
                    current_long_positions >= self.symbols_allowed
                    or len(self.unique_active_symbols) >= self.symbols_allowed
                ):
                    self.graceful_stop_long = True
                    logging.info(
                        f"GS Auto Check: Automatically enabled graceful stop for long positions. Current long positions: {current_long_positions}, Unique active symbols: {len(self.unique_active_symbols)}"
                    )
                else:
                   self.graceful_stop_long = self.config_graceful_stop_long
                   logging.info(
                        f"GS Auto Check: Reverting to config value for graceful stop long. Current long positions: {current_long_positions}, Unique active symbols: {len(self.unique_active_symbols)}, Config value: {self.config_graceful_stop_long}"
                    )

                if (
                    current_short_positions >= self.symbols_allowed
                    or len(self.unique_active_symbols) >= self.symbols_allowed
                ):
                    self.graceful_stop_short = True
                    logging.info(
                        f"GS Auto Check: Automatically enabled graceful stop for short positions. Current short positions: {current_short_positions}, Unique active symbols: {len(self.unique_active_symbols)}"
                    )
                else:
                    self.graceful_stop_short = self.config_graceful_stop_short
                    logging.info(
                        f"GS Auto Check: Reverting to config value for graceful stop short. Current short positions: {current_short_positions}, Unique active symbols: {len(self.unique_active_symbols)}, Config value: {self.config_graceful_stop_short}"
                    )

        except Exception as e:
            logging.info(f"Exception caught in _process_positions_update: {str(e)}")
            logging.info(traceback.format_exc())

    def _process_signal(self, symbol, signal):
        try:
            logging.info(f"Processing signal for symbol {symbol}. Signal: {signal}")

            if symbol in self.open_position_symbols:
                logging.info(
                    f"Received signal for open position symbol {symbol}. Checking open positions."
                )
                action_taken = self._start_thread_for_open_symbol(symbol, signal)
                return action_taken

            if signal != "neutral" and symbol in self.finished_trading_symbols[signal]:
                logging.info(
                    f"[{symbol}] Received {signal} signal for finished trading symbol. Skipping."
                )
                return

            logging.info(
                f"Signal: {signal} for new rotator symbol {symbol}, "
                f"Long Mode: {self.long_mode}, Short Mode: {self.short_mode}"
            )

            # Flags to track if actions are taken
            action_taken_long = False
            action_taken_short = False

            # Handle long signals
            if signal.lower() == "long":
                if self.long_mode:
                    # Determine if the bot can add a new long/short symbol
                    can_add_new_long_symbol = (
                        len(self.active_long_symbols) < self.exchange_config.symbols_allowed
                    )

                    if can_add_new_long_symbol or symbol in self.unique_active_symbols:
                        if self.graceful_stop_long:
                            logging.info(
                                f"Skipping long signal for {symbol} due to graceful stop long enabled and no open long position."
                            )
                        elif (
                            symbol not in self.long_threads or self.long_threads[symbol].done()
                        ):
                            logging.info(f"Starting long thread for symbol {symbol}.")
                            action_taken_long = self._start_thread_for_symbol(symbol, signal, "long")
                        else:
                            logging.info(
                                f"Long thread already running for symbol {symbol}. Skipping."
                            )
                    else:
                        logging.info(
                            f"Cannot open long position for {symbol}. Long positions limit reached or position already exists."
                        )
            # Handle short signals
            elif signal.lower() == "short":
                if self.short_mode:
                    can_add_new_short_symbol = (
                        len(self.active_short_symbols) < self.exchange_config.symbols_allowed
                    )

                    if (
                        can_add_new_short_symbol or symbol in self.unique_active_symbols
                    ):
                        if self.graceful_stop_short:
                            logging.info(
                                f"Skipping short signal for {symbol} due to graceful stop short enabled and no open short position."
                            )
                        elif (
                            symbol not in self.short_threads
                            or self.short_threads[symbol].done()
                        ):
                            logging.info(f"Starting short thread for symbol {symbol}.")
                            action_taken_short = self._start_thread_for_symbol(symbol, signal, "short")
                        else:
                            logging.info(
                                f"Short thread already running for symbol {symbol}. Skipping."
                            )
                    else:
                        logging.info(
                            f"Cannot open short position for {symbol}. Short positions limit reached or position already exists."
                        )
            else:
                logging.info(f"No action taken for neutral signal for new rotator symbol {symbol}.")

            # Update active symbols based on actions taken
            if action_taken_long or action_taken_short:
                self.unique_active_symbols.add(symbol)
                logging.info(
                    f"Action taken for new symbol {symbol}."
                )
            else:
                logging.info(
                    f"No action taken for new symbol {symbol}."
                )
        except Exception as e:
            logging.info(f"Exception caught in bybit_auto_rotation: {str(e)}")
            logging.info(traceback.format_exc())

    def _update_active_symbols(self):
        self.active_long_symbols = {
            symbol
            for symbol, positions in self.open_position_data.items()
            if any(pos["side"] and pos["side"].lower() == "long" for pos in positions)
        }
        self.active_short_symbols = {
            symbol
            for symbol, positions in self.open_position_data.items()
            if any(pos["side"] and pos["side"].lower() == "short" for pos in positions)
        }
        self.unique_active_symbols = self.active_long_symbols.union(
            self.active_short_symbols
        )

        logging.info(
            f"Updated active long symbols ({len(self.active_long_symbols)}): {self.active_long_symbols}"
        )
        logging.info(
            f"Updated active short symbols ({len(self.active_short_symbols)}): {self.active_short_symbols}"
        )
        logging.info(
            f"Updated unique active symbols ({len(self.unique_active_symbols)}): {self.unique_active_symbols}"
        )

    async def _fetch_updated_symbols(self) -> List[str]:
        current_time = time.time()

        # Check if the cached data is still valid
        if self.rotator_symbols_cache["timestamp"] and current_time - self.rotator_symbols_cache["timestamp"] < self.CACHE_DURATION:
            logging.info("Using cached rotator symbols")
            return self.rotator_symbols_cache["symbols"]

        # Fetch new data if cache is expired
        potential_symbols = []

        if self.whitelist:
            potential_symbols = [symbol for symbol in self.whitelist]
        else:
            potential_symbols = await self.manager.get_auto_rotate_symbols_async(
                min_qty_threshold=None,
                blacklist=self.config.bot.blacklist,
                whitelist=self.whitelist,
                max_usd_value=self.config.bot.max_usd_value,
            )

        # Update the cache with new data and timestamp
        # using list to preserve order
        self.rotator_symbols_cache["symbols"] = [
            self._bybit_symbol_reverse(sym) for sym in potential_symbols
        ]
        self.rotator_symbols_cache["timestamp"] = current_time

        logging.info(
            f"Fetched new rotator symbols: {self.rotator_symbols_cache['symbols']}"
        )
        return self.rotator_symbols_cache["symbols"]

    def _start_thread_for_open_symbol(self, symbol, signal):
        has_open_long = symbol in self.active_long_symbols
        has_open_short = symbol in self.active_short_symbols
        action_taken = False

        # Handle neutral signals by managing open positions
        if signal == "neutral":
            logging.info(
                f"Neutral signal received for {symbol}. Managing open positions."
            )
        else:
            logging.info(f"{signal} signal received for {symbol}. Starting thread.")

        logging.info(f"[{symbol}] Has open long: {has_open_long}")

        if has_open_long or self.long_mode:
            running_long_thread = symbol in self.long_threads and not self.long_threads[symbol].done()

            if not running_long_thread:
                thread_started = self._start_thread_for_symbol(symbol, signal, "long")
                action_taken = thread_started
                logging.debug(
                    f"{'Started' if thread_started else 'Failed to start'} long thread for symbol {symbol} based on {signal} signal"
                )
            else:
                logging.debug(f"Long thread already running for symbol {symbol}. Skipping.")

        logging.info(f"[{symbol}] Has open short: {has_open_short}")

        if has_open_short or self.short_mode:
            running_short_thread = symbol in self.short_threads and not self.short_threads[symbol].done()

            if not running_short_thread:
                thread_started = self._start_thread_for_symbol(symbol, signal, "short")
                action_taken = thread_started
                logging.debug(
                    f"{'Started' if thread_started else 'Failed to start'} short thread for symbol {symbol} based on {signal} signal"
                )
            else:
                logging.debug(f"Short thread already running for symbol {symbol}. Skipping.")

        # Log if no action was taken
        if not action_taken:
            logging.info(f"No thread started for open symbol {symbol}.")

        return action_taken

    def _start_thread_for_symbol(self, symbol, signal, action):
        # Check if a long thread is already running for this symbol
        if action == "long":
            if symbol in self.long_threads and not self.long_threads[symbol].done():
                logging.info(
                    f"Long thread already running for symbol {symbol}. Skipping."
                )
                return False
        # Check if a short thread is already running for this symbol
        elif action == "short":
            if symbol in self.short_threads and not self.short_threads[symbol].done():
                logging.info(
                    f"Short thread already running for symbol {symbol}. Skipping."
                )
                return False
        else:
            logging.warning(f"Invalid action {action} for symbol {symbol}")
            return False

        # Create an asyncio task for the strategy
        task = asyncio.create_task(self._run_strategy(symbol, signal, action))

        if action == "long":
            self.long_threads[symbol] = task
        elif action == "short":
            self.short_threads[symbol] = task

        # Start thread and log the action
        self.unique_active_symbols.add(symbol)
        logging.info(
            f"Started thread for symbol {symbol} with action {action} based on MFIRSI signal."
        )

        return True

    async def _run_strategy(
        self, symbol, signal=None, action='long'
    ):
        logging.info(f"Received rotator symbols in run_strategy for {symbol}")

        logging.info(f"[{symbol}] Traders: {self.traders.get(symbol, {}).get(action, None)}")

        if symbol not in self.traders[action] or not self.traders[action][symbol].running_trading(action):
            if symbol in self.traders[action] and not self.traders[action][symbol].running_trading(action):
                logging.info(f"Symbol {symbol} in traders for action {action}: false")

                if signal == 'neutral':
                    logging.info(f"Skipping {action} thread for symbol {symbol} for neutral signal if no open positions")
                    return

            logging.info(f"Configuring {action} trader for symbol {symbol}")

            trader = LinearGridBaseFuturesAsync(self.exchange, self.manager, self.config.bot, self.symbols_allowed)
            await trader.configure_trader(symbol)
            self.traders[action][symbol] = trader
        else:
            logging.info(f"[{symbol}] Trader already configured for action {action}")

            trader = self.traders[action][symbol]

        try:
            logging.info(f"Running strategy for symbol {symbol} with action {action}")

            if action == "long":
                await asyncio.to_thread(trader.run, symbol, mfirsi_signal=signal, action="long", open_symbols=self.open_position_symbols)

                logging.info(f"Long thread for symbol {symbol} completed")
                logging.info(f"[{symbol}] long can trade: {trader.running_long}")

                if not trader.running_long:
                    logging.info(f"Removing {symbol} from trading long symbols")
                    del self.traders[action][symbol]

                    # Remove only long position for this specific symbol
                    if symbol in self.open_position_data:
                        positions = self.open_position_data[symbol]
                        # Find the long position index
                        long_pos_idx = next((idx for idx, pos in enumerate(positions) if pos["side"].lower() == "long"), None)
                        if long_pos_idx is not None:
                            logging.info(f"Removing open long position for {symbol}")
                            # Remove only the long position
                            self.open_position_data[symbol].pop(long_pos_idx)

                    # self.trading_symbols.discard(symbol)
                    # self.finished_trading_symbols["long"].add(symbol)
            elif action == "short":
                await asyncio.to_thread(trader.run, symbol, mfirsi_signal=signal, action="short", open_symbols=self.open_position_symbols)

                logging.info(f"Short thread for symbol {symbol} completed")
                logging.info(f"[{symbol}] short can trade: {trader.running_short}")

                if not trader.running_short:
                    logging.info(f"Removing {symbol} from trading short symbols")
                    del self.traders[action][symbol]
                    # self.trading_symbols.discard(symbol)

                    # Remove only short position for this specific symbol
                    if symbol in self.open_position_data:
                        positions = self.open_position_data[symbol]
                        # Find the short position index
                        short_pos_idx = next((idx for idx, pos in enumerate(positions) if pos["side"].lower() == "short"), None)
                        if short_pos_idx is not None:
                            logging.info(f"Removing open short position for {symbol}")
                            # Remove only the short position
                            self.open_position_data[symbol].pop(short_pos_idx)

                    # self.finished_trading_symbols["short"].add(symbol)

            if not self.open_position_data[symbol]:
                logging.info(f"Removing {symbol} from open position data")
                self.open_position_data.pop(symbol, None)
                self.open_position_symbols.discard(symbol)
        except Exception as e:
            logging.error(f"Error running strategy for {symbol}: {e}")

    # convert symbol to bybit format
    def _bybit_symbol(self, symbol):
        return symbol.replace("USDT", "/USDT:USDT")

    # convert bybit symbol to standard format
    def _bybit_symbol_reverse(self, symbol):
        return symbol.replace("/USDT:USDT", "USDT")

    async def _update_position_data(self):
        logging.info("Updating position data")

        # preloading open positions here to have them available immediately for websocket
        open_position_data = await self.exchange.get_all_open_positions_async()

        self.open_position_data = {}

        for position in open_position_data:
            symbol = self._bybit_symbol_reverse(position["symbol"])

            if symbol in self.config.bot.blacklist:
                logging.info(f"[{symbol}] skipping position update due to blacklist")
                continue

            if symbol not in self.open_position_data:
                self.open_position_data[symbol] = [position]
            else:
                self.open_position_data[symbol].append(position)

        self.open_position_symbols = set(self.open_position_data.keys())

        self._process_positions_update()

    async def _fetch_ohlcvc(self, symbol, timeframe, limit=MAX_CANDLES):
        try:
            ohlcvc = await self.exchange.exchange_async.fetch_ohlcv(
                symbol=self._bybit_symbol(symbol),
                timeframe=timeframe,
                limit=limit,
                params={"paginate": True}
            )

            logging.info(f"Fetched OHLCVCS for symbol {symbol} on {timeframe}")

            return self.exchange.convert_ohlcv_to_df(ohlcvc)
        except Exception as e:
            logging.exception(e)
            return None

    async def _fetch_ohlcvcs(self, symbols):
        logging.info(f"Fetching new OHLCVCS for symbols: {symbols}")

        for timeframe in self.OHLCV_TIMEFRAMES:
            tasks = [
                self._fetch_ohlcvc(symbol, timeframe)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks)

            for symbol, ohlcvc in zip(symbols, results):
                if timeframe not in self.ohlcvcs:
                    self.ohlcvcs[timeframe] = {}
                self.ohlcvcs[timeframe][symbol] = ohlcvc

    async def _update_balance(self):
        logging.info("Fetching balance")

        balance = await self.exchange.get_balance_async()

        state.balance['available'], state.balance['total'] = balance
        state.balance["updated_at"] = self.exchange.exchange_async.milliseconds()

    async def _remove_dangling_orders(self):
        open_orders = await self.exchange.get_all_open_orders_async()

        for order in open_orders:
            symbol = order['info']['symbol']

            if symbol in self.config.bot.blacklist:
                logging.info(f"[{symbol}] skipping dangling order management due to blacklist")
                continue

            if order['info']['stopOrderType'] == 'TakeProfit' or order['info']['stopOrderType'] == 'PartialTakeProfit':
                logging.info(f"[{symbol}] skipping take profit order. Likely manual.")
                continue

            open_long_time_ago = order['timestamp'] < (datetime.now() - timedelta(minutes=5)).timestamp() * 1000

            # no active traders running for symbol OR open long time ago
            if symbol not in self.open_position_symbols and open_long_time_ago:
                logging.info(f"[{symbol}] found open order without position")
                running_long = symbol in self.traders['long'] and self.traders['long'][symbol].running_trading('long')
                running_short = symbol in self.traders['short'] and self.traders['short'][symbol].running_trading('short')

                if not open_long_time_ago and (running_long or running_short):
                    logging.info(f"[{symbol}] have running trader for order order")
                else:
                    await self.exchange.cancel_order_by_id_async(order['id'], symbol)
                    logging.info(f"[{symbol}] cancelled dangling open orders")

    def _cleanup_threads(self):
        for symbol in list(self.long_threads.keys()):
            if self.long_threads[symbol].done():
                self.long_threads.pop(symbol)

        for symbol in list(self.short_threads.keys()):
            if self.short_threads[symbol].done():
                self.short_threads.pop(symbol)

    # signal handler for graceful stop
    def _signal_handler(self, sig, frame):
        self.sigint_count += 1

        if self.sigint_count == 1:
            print(
                "SIGINT received. Stopping the bot gracefully. To force exit, send SIGINT again."
            )
            self.stop = True
        elif self.sigint_count == 2:
            logging.info("Second SIGINT received. Forcing exit...")
            os._exit(0)

    async def _periodic_position_update(self):
        while not self.stop:
            try:
                await self._update_position_data()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logging.exception(f"Error in periodic position update: {e}")
                await asyncio.sleep(5)  # On error, wait 5 seconds before retrying


async def main():
    sword = f"{Fore.CYAN}====={Fore.WHITE}||{Fore.RED}====>"

    print("\n" + Fore.YELLOW + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + f"DirectionalScalper {VERSION}".center(60))
    print(Fore.GREEN + f"Developed by Tyler Simpson and contributors at".center(60))
    print(Fore.GREEN + f"Quantum Void Labs".center(60))
    print(Fore.YELLOW + "=" * 60 + "\n")

    print(Fore.WHITE + "Initializing", end="")
    for i in range(3):
        time.sleep(0.5)
        print(Fore.YELLOW + ".", end="", flush=True)
    print("\n")

    print(Fore.MAGENTA + Style.BRIGHT + "Battle-Ready Algorithm".center(60))
    print(sword.center(73) + "\n")  # Adjusted for color codes

    # ASCII Art
    ascii_art = r"""
    ⚔️  Prepare for Algorithmic Conquest  ⚔️
      _____   _____   _____   _____   _____
     /     \ /     \ /     \ /     \ /     \
    /       V       V       V       V       \
   |     DirectionalScalper Activated     |
    \       ^       ^       ^       ^       /
     \_____/ \_____/ \_____/ \_____/ \_____/
    """

    print(Fore.CYAN + ascii_art)

    # Simulated loading bar
    # print(Fore.YELLOW + "Loading trading modules:")
    # for i in range(101):
    #     time.sleep(0.03)  # Adjust speed as needed
    #     print(f"\r[{'=' * (i // 2)}{' ' * (50 - i // 2)}] {i}%", end="", flush=True)
    # print("\n")

    print(Fore.GREEN + Style.BRIGHT + "DirectionalScalper is ready for battle!")
    print(Fore.RED + "May the odds be ever in your favor.\n")

    parser = argparse.ArgumentParser(description="DirectionalScalper")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--account_name", type=str, help="The name of the account to use"
    )
    parser.add_argument("--exchange", type=str, help="The name of the exchange to use")

    args = parser.parse_args()
    # args = ask_for_missing_arguments(args)

    print(f"DirectionalScalper {VERSION} Initialized Successfully!".center(50))
    print("=" * 50 + "\n")

    config_file_path = Path(args.config)
    account_path = Path("configs/account.json")

    try:
        config = load_config(config_file_path, account_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        logging.error(
            "There is probably an issue with your path try using --config configs/config.json"
        )
        os._exit(1)

    exchange_name = args.exchange

    bot = SingleBot(
        config=config, exchange_name=exchange_name, account_name=args.account_name
    )

    try:
        await bot.run()
    finally:
        await bot._cleanup()


if __name__ == "__main__":
    asyncio.run(main())
