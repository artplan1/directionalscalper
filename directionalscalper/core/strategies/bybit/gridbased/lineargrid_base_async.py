import time
import json
import os
import copy
import traceback
from datetime import datetime

from directionalscalper.core.config_initializer import ConfigInitializer
from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy
from directionalscalper.core.strategies.logger import Logger
from rate_limit import RateLimit
import state

logging = Logger(
    logger_name="LinearGridBaseFuturesAsync", filename="LinearGridBaseFuturesAsync.log", stream=True
)


class LinearGridBaseFuturesAsync(BybitStrategy):
    def __init__(
        self, exchange, manager, config, symbols_allowed=None, mfirsi_signal=None
    ):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.rate_limiter = RateLimit(10, 1)
        self.general_rate_limiter = RateLimit(50, 1)
        self.order_rate_limiter = RateLimit(5, 1)
        self.mfirsi_signal = mfirsi_signal
        self.is_order_history_populated = False
        self.last_health_check_time = time.time()
        self.health_check_interval = 600
        self.last_helper_order_cancel_time = 0
        self.helper_active = False
        self.helper_wall_size = 5
        self.helper_duration = 5
        self.helper_interval = 1
        self.running_long = None
        self.running_short = None
        self.last_known_equity = 0.0
        self.last_known_upnl = {}
        self.last_known_mas = {}
        self.previous_long_pos_qty = 0
        self.previous_short_pos_qty = 0
        self.iteration_cnt = 0
        self.min_qty = None
        self.price_precision = None

        ConfigInitializer.initialize_config_attributes(self, config)

    def run(self, symbol, open_symbols={}, mfirsi_signal=None, action=None):
        try:
            standardized_symbol = symbol.upper()
            logging.info(f"[{symbol}] Standardized symbol: {standardized_symbol}")

            if action == "long":
                self.run_long_trades(standardized_symbol, mfirsi_signal, open_symbols=open_symbols)
            elif action == "short":
                self.run_short_trades(standardized_symbol, mfirsi_signal, open_symbols=open_symbols)
        except Exception as e:
            logging.error(f"[{symbol}] Exception in run function: {e}")
            logging.debug(traceback.format_exc())

    def run_long_trades(self, symbol, mfirsi_signal=None, open_symbols={}):
        if self.running_long is None:
            self.running_long = True

        self.run_single_symbol(symbol, mfirsi_signal, "long", open_symbols=open_symbols)

    def run_short_trades(self, symbol, mfirsi_signal=None, open_symbols={}):
        if self.running_short is None:
            self.running_short = True

        self.run_single_symbol(symbol, mfirsi_signal, "short", open_symbols=open_symbols)

    async def configure_trader(self, symbol):
        logging.info(f"[{symbol}] Setting up exchange")
        await self.exchange.setup_exchange_async(symbol)

        # Check leverages only at startup
        self.current_leverage = self.max_leverage =  await self.exchange.get_current_max_leverage_async(symbol)

        logging.info(f"[{symbol}] Current leverage: {self.current_leverage}")

        logging.info(f"[{symbol}] Max leverage: {self.max_leverage}")

        await self.exchange.set_leverage_async(self.max_leverage, symbol)
        await self.exchange.set_symbol_to_cross_margin_async(symbol, self.max_leverage)

    EQUITY_REFRESH_INTERVAL = 30

    def run_single_symbol(
        self, symbol, mfirsi_signal=None, action=None, open_symbols={}
    ):
        self.iteration_cnt += 1

        try:
            logging.info(f"[{symbol}] Starting iteration: {self.iteration_cnt}")
            logging.info(f"[{symbol}] Initializing default values")

            five_minute_volume = None
            five_minute_distance = None
            ma_trend = "neutral"  # Initialize with default value
            ema_trend = "undefined"  # Initialize with default value
            long_upnl = 0
            short_upnl = 0

            fetched_total_equity = None

            if action == "long":
                logging.info(
                    f"[{symbol}] Trading in while loop in obstrategy with long: {self.running_long}"
                )

                if not self.running_long:
                    logging.info(
                        f"[{symbol}] Killing thread because not running long"
                    )
                    return


            if action == "short":
                logging.info(
                    f"[{symbol}] Trading in while loop in obstrategy with short: {self.running_short}"
                )

                if not self.running_short:
                    logging.info(
                        f"[{symbol}] Killing thread because not running short"
                    )
                    return

            iteration_start_time = time.time()

            # logging.info(f"Max USD value: {self.max_usd_value}")

            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if not self.min_qty or not self.price_precision:
                market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)

                self.min_qty = float(market_data["min_qty"] or 1)
                self.price_precision = float(market_data['precision'] or 1)

            if not self.min_qty or not self.price_precision:
                logging.warning(f"[{symbol}] market data is not loaded")
                return

            logging.info(f"[{symbol}] Symbol precision: {self.price_precision}")

            fetched_total_equity = state.balance.get("total")
            last_equity_fetch_time = state.balance.get("updated_at")

            if (
                not last_equity_fetch_time
                or time.time() - last_equity_fetch_time > self.EQUITY_REFRESH_INTERVAL
                or fetched_total_equity == 0.0
                or fetched_total_equity is None
            ):
                logging.info(f"[{symbol}] Fetching balance from API")
                # Fetch equity data
                fetched_total_equity = self.retry_api_call(
                    self.exchange.get_futures_balance_bybit
                )
                last_equity_fetch_time = time.time()

            logging.info(f"[{symbol}] Fetched total equity: {fetched_total_equity}")

            # Attempt to convert fetched_total_equity to a float
            try:
                fetched_total_equity = float(fetched_total_equity)
            except (ValueError, TypeError):
                logging.warning(
                    f"[{symbol}] Fetched total equity could not be converted to float: {fetched_total_equity}. Resorting to last known equity."
                )
                return

            # Refresh equity if interval passed or fetched equity is 0.0
            if not fetched_total_equity or fetched_total_equity == 0:
                logging.error(
                    f"[{symbol}] This should not happen as total_equity should never be None. Skipping this iteration."
                )
                return

            if fetched_total_equity is not None and fetched_total_equity > 0.0:
                total_equity = fetched_total_equity
                self.last_known_equity = (
                    total_equity  # Update the last known equity
                )
            else:
                logging.warning(
                    f"[{symbol}] Failed to fetch valid total_equity or received 0.0. Using last known value."
                )
                total_equity = self.last_known_equity  # Use last known equity

            available_equity = state.balance.get("available")

            logging.info(f"[{symbol}] Available equity from state: {available_equity}")

            if not available_equity:
                logging.warning(
                    f"[{symbol}] Failed to fetch valid available_equity. Fetching from API."
                )
                available_equity = self.retry_api_call(self.exchange.get_available_balance_bybit)

            logging.info(f"[{symbol}] Total equity: {total_equity}")
            logging.info(f"[{symbol}] Available equity: {available_equity}")

            if symbol in self.config.blacklist:
                logging.info(
                    f"[{symbol}] Symbol is in the blacklist. Stopping operations for this symbol."
                )
                return

            funding_check = self.is_funding_rate_acceptable(symbol)
            logging.info(f"[{symbol}] Funding check: {funding_check}")

            current_price = self.exchange.get_current_price(symbol)

            if not current_price:
                logging.warning(f"[{symbol}] Current price is not available.")
                return

            order_book = self.exchange.get_orderbook(symbol)

            # Handling best ask price with fallback mechanism
            if "asks" in order_book and len(order_book["asks"]) > 0:
                best_ask_price = order_book["asks"][0][0]
            else:
                logging.warning(f"[{symbol}] Best ask price is not available.")
                return

            # Convert best_ask_price to float to ensure no type mismatches
            best_ask_price = float(best_ask_price)
            logging.info(f"[{symbol}] Best ask price: {best_ask_price}")

            # Handling best bid price with fallback mechanism
            if "bids" in order_book and len(order_book["bids"]) > 0:
                best_bid_price = order_book["bids"][0][0]
            else:
                logging.warning(f"[{symbol}] Best bid price is not available.")
                return

            # Convert best_bid_price to float to ensure no type mismatches
            best_bid_price = float(best_bid_price)
            logging.info(f"[{symbol}] Best bid price: {best_bid_price}")

            # Fetch moving averages with fallback mechanism
            # try:
            #     moving_averages = self.get_all_moving_averages(symbol)
            # except ValueError as e:
            #     logging.info(
            #         f"[{symbol}] Failed to get new moving averages: {e}"
            #     )
            #     return

            # Ensure the moving averages are valid, fallback to last known if not present
            # ma_3_high = moving_averages.get(
            #     "ma_3_high",
            #     self.last_known_mas.get(symbol, {}).get("ma_3_high", 0.0),
            # )
            # ma_3_low = moving_averages.get(
            #     "ma_3_low", self.last_known_mas.get(symbol, {}).get("ma_3_low", 0.0)
            # )
            # ma_6_high = moving_averages.get(
            #     "ma_6_high",
            #     self.last_known_mas.get(symbol, {}).get("ma_6_high", 0.0),
            # )
            # ma_6_low = moving_averages.get(
            #     "ma_6_low", self.last_known_mas.get(symbol, {}).get("ma_6_low", 0.0)
            # )

            # Convert moving averages to float to ensure no type mismatches
            # ma_3_high = float(ma_3_high)
            # ma_3_low = float(ma_3_low)
            # ma_6_high = float(ma_6_high)
            # ma_6_low = float(ma_6_low)

            # Log warnings if any of the moving averages are missing
            # if None in [ma_3_high, ma_3_low, ma_6_high, ma_6_low]:
            #     logging.info(
            #         f"[{symbol}] Missing moving averages. Using fallback values."
            #     )

            logging.info(f"Symbols allowed: {self.symbols_allowed}")

            trading_allowed = self.can_trade_new_symbol(
                list(open_symbols), self.symbols_allowed, symbol
            )
            logging.info(
                f"[{symbol}] Checking trading. Can trade: {trading_allowed}"
            )
            logging.info(
                f"[{symbol}] Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}"
            )


            position_data = self.retry_api_call(self.exchange.get_positions_bybit, symbol)

            long_pos_qty = position_data.get("long", {}).get("qty", 0)
            short_pos_qty = position_data.get("short", {}).get("qty", 0)

            # Position side for symbol recently closed
            logging.info(f"[{symbol}] Current long pos qty: {long_pos_qty}")
            logging.info(f"[{symbol}] Current short pos qty: {short_pos_qty}")

            if self.previous_long_pos_qty > 0 and long_pos_qty == 0:
                logging.info(
                    f"[{symbol}] Long position closed. Canceling long grid orders."
                )
                self.cancel_grid_orders(symbol, "buy")
                self.active_long_grids.discard(symbol)
                if short_pos_qty == 0:
                    logging.info(
                        f"[{symbol}] No open positions. Removing from shared symbols data."
                    )
                    state.shared_symbols_data.pop(symbol, None)
                self.running_long = False
                self.previous_long_pos_qty = 0
                return
            elif self.previous_short_pos_qty > 0 and short_pos_qty == 0:
                logging.info(
                    f"[{symbol}] Short position closed. Canceling short grid orders."
                )
                self.cancel_grid_orders(symbol, "sell")
                self.active_short_grids.discard(symbol)
                if long_pos_qty == 0:
                    logging.info(
                        f"[{symbol}] No open positions. Removing from shared symbols data."
                    )
                    state.shared_symbols_data.pop(symbol, None)
                self.running_short = False
                self.previous_short_pos_qty = 0
                return

            try:
                logging.info(f"[{symbol}] Checking position inactivity")
                # Check for position inactivity
                inactive_pos_time_threshold = 60
                if self.check_position_inactivity(
                    symbol,
                    inactive_pos_time_threshold,
                    long_pos_qty,
                    short_pos_qty,
                    previous_long_pos_qty=self.previous_long_pos_qty,
                    previous_short_pos_qty=self.previous_short_pos_qty,
                ):
                    logging.info(
                        f"[{symbol}] No open positions in the last {inactive_pos_time_threshold} seconds. Terminating the thread."
                    )
                    state.shared_symbols_data.pop(symbol, None)
                    return
            except Exception as e:
                logging.info(f"[{symbol}] Exception caught in check_position_inactivity {e}")

            # Optionally, break out of the loop if all trading sides are closed
            if not self.running_long and not self.running_short:
                state.shared_symbols_data.pop(symbol, None)
                self.cancel_grid_orders(symbol, "buy")
                self.cancel_grid_orders(symbol, "sell")
                self.active_long_grids.discard(symbol)
                self.active_short_grids.discard(symbol)
                self.cleanup_before_termination(symbol)
                logging.info(
                    f"[{symbol}] Both long and short operations have terminated. Exiting the loop."
                )
                return

            # Determine if positions have just been closed
            if self.previous_long_pos_qty > 0 and long_pos_qty == 0:
                logging.info(
                    f"[{symbol}] All long positions were recently closed. Checking for inactivity."
                )

            if self.previous_short_pos_qty > 0 and short_pos_qty == 0:
                logging.info(
                    f"[{symbol}] All short positions were recently closed. Checking for inactivity."
                )

            # Update previous quantities for the next iteration
            if self.running_long:
                self.previous_long_pos_qty = long_pos_qty

            if self.running_short:
                self.previous_short_pos_qty = short_pos_qty

            if not self.running_long and not self.running_short:
                logging.info(
                    f"[{symbol}] Both long and short operations have ended. Preparing to exit loop."
                )
                state.shared_symbols_data.pop(
                    symbol, None
                )  # Remove the symbol from shared_symbols_data
                return

            # If the symbol is in rotator_symbols and either it's already being traded or trading is allowed.
            if symbol in open_symbols or trading_allowed:  # and instead of or
                # Fetch the API data
                api_data = self.manager.get_api_data(symbol)

                # Extract the required metrics using the new implementation
                metrics = self.manager.extract_metrics(api_data, symbol)

                # Assign the metrics to the respective variables
                five_minute_volume = metrics["5mVol"]
                five_minute_distance = metrics["5mSpread"]
                ma_trend = metrics["MA Trend"]
                ema_trend = metrics["EMA Trend"]

                eri_trend = metrics["ERI Trend"]

                logging.info(f"[{symbol}] ERI Trend: {eri_trend}")

                logging.info(f"[{symbol}] metrics: {metrics}")

                # mfirsi_signal_in_strat = self.exchange.generate_l_signals(symbol)

                # logging.info(f"[{symbol}] MFIRSI Signal in strat: {mfirsi_signal_in_strat}")

                # mfirsi_signal = mfirsi_signal_in_strat

                logging.info(f"[{symbol}] MFIRSI Signal: {mfirsi_signal}")

                long_liquidation_price = position_data.get("long", {}).get("liq_price")
                short_liquidation_price = position_data.get("short", {}).get("liq_price")

                logging.info(
                    f"[{symbol}] Long liquidation price: {long_liquidation_price}"
                )
                logging.info(
                    f"[{symbol}] Short liquidation price: {short_liquidation_price}"
                )

                logging.info(f"Rotator symbol trading: {symbol}")
                logging.info(f"Open symbols: {open_symbols}")

                logging.info(f"[{symbol}] Long pos qty: {long_pos_qty}")
                logging.info(f"[{symbol}] Short pos qty: {short_pos_qty}")

                # Calculate dynamic entry sizes for long and short positions
                long_dynamic_amount, short_dynamic_amount = (
                    self.calculate_dynamic_amounts_notional(
                        symbol=symbol,
                        total_equity=total_equity,
                        best_ask_price=best_ask_price,
                        best_bid_price=best_bid_price,
                        wallet_exposure_limit_long=self.wallet_exposure_limit_long,
                        wallet_exposure_limit_short=self.wallet_exposure_limit_short,
                        min_qty=self.min_qty,
                        price_precision=self.price_precision
                    )
                )

                logging.info(
                    f"[{symbol}] Long dynamic amount: {long_dynamic_amount}"
                )
                logging.info(
                    f"[{symbol}] Short dynamic amount: {short_dynamic_amount}"
                )

                long_dynamic_amount_helper, short_dynamic_amount_helper = (
                    self.calculate_dynamic_amounts_notional_nowelimit(
                        symbol=symbol,
                        total_equity=total_equity,
                        best_bid_price=best_bid_price,
                        best_ask_price=best_ask_price,
                        min_qty=self.min_qty,
                        price_precision=self.price_precision
                    )
                )

                logging.info(
                    f"[{symbol}] Long dynamic amount helper: {long_dynamic_amount_helper}"
                )
                logging.info(
                    f"[{symbol}] Short dynamic amount helper: {short_dynamic_amount_helper}"
                )

                cum_realised_pnl_long = position_data["long"]["cum_realised"]
                cum_realised_pnl_short = position_data["short"]["cum_realised"]

                # Get the average price for long and short positions
                long_pos_price = position_data.get("long", {}).get("price", None)
                short_pos_price = position_data.get("short", {}).get("price", None)

                if self.failsafe_enabled:
                    try:
                        self.failsafe_method_leveraged(
                            symbol,
                            long_pos_qty,
                            short_pos_qty,
                            long_pos_price,
                            short_pos_price,
                            long_upnl,
                            short_upnl,
                            total_equity,
                            current_price,
                            self.failsafe_enabled,
                            self.long_failsafe_upnl_pct,
                            self.short_failsafe_upnl_pct,
                            self.failsafe_start_pct,
                        )
                    except Exception as e:
                        logging.info(f"[{symbol}] Failsafe failed: {e}")

                if self.auto_reduce_enabled:
                    try:
                        self.auto_reduce_logic_grid_hardened_cooldown(
                            symbol,
                            self.min_qty,
                            long_pos_price,
                            short_pos_price,
                            long_pos_qty,
                            short_pos_qty,
                            long_upnl,
                            short_upnl,
                            self.auto_reduce_cooldown_enabled,
                            total_equity,
                            current_price,
                            long_dynamic_amount,
                            short_dynamic_amount,
                            self.auto_reduce_cooldown_start_pct,
                            self.min_buffer_percentage_ar,
                            self.max_buffer_percentage_ar,
                            self.upnl_auto_reduce_threshold_long,
                            self.upnl_auto_reduce_threshold_short,
                            self.current_leverage,
                        )
                    except Exception as e:
                        logging.info(f"[{symbol}] Hardened grid AR exception caught {e}")

                    try:
                        self.auto_reduce_logic_grid_hardened(
                            symbol,
                            self.min_qty,
                            long_pos_price,
                            short_pos_price,
                            long_pos_qty,
                            short_pos_qty,
                            long_upnl,
                            short_upnl,
                            self.auto_reduce_enabled,
                            total_equity,
                            current_price,
                            long_dynamic_amount,
                            short_dynamic_amount,
                            self.auto_reduce_start_pct,
                            self.min_buffer_percentage_ar,
                            self.max_buffer_percentage_ar,
                            self.upnl_auto_reduce_threshold_long,
                            self.upnl_auto_reduce_threshold_short,
                            self.current_leverage,
                        )
                    except Exception as e:
                        logging.info(f"[{symbol}] Hardened grid AR exception caught {e}")

                if self.percentile_auto_reduce_enabled:
                    self.auto_reduce_percentile_logic(
                        symbol,
                        long_pos_qty,
                        long_pos_price,
                        short_pos_qty,
                        short_pos_price,
                        self.percentile_auto_reduce_enabled,
                        self.auto_reduce_start_pct,
                        self.auto_reduce_maxloss_pct,
                        long_dynamic_amount,
                        short_dynamic_amount,
                    )

                if self.liq_stoploss_enabled:
                    self.liq_stop_loss_logic(
                        long_pos_qty,
                        long_pos_price,
                        long_liquidation_price,
                        short_pos_qty,
                        short_pos_price,
                        short_liquidation_price,
                        liq_stoploss_enabled=self.liq_stoploss_enabled,
                        symbol=symbol,
                        liq_price_stop_pct=self.liq_price_stop_pct,
                    )

                # NOTE: stop losses are controlled in lineargrid_base
                # self.stop_loss_logic(
                #     long_pos_qty,
                #     long_pos_price,
                #     short_pos_qty,
                #     short_pos_price,
                #     stoploss_enabled=self.config.stop_loss_enabled,
                #     symbol=symbol,
                #     stoploss_upnl_pct=self.stoploss_upnl_pct,
                # )

                if self.auto_reduce_marginbased_enabled:
                    self.auto_reduce_marginbased_logic(
                        self.auto_reduce_marginbased_enabled,
                        long_pos_qty,
                        short_pos_qty,
                        long_pos_price,
                        short_pos_price,
                        symbol,
                        total_equity,
                        self.auto_reduce_wallet_exposure_pct,
                        open_position_data=None, # it's fetched inside auto_reduce_marginbased_logic
                        current_market_price=current_price,
                        long_dynamic_amount=long_dynamic_amount,
                        short_dynamic_amount=short_dynamic_amount,
                        auto_reduce_start_pct=self.auto_reduce_start_pct,
                        auto_reduce_maxloss_pct=self.auto_reduce_maxloss_pct,
                    )

                if self.auto_reduce_enabled or self.auto_reduce_marginbased_enabled or self.percentile_auto_reduce_enabled:
                    self.cancel_auto_reduce_orders_bybit(
                        symbol,
                        total_equity,
                        self.max_pos_balance_pct,
                        open_position_data=[], # balances are passed separately, no need for all positions
                        long_pos_qty=long_pos_qty,
                        short_pos_qty=short_pos_qty,
                        long_position_balance=position_data["long"]["balance"],
                        short_position_balance=position_data["short"]["balance"]
                    )

                logging.info(
                    f"[{symbol}] Five minute volume: {five_minute_volume}"
                )

                tp_order_counts = self.exchange.get_open_tp_order_count(open_orders)

                logging.info(f"[{symbol}] Open TP order count: {tp_order_counts}")

                # Check for long position
                if long_pos_qty > 0:
                    try:
                        unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                        long_upnl = unrealized_pnl.get("long")
                        self.last_known_upnl[symbol] = self.last_known_upnl.get(symbol, {})
                        self.last_known_upnl[symbol]["long"] = long_upnl  # Store the last known long uPNL
                        logging.info(f"[{symbol}] Long UPNL: {long_upnl}")
                    except Exception as e:
                        # Fallback to last known uPNL if an exception occurs
                        long_upnl = self.last_known_upnl.get(symbol, {}).get("long", 0.0)
                        logging.info(
                            f"[{symbol}] Exception fetching Long UPNL: {e}. Using last known UPNL: {long_upnl}"
                        )

                # Check for short position
                if short_pos_qty > 0:
                    try:
                        unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                        short_upnl = unrealized_pnl.get("short")
                        self.last_known_upnl[symbol] = self.last_known_upnl.get(
                            symbol, {}
                        )
                        self.last_known_upnl[symbol][
                            "short"
                        ] = short_upnl  # Store the last known short uPNL
                        logging.info(f"[{symbol}] Short UPNL: {short_upnl}")
                    except Exception as e:
                        # Fallback to last known uPNL if an exception occurs
                        short_upnl = self.last_known_upnl.get(symbol, {}).get(
                            "short", 0.0
                        )
                        logging.info(
                            f"[{symbol}] Exception fetching Short UPNL: {e}. Using last known UPNL: {short_upnl}"
                        )

                long_tp_counts = tp_order_counts["long_tp_count"]
                short_tp_counts = tp_order_counts["short_tp_count"]

                symbol_data = {
                    "symbol": symbol,
                    "min_qty": self.min_qty,
                    "current_price": current_price,
                    "balance": total_equity,
                    "available_bal": available_equity,
                    "volume": five_minute_volume,
                    "spread": five_minute_distance,
                    "ma_trend": ma_trend,
                    "ema_trend": ema_trend,
                    "long_pos_qty": long_pos_qty,
                    "short_pos_qty": short_pos_qty,
                    "long_upnl": long_upnl,
                    "short_upnl": short_upnl,
                    "long_cum_pnl": cum_realised_pnl_long,
                    "short_cum_pnl": cum_realised_pnl_short,
                    "long_pos_price": long_pos_price,
                    "short_pos_price": short_pos_price,
                }

                self.update_shared_data(symbol, symbol_data)

                start_linear = time.time()
                logging.info(f"[{symbol}] started lineargrid_base")
                try:
                    self.lineargrid_base(
                        symbol,
                        list(open_symbols),
                        total_equity,
                        long_pos_price,
                        short_pos_price,
                        long_pos_qty,
                        short_pos_qty,
                        self.levels,
                        self.strength,
                        self.min_outer_price_distance,
                        self.max_outer_price_distance_long,
                        self.max_outer_price_distance_short,
                        self.reissue_threshold,
                        self.wallet_exposure_limit_long,
                        self.wallet_exposure_limit_short,
                        self.long_mode,
                        self.short_mode,
                        self.initial_entry_buffer_pct,
                        self.min_buffer_percentage,
                        self.max_buffer_percentage,
                        self.symbols_allowed,
                        self.enforce_full_grid,
                        mfirsi_signal,
                        self.upnl_profit_pct,
                        self.max_upnl_profit_pct,
                        tp_order_counts,
                        self.entry_during_autoreduce,
                        self.max_qty_percent_long,
                        self.max_qty_percent_short,
                        self.graceful_stop_long,
                        self.graceful_stop_short,
                        self.additional_entries_from_signal,
                        open_position_data=[],
                        drawdown_behavior=self.drawdown_behavior,
                        grid_behavior=self.grid_behavior,
                        stop_loss_long=self.stop_loss_long,
                        stop_loss_short=self.stop_loss_short,
                        stop_loss_enabled=self.stop_loss_enabled,
                        price_precision=self.price_precision,
                        min_qty=self.min_qty
                    )
                except Exception as e:
                    logging.info(f"[{symbol}] Something is up with variables for the grid: {e}")

                end_linear = time.time()
                logging.info(f"[{symbol}] ended lineargrid_base in {end_linear-start_linear} seconds")

                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                tp_order_counts = self.exchange.get_open_tp_order_count(open_orders)

                logging.info(f"[{symbol}] Open TP order count: {tp_order_counts}")

                long_tp_counts = tp_order_counts["long_tp_count"]
                short_tp_counts = tp_order_counts["short_tp_count"]

                logging.info(f"[{symbol}] Long tp counts: {long_tp_counts}")
                logging.info(f"[{symbol}] Short tp counts: {short_tp_counts}")

                position_data = self.retry_api_call(self.exchange.get_positions_bybit, symbol)

                long_pos_qty = position_data.get("long", {}).get("qty", 0)
                short_pos_qty = position_data.get("short", {}).get("qty", 0)

                logging.info(f"[{symbol}] Long pos qty: {long_pos_qty}")
                logging.info(f"[{symbol}] Short pos qty: {short_pos_qty}")

                current_latest_time = datetime.now()
                logging.info(f"[{symbol}] Current time: {current_latest_time}")

                # Update TP for long position
                if long_pos_qty > 0:
                    logging.info(
                        f"[{symbol}] Next long TP update time: {self.next_long_tp_update}"
                    )

                    long_pos_price = position_data.get("long", {}).get("price", None)

                    new_long_tp_min, new_long_tp_max = (
                        self.calculate_quickscalp_long_take_profit_dynamic_distance(
                            long_pos_price,
                            symbol,
                            self.upnl_profit_pct,
                            self.max_upnl_profit_pct,
                        )
                    )

                    logging.info(f"[{symbol}] Long take profit - min: {new_long_tp_min} max: {new_long_tp_max}")

                    if new_long_tp_min is not None and new_long_tp_max is not None:
                        self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                            symbol=symbol,
                            pos_qty=long_pos_qty,
                            upnl_profit_pct=self.upnl_profit_pct,  # Minimum desired profit percentage
                            max_upnl_profit_pct=self.max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                            short_pos_price=None,  # Not relevant for long TP settings
                            long_pos_price=long_pos_price,
                            positionIdx=1,
                            order_side="sell",
                            last_tp_update=self.next_long_tp_update,
                            tp_order_counts=tp_order_counts,
                            open_orders=open_orders,
                        )

                if short_pos_qty > 0:
                    logging.info(
                        f"[{symbol}] Next short TP update time: {self.next_short_tp_update}"
                    )

                    short_pos_price = position_data.get("short", {}).get("price", None)

                    new_short_tp_min, new_short_tp_max = (
                        self.calculate_quickscalp_short_take_profit_dynamic_distance(
                            short_pos_price,
                            symbol,
                            self.upnl_profit_pct,
                            self.max_upnl_profit_pct,
                        )
                    )

                    logging.info(f"[{symbol}] Short take profit - min: {new_short_tp_min} max: {new_short_tp_max}")

                    if (
                        new_short_tp_min is not None
                        and new_short_tp_max is not None
                    ):
                        self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                            symbol=symbol,
                            pos_qty=short_pos_qty,
                            upnl_profit_pct=self.upnl_profit_pct,  # Minimum desired profit percentage
                            max_upnl_profit_pct=self.max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                            short_pos_price=short_pos_price,
                            long_pos_price=None,  # Not relevant for short TP settings
                            positionIdx=2,
                            order_side="buy",
                            last_tp_update=self.next_short_tp_update,
                            tp_order_counts=tp_order_counts,
                            open_orders=open_orders,
                        )

                if self.test_orders_enabled and time.time() - self.last_helper_order_cancel_time >= self.helper_interval:
                    if symbol in open_symbols:
                        self.helper_active = True
                        self.helperv2(symbol, short_dynamic_amount_helper, long_dynamic_amount_helper)
                    else:
                        logging.info(f"[{symbol}] Skipping test orders as it's not in open symbols list.")

                # Check to terminate the loop if both long and short are no longer running
                if not self.running_long and not self.running_short:
                    logging.info(
                        f"[{symbol}] Both long and short operations have ended. Preparing to exit loop."
                    )
                    state.shared_symbols_data.pop(symbol, None)  # Remove the symbol from shared symbols data

            if self.config.dashboard_enabled:
                dashboard_path = os.path.join(
                    self.config.shared_data_path, "shared_data.json"
                )

                try:
                    data_to_save = copy.deepcopy(state.shared_symbols_data)
                    with open(dashboard_path, "w") as f:
                        json.dump(data_to_save, f)
                except Exception as e:
                    logging.info(f"Dashboard saving is not working properly {e}")

                try:
                    dashboard_path = os.path.join(
                        self.config.shared_data_path, "shared_data.json"
                    )
                    logging.info(f"Dashboard path: {dashboard_path}")

                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
                    logging.info(
                        f"Directory created: {os.path.dirname(dashboard_path)}"
                    )

                    if os.path.exists(dashboard_path):
                        with open(dashboard_path, "r") as file:
                            # Read or process file data
                            data = json.load(file)
                            logging.info(
                                "Loaded existing data from shared_data.json"
                            )
                    else:
                        logging.warning(
                            "shared_data.json does not exist. Creating a new file."
                        )
                        data = {}  # Initialize data as an empty dictionary

                    # Save the updated data to the JSON file
                    with open(dashboard_path, "w") as file:
                        json.dump(data, file)
                        logging.info("Data saved to shared_data.json")

                except FileNotFoundError:
                    logging.info(f"File not found: {dashboard_path}")
                    # Handle the absence of the file, e.g., by creating it or using default data
                except IOError as e:
                    logging.info(f"I/O error occurred: {e}")
                    # Handle other I/O errors
                except Exception as e:
                    logging.info(
                        f"An unexpected error occurred in saving json: {e}"
                    )

            iteration_end_time = time.time()  # Record the end time of the iteration
            iteration_duration = iteration_end_time - iteration_start_time
            logging.info(f"[{symbol}] Iteration took {iteration_duration:.2f} seconds")
        except Exception as e:
            traceback_info = traceback.format_exc()  # Get the full traceback
            logging.info(
                f"[{symbol}] Exception caught in quickscalp strategy: {e}\nTraceback:\n{traceback_info}"
            )

    def update_shared_data(self, symbol, symbol_data):
        state.shared_symbols_data[symbol] = symbol_data
        return

    def running_trading(self, action):
        return getattr(self, f"running_{action}", False)
