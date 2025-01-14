import asyncio
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import inspect
import math
import time
import pandas as pd
import ta as ta
import logging
import traceback

from directionalscalper.core.strategies.base_strategy_async import BaseStrategyAsync
from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy
from ..logger import Logger


from directionalscalper.core.exchanges.bybit_async import BybitExchangeAsync

from rate_limit import RateLimitAsync

logging = Logger(logger_name="BybitBaseStrategyAsync", filename="BybitBaseStrategyAsync.log", stream=True)

grid_lock_async = asyncio.Lock()


class BybitStrategyAsync(BybitStrategy, BaseStrategyAsync):
    def __init__(self, exchange: 'BybitExchangeAsync', config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)

        self.general_rate_limiter_async = RateLimitAsync(50, 1)

    async def get_market_data_with_retry_async(self, symbol, max_retries=5, retry_delay=5):
        for i in range(max_retries):
            try:
                async with self.general_rate_limiter_async:
                    return await self.exchange.get_market_data_async(symbol)
            except Exception as e:
                if i < max_retries - 1:
                    logging.info(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
                    logging.info(f"Call Stack: {traceback.format_exc()}")
                    await asyncio.sleep(retry_delay)
                else:
                    #logging.info("Excessive call stack:\n" + "".join(traceback.format_stack()))
                    logging.info(f"get_market_data_with_retry failure from bybit_strategy.py")
                    raise e

    async def cancel_grid_orders_async(self, symbol: str, side: str):
        try:
            open_orders = await self.retry_api_call_async(self.exchange.get_open_orders_async, symbol)

            orders_canceled = 0
            for order in open_orders:
                if order['side'].lower() == side.lower():
                    await self.exchange.cancel_order_by_id_async(order['id'], symbol)
                    orders_canceled += 1
                    logging.info(f"Canceled order {order['id']} for {symbol}")

            if orders_canceled > 0:
                logging.info(f"Canceled {orders_canceled} {side} grid orders for {symbol}")
            else:
                logging.info(f"No {side} grid orders found to cancel for {symbol}")

            # Remove the symbol from active grids
            if side.lower() == 'buy' and orders_canceled > 0:
                self.active_long_grids.discard(symbol)
                logging.info(f"Removed {symbol} from active long grids")
            elif side.lower() == 'sell' and orders_canceled > 0:
                self.active_short_grids.discard(symbol)
                logging.info(f"Removed {symbol} from active short grids")

        except Exception as e:
            logging.error(f"Exception in cancel_grid_orders for {symbol} - {side}: {e}")
            logging.error("Traceback: %s", traceback.format_exc())

    async def handle_inactivity_async(self, symbol, side, current_time, inactive_time_threshold, previous_qty):
        if symbol in self.last_activity_time:
            last_activity_time = self.last_activity_time[symbol]
            inactive_time = current_time - last_activity_time
            logging.info(f"{symbol} ({side}) last active {inactive_time} seconds ago")
            if inactive_time >= inactive_time_threshold and previous_qty > 0:
                logging.info(f"{symbol} ({side}) has been inactive for {inactive_time} seconds, exceeding threshold of {inactive_time_threshold} seconds")
                if side == 'long':
                    await self.cancel_grid_orders_async(symbol, 'buy')
                    self.active_long_grids.discard(symbol)
                    self.running_long = False
                elif side == 'short':
                    await self.cancel_grid_orders_async(symbol, 'sell')
                    self.active_short_grids.discard(symbol)
                    self.running_short = False
                return True
        else:
            self.last_activity_time[symbol] = current_time
            logging.info(f"Recording initial activity time for {symbol} ({side})")
        return False

    async def check_position_inactivity_async(self, symbol, inactive_pos_time_threshold, long_pos_qty, short_pos_qty, previous_long_pos_qty, previous_short_pos_qty):
        current_time = time.time()
        logging.info(f"Checking position inactivity for {symbol} at {current_time}")

        has_open_long_position = long_pos_qty > 0
        has_open_short_position = short_pos_qty > 0

        logging.info(f"Open positions status for {symbol} - Long: {'found' if has_open_long_position else 'none'}, Short: {'found' if has_open_short_position else 'none'}")

        # Determine inactivity and handle accordingly
        if not has_open_long_position:
            if await self.handle_inactivity_async(symbol, 'long', current_time, inactive_pos_time_threshold, previous_long_pos_qty):
                return True

        if not has_open_short_position:
            if await self.handle_inactivity_async(symbol, 'short', current_time, inactive_pos_time_threshold, previous_short_pos_qty):
                return True

        return False

    async def cleanup_before_termination_async(self, symbol):
        # Cancel all orders for the symbol and perform any other cleanup needed
        await self.exchange.cancel_all_orders_for_symbol_async(symbol)

    async def calculate_dynamic_amounts_notional_async(self, symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit_long, wallet_exposure_limit_short, max_retries=20):
        market_data = await self.get_market_data_with_retry_async(symbol, max_retries=max_retries, retry_delay=5)

        # Log the market data for debugging
        logging.info(f"Market data for {symbol}: {market_data}")

        try:
            min_qty = float(market_data.get('min_qty', 1.0))  # Use min_qty directly
            qty_precision = float(market_data.get('precision', 1))  # Use precision if available

            # Determine the minimum notional value based on the symbol
            if symbol in ["BTCUSDT", "BTC-PERP"]:
                min_notional_value = 100  # Minimum value in USDT
            elif symbol in ["ETHUSDT", "ETH-PERP"] or "BTC" in symbol or "ETH" in symbol:
                min_notional_value = 20  # Minimum value in USDT
            else:
                min_notional_value = 5  # Minimum value in USDT for other contracts

            # Ensure values are non-zero and valid
            if total_equity <= 0 or best_ask_price <= 0 or best_bid_price <= 0:
                logging.warning(f"Invalid values detected: total_equity={total_equity}, best_ask_price={best_ask_price}, best_bid_price={best_bid_price}")
                return 0, 0  # Return zero values for long and short entry sizes

            # Calculate dynamic entry sizes based on risk parameters
            max_equity_for_long_trade = total_equity * wallet_exposure_limit_long
            long_entry_size = max(max_equity_for_long_trade / best_ask_price, min_qty)

            max_equity_for_short_trade = total_equity * wallet_exposure_limit_short
            short_entry_size = max(max_equity_for_short_trade / best_bid_price, min_qty)

            # Ensure the adjusted entry sizes meet the minimum notional value requirement
            long_notional = long_entry_size * best_ask_price
            short_notional = short_entry_size * best_bid_price

            if long_notional < min_notional_value:
                long_entry_size = min_notional_value / best_ask_price

            if short_notional < min_notional_value:
                short_entry_size = min_notional_value / best_bid_price

            # Adjust for precision
            long_entry_size = round(long_entry_size / qty_precision) * qty_precision
            short_entry_size = round(short_entry_size / qty_precision) * qty_precision

            # Ensure quantities respect the minimum quantity requirement
            long_entry_size = max(min_qty, round(long_entry_size))
            short_entry_size = max(min_qty, round(short_entry_size))

            logging.info(f"Calculated long entry size for {symbol}: {long_entry_size} units")
            logging.info(f"Calculated short entry size for {symbol}: {short_entry_size} units")

            return long_entry_size, short_entry_size
        except (TypeError, ValueError) as e:
            logging.error(f"Error occurred: {e}")
            return 0, 0

    async def calculate_dynamic_amounts_notional_nowelimit_async(self, symbol, total_equity, best_ask_price, best_bid_price, max_retries=20):
        market_data = await self.get_market_data_with_retry_async(symbol, max_retries=max_retries, retry_delay=5)

        # Log the market data for debugging
        logging.info(f"Market data for {symbol}: {market_data}")

        try:
            min_qty = float(market_data.get('min_qty', 1.0))  # Use min_qty directly
            qty_precision = float(market_data.get('precision', 1))  # Use precision if available

            # Determine the minimum notional value based on the symbol
            if symbol in ["BTCUSDT", "BTC-PERP"]:
                min_notional_value = 100  # Minimum value in USDT
            elif symbol in ["ETHUSDT", "ETH-PERP"] or "BTC" in symbol or "ETH" in symbol:
                min_notional_value = 20  # Minimum value in USDT
            else:
                min_notional_value = 5  # Minimum value in USDT for other contracts

            # Ensure values are non-zero and valid
            if total_equity <= 0 or best_ask_price <= 0 or best_bid_price <= 0:
                logging.warning(f"Invalid values detected: total_equity={total_equity}, best_ask_price={best_ask_price}, best_bid_price={best_bid_price}")
                return 0, 0  # Return zero values for long and short entry sizes

            # Calculate dynamic entry sizes without amplifying by wallet exposure limit
            long_entry_size = max(total_equity / best_ask_price, min_qty)
            short_entry_size = max(total_equity / best_bid_price, min_qty)

            # Ensure the adjusted entry sizes meet the minimum notional value requirement
            long_notional = long_entry_size * best_ask_price
            short_notional = short_entry_size * best_bid_price

            if long_notional < min_notional_value:
                long_entry_size = min_notional_value / best_ask_price

            if short_notional < min_notional_value:
                short_entry_size = min_notional_value / best_bid_price

            # Adjust for precision
            long_entry_size = round(long_entry_size / qty_precision) * qty_precision
            short_entry_size = round(short_entry_size / qty_precision) * qty_precision

            # Ensure quantities respect the minimum quantity requirement
            long_entry_size = max(min_qty, round(long_entry_size))
            short_entry_size = max(min_qty, round(short_entry_size))

            logging.info(f"Calculated long entry size for {symbol}: {long_entry_size} units")
            logging.info(f"Calculated short entry size for {symbol}: {short_entry_size} units")

            return long_entry_size, short_entry_size
        except (TypeError, ValueError) as e:
            logging.error(f"Error occurred: {e}")
            return 0, 0

    async def update_quickscalp_tp_dynamic_async(self, symbol, pos_qty, upnl_profit_pct, max_upnl_profit_pct, short_pos_price, long_pos_price, positionIdx, order_side, last_tp_update, tp_order_counts, open_orders):
        # Fetch the current open TP orders and TP order counts for the symbol
        long_tp_orders, short_tp_orders = self.exchange.get_open_tp_orders(open_orders)
        long_tp_count = tp_order_counts['long_tp_count']
        short_tp_count = tp_order_counts['short_tp_count']

        # Determine the minimum notional value for dynamic scaling
        min_notional_value = self.min_notional(symbol)
        current_price = await self.exchange.get_current_price_async(symbol)

        # Calculate the position's market value
        position_market_value = pos_qty * current_price

        # Calculate the dynamic TP range based on how many minimum notional units fit in the position's market value
        num_units = position_market_value / min_notional_value

        # Modify scaling factor calculation using logarithmic scaling for a smoother increase
        scaling_factor = math.log10(num_units + 1)  # Logarithmic scaling to smooth out the scaling progression

        # Calculate scaled TP percentage within the defined range
        scaled_tp_pct = upnl_profit_pct + (max_upnl_profit_pct - upnl_profit_pct) * min(scaling_factor, 1)  # Cap scaling at 100% to avoid excessive TP targets

        # Calculate the new TP values using the quickscalp method
        new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, upnl_profit_pct, scaled_tp_pct)
        new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, upnl_profit_pct, scaled_tp_pct)

        # Determine the relevant TP orders based on the order side
        relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders

        # Check if there's an existing TP order with a mismatched quantity
        mismatched_qty_orders = [order for order in relevant_tp_orders if order['qty'] != pos_qty and order['id'] not in self.auto_reduce_order_ids.get(symbol, [])]

        # Cancel mismatched TP orders if any
        for order in mismatched_qty_orders:
            try:
                await self.exchange.cancel_order_by_id_async(order['id'], symbol)
                logging.info(f"Cancelled TP order {order['id']} for update.")
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

        # Using datetime.now() for checking if update is needed
        now = datetime.now()
        if now >= last_tp_update or mismatched_qty_orders:
            # Check if a TP order already exists
            tp_order_exists = (order_side == "sell" and long_tp_count > 0) or (order_side == "buy" and short_tp_count > 0)

            # Set new TP order with updated prices only if no TP order exists
            if not tp_order_exists:
                new_tp_price_min = new_long_tp_min if order_side == "sell" else new_short_tp_min
                logging.info(f"New tp price min: {new_tp_price_min} with order side as {order_side}")
                new_tp_price_max = new_long_tp_max if order_side == "sell" else new_short_tp_max
                current_price = await self.exchange.get_current_price_async(symbol)
                order_book = await self.exchange.get_orderbook_async(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                # Ensure TP setting checks are correct for direction
                if (order_side == "sell" and current_price >= new_tp_price_min) or (order_side == "buy" and current_price <= new_tp_price_max):
                    # Check if current price has already surpassed the max TP price
                    if (order_side == "sell" and current_price > new_tp_price_max) or (order_side == "buy" and current_price < new_tp_price_min):
                        try:
                            tp_price = best_ask_price if order_side == "sell" else best_bid_price
                            await self.exchange.create_normal_take_profit_order_async(symbol, "limit", order_side, pos_qty, tp_price, positionIdx=positionIdx, reduce_only=True)
                            logging.info(f"New {order_side.capitalize()} TP set at current/best price {tp_price} as current price has surpassed the max TP")
                        except Exception as e:
                            logging.info(f"Failed to set new {order_side} TP for {symbol} at current/best price. Error: {e}")
                    else:
                        try:
                            await self.exchange.create_normal_take_profit_order_async(symbol, "limit", order_side, pos_qty, new_tp_price_min, positionIdx=positionIdx, reduce_only=True)
                            logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price_min} using a normal limit order")
                        except Exception as e:
                            logging.info(f"Failed to set new {order_side} TP for {symbol} using a normal limit order. Error: {e}")
                else:
                    try:
                        await self.exchange.create_take_profit_order_async(symbol, "limit", order_side, pos_qty, new_tp_price_max, positionIdx=positionIdx, reduce_only=True)
                        logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price_max} using a post-only order")
                    except Exception as e:
                        logging.info(f"Failed to set new {order_side} TP for {symbol} using a post-only order. Error: {e}")
            else:
                logging.info(f"Skipping TP update as a TP order already exists for {symbol}")

            # Calculate and return the next update time
            return self.calculate_next_update_time()
        else:
            logging.info(f"No immediate update needed for TP orders for {symbol}. Last update at: {last_tp_update}")
            return last_tp_update

    async def get_4h_candle_spread_async(self, symbol: str) -> float:
        ohlcv_data = await self.exchange.fetch_ohlcv_async(symbol=symbol, timeframe='4h', limit=1)
        logging.info(f'[{symbol}] ohlcv_dataohlcv_dataohlcv_data: {ohlcv_data}')

        if ohlcv_data is None or ohlcv_data.empty:
            logging.warning(f"No OHLCV data available for {symbol}")
            return 0.0  # Return a default value or handle as needed

        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert 'high' and 'low' columns to numeric
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])

        high_low_spread = df['high'].iloc[0] - df['low'].iloc[0]
        return high_low_spread

    async def get_spread_and_price_async(self, symbol):
        spread = await self.get_4h_candle_spread_async(symbol)
        logging.info(f"4h Candle spread for {symbol}: {spread}")

        current_price = await self.exchange.get_current_price_async(symbol)
        logging.info(f"[{symbol}] Current price: {current_price}")

        return spread, current_price

    async def clear_grid_async(self, symbol, side, max_retries=50, delay=2):
        """Clear all orders and internal states for a specific grid side with retries, excluding reduce-only orders."""
        logging.info(f"Clearing {side} grid for {symbol}")

        retry_counter = 0
        while retry_counter < max_retries:
            # Cancel all orders for the specified side
            await self.cancel_grid_orders_async(symbol, side)

            # Re-fetch the open orders and filter out reduce-only orders
            open_orders_after_clear = await self.retry_api_call_async(self.exchange.get_open_orders_async, symbol)
            lingering_orders = [o for o in open_orders_after_clear if o['info']['side'].lower() == side and not o['info'].get('reduceOnly', False)]

            if not lingering_orders:
                # Clear filled levels for the side when no more lingering orders
                self.filled_levels[symbol][side].clear()
                logging.info(f"Successfully cleared {side} grid for {symbol}.")

                # Mark the grid as cleared
                if side == 'buy':
                    self.grid_cleared_status[symbol]['long'] = True
                elif side == 'sell':
                    self.grid_cleared_status[symbol]['short'] = True

                break
            else:
                logging.warning(f"Lingering {side} orders detected for {symbol} (non-reduce-only): {lingering_orders}. Retrying...")
                retry_counter += 1
                await asyncio.sleep(delay)  # Wait before retrying

        if retry_counter == max_retries:
            logging.error(f"Failed to clear all {side} grid orders for {symbol} after {max_retries} attempts.")
        else:
            logging.info(f"{side.capitalize()} grid cleared after {retry_counter} retries for {symbol}.")

    async def check_and_manage_positions_async(self, long_pos_qty, short_pos_qty, symbol, total_equity, current_price, wallet_exposure_limit_long, wallet_exposure_limit_short, max_qty_percent_long, max_qty_percent_short):
        try:
            logging.info(f"Checking and managing positions for {symbol}")

            # Calculate the maximum allowed positions based on the total equity and wallet exposure limits
            max_qty_long = (total_equity * wallet_exposure_limit_long) / current_price
            max_qty_short = (total_equity * wallet_exposure_limit_short) / current_price

            # Calculate the position exposure percentages
            long_pos_exposure_percent = (long_pos_qty * current_price / total_equity) * 100
            short_pos_exposure_percent = (short_pos_qty * current_price / total_equity) * 100

            # Log the position utilization and the actual utilization based on total equity
            logging.info(f"Position utilization for {symbol}:")
            logging.info(f"  - Long position exposure: {long_pos_exposure_percent:.2f}% of total equity")
            logging.info(f"  - Short position exposure: {short_pos_exposure_percent:.2f}% of total equity")

            # Log detailed information about the configuration parameters and maximum allowed positions
            logging.info(f"Configuration for {symbol}:")
            logging.info(f"  - Total equity: {total_equity:.2f} USD")
            logging.info(f"  - Current price: {current_price:.8f} USD")
            logging.info(f"  - Wallet exposure limit for long: {wallet_exposure_limit_long * 100:.2f}%")
            logging.info(f"  - Wallet exposure limit for short: {wallet_exposure_limit_short * 100:.2f}%")
            logging.info(f"  - Max quantity percentage for long: {max_qty_percent_long}%")
            logging.info(f"  - Max quantity percentage for short: {max_qty_percent_short}%")
            logging.info(f"Maximum allowed positions for {symbol}:")
            logging.info(f"  - Max quantity for long: {max_qty_long:.4f} units")
            logging.info(f"  - Max quantity for short: {max_qty_short:.4f} units")
            logging.info(f"{symbol} Long position quantity: {long_pos_qty}")
            logging.info(f"{symbol} Short position quantity: {short_pos_qty}")

            # Check if current positions exceed the maximum allowed quantities
            if long_pos_exposure_percent > max_qty_percent_long:
                logging.info(f"[{symbol}] Long position exposure exceeds the maximum allowed. Current long position exposure: {long_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_long}%. Clearing long grid.")
                await self.clear_grid_async(symbol, 'buy')
                self.active_grids.discard(symbol)
                self.max_qty_reached_symbol_long.add(symbol)
            elif symbol in self.max_qty_reached_symbol_long and long_pos_exposure_percent <= max_qty_percent_long:
                logging.info(f"[{symbol}] Long position exposure is below the maximum allowed. Removing from max_qty_reached_symbol_long. Current long position exposure: {long_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_long}%.")
                self.max_qty_reached_symbol_long.remove(symbol)

            if short_pos_exposure_percent > max_qty_percent_short:
                logging.info(f"[{symbol}] Short position exposure exceeds the maximum allowed. Current short position exposure: {short_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_short}%. Clearing short grid.")
                await self.clear_grid_async(symbol, 'sell')
                self.active_grids.discard(symbol)
                self.max_qty_reached_symbol_short.add(symbol)
            elif symbol in self.max_qty_reached_symbol_short and short_pos_exposure_percent <= max_qty_percent_short:
                logging.info(f"[{symbol}] Short position exposure is below the maximum allowed. Removing from max_qty_reached_symbol_short. Current short position exposure: {short_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_short}%.")
                self.max_qty_reached_symbol_short.remove(symbol)

        except Exception as e:
            logging.error(f"Exception caught in check and manage positions: {e}")
            logging.info("Traceback:", traceback.format_exc())

    async def get_order_book_prices_async(self, symbol, current_price):
        order_book = await self.exchange.get_orderbook_async(symbol)
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
        return order_book, best_ask_price, best_bid_price

    async def get_precision_and_min_qty_async(self, symbol):
        market_data = await self.get_market_data_with_retry_async(symbol, max_retries=100, retry_delay=5)
        min_qty = float(market_data["min_qty"])
        qty_precision = float(market_data["precision"])
        logging.info(f"Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")
        return qty_precision, min_qty

    async def issue_grid_orders_async(self, symbol: str, side: str, grid_levels: list, amounts: list, is_long: bool, filled_levels: set):
        """
        Check the status of existing grid orders and place new orders for unfilled levels.
        """
        try:
            # Fetch open orders from the exchange
            open_orders = await self.retry_api_call_async(self.exchange.get_open_orders_async, symbol)

            # Get the current price to update last reissue prices
            current_price = await self.exchange.get_current_price_async(symbol)

            # Clear existing grid before placing new orders
            if is_long:
                logging.info(f"Clearing existing long grid for {symbol} before issuing new orders. (line: {inspect.currentframe().f_lineno})")
                await self.clear_grid_async(symbol, 'buy')
                self.last_reissue_price_long[symbol] = current_price
                logging.info(f"Updated last reissue price for long orders of {symbol} to {current_price}")
            else:
                logging.info(f"Clearing existing short grid for {symbol} before issuing new orders. (line: {inspect.currentframe().f_lineno})")
                await self.clear_grid_async(symbol, 'sell')
                self.last_reissue_price_short[symbol] = current_price
                logging.info(f"Updated last reissue price for short orders of {symbol} to {current_price}")

            # Clear the filled_levels set before placing new orders
            filled_levels.clear()

            # Add logging to verify types before the zip operation
            logging.info(f"Inside issue_grid_orders - Type of grid_levels: {type(grid_levels)}, Value: {grid_levels}")
            logging.info(f"Inside issue_grid_orders - Type of amounts: {type(amounts)}, Value: {amounts}")

            # Ensure grid_levels and amounts are lists
            assert isinstance(grid_levels, list), f"Expected grid_levels to be a list, but got {type(grid_levels)}"
            assert isinstance(amounts, list), f"Expected amounts to be a list, but got {type(amounts)}"

            # Ensure all elements within grid_levels and amounts are of correct type
            for level in grid_levels:
                assert isinstance(level, (float, int)), f"Each level in grid_levels should be a float or int, but got {type(level)}"

            for amount in amounts:
                assert isinstance(amount, (float, int)), f"Each amount in amounts should be a float or int, but got {type(amount)}"

            # Place new grid orders for unfilled levels
            for level, amount in zip(grid_levels, amounts):
                order_exists = any(order['price'] == level and order['side'].lower() == side.lower() for order in open_orders)
                if not order_exists:
                    order_link_id = self.generate_order_link_id(symbol, side, level)
                    position_idx = 1 if is_long else 2
                    try:
                        order = await self.exchange.create_tagged_limit_order_async(symbol, side, amount, level, positionIdx=position_idx, orderLinkId=order_link_id)
                        if order and 'id' in order:
                            logging.info(f"Placed {side} order at level {level} for {symbol} with amount {amount}")
                            filled_levels.add(level)  # Add the level to filled_levels
                        else:
                            logging.info(f"Failed to place {side} order at level {level} for {symbol} with amount {amount}")
                    except Exception as e:
                        logging.error(f"Exception when placing {side} order at level {level} for {symbol}: {e}")
                else:
                    logging.info(f"Skipping {side} order at level {level} for {symbol} as it already exists.")

            logging.info(f"[{symbol}] {side.capitalize()} grid orders issued for unfilled levels.")
        except Exception as e:
            logging.error(f"Exception in issue_grid_orders: {e}")

    async def get_best_bid_price_async(self, symbol):
        """Fetch the best bid price for a given symbol, with a fallback to last known bid price."""
        try:
            order_book = await self.exchange.get_orderbook_async(symbol)

            # Check for bids in the order book and fetch the best bid price
            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
                self.last_known_bid[symbol] = best_bid_price  # Update last known bid price
            else:
                best_bid_price = self.last_known_bid.get(symbol)  # Use last known bid price
                if best_bid_price is None:
                    logging.warning(f"Best bid price is not available for {symbol}. Defaulting to 0.0.")
                    best_bid_price = 0.0  # Default to 0.0 if no known bid price

            # Ensure the bid price is a float
            return float(best_bid_price)

        except Exception as e:
            logging.error(f"Error fetching best bid price for {symbol}: {e}")
            return 0.0  # Return 0.0 in case of failure

    async def get_best_ask_price_async(self, symbol):
        """Fetch the best ask price for a given symbol, with a fallback to last known ask price."""
        try:
            order_book = await self.exchange.get_orderbook_async(symbol)

            # Check for asks in the order book and fetch the best ask price
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
                self.last_known_ask[symbol] = best_ask_price  # Update last known ask price
            else:
                best_ask_price = self.last_known_ask.get(symbol)  # Use last known ask price
                if best_ask_price is None:
                    logging.warning(f"Best ask price is not available for {symbol}. Defaulting to 0.0.")
                    best_ask_price = 0.0  # Default to 0.0 if no known ask price

            # Ensure the ask price is a float
            return float(best_ask_price)

        except Exception as e:
            logging.error(f"Error fetching best ask price for {symbol}: {e}")
            return 0.0  # Return 0.0 in case of failure

    async def get_position_qty_async(self, symbol, side):
        # Fetch open position data
        open_position_data = await self.retry_api_call_async(self.exchange.get_all_open_positions_async)
        position_details = {}

        # Process the fetched position data
        for position in open_position_data:
            info = position.get('info', {})
            position_symbol = info.get('symbol', '').split(':')[0]

            if 'size' in info and 'side' in info and 'avgPrice' in info and 'liqPrice' in info:
                size = float(info['size'])
                side_type = info['side'].lower()
                avg_price = float(info['avgPrice'])
                liq_price = info.get('liqPrice', None)

                if position_symbol not in position_details:
                    position_details[position_symbol] = {
                        'long': {'qty': 0, 'avg_price': 0, 'liq_price': None},
                        'short': {'qty': 0, 'avg_price': 0, 'liq_price': None}
                    }

                if side_type == 'buy':
                    position_details[position_symbol]['long']['qty'] += size
                    position_details[position_symbol]['long']['avg_price'] = avg_price
                    position_details[position_symbol]['long']['liq_price'] = liq_price
                elif side_type == 'sell':
                    position_details[position_symbol]['short']['qty'] += size
                    position_details[position_symbol]['short']['avg_price'] = avg_price
                    position_details[position_symbol]['short']['liq_price'] = liq_price

        if side == 'long':
            return position_details.get(symbol, {}).get('long', {}).get('qty', 0)
        elif side == 'short':
            return position_details.get(symbol, {}).get('short', {}).get('qty', 0)
        return 0

    async def generate_l_signals_async(self, symbol):
        async with self.general_rate_limiter_async:
            return await self.exchange.generate_l_signals_async(symbol)

    async def handle_grid_trades_async(
        self,
        symbol,
        grid_levels_long, grid_levels_short,
        long_grid_active, short_grid_active,
        long_pos_qty, short_pos_qty, current_price,
        dynamic_outer_price_distance_long, dynamic_outer_price_distance_short,
        min_outer_price_distance_long, min_outer_price_distance_short,
        buffer_percentage_long, buffer_percentage_short,
        adjusted_grid_levels_long, adjusted_grid_levels_short,
        levels, amounts_long, amounts_short,
        best_bid_price, best_ask_price,
        mfirsi_signal, open_orders, initial_entry_buffer_pct,
        reissue_threshold, entry_during_autoreduce, min_qty, open_symbols,
        symbols_allowed, long_mode, short_mode,
        long_pos_price, short_pos_price,
        graceful_stop_long, graceful_stop_short,
        min_buffer_percentage, max_buffer_percentage,
        additional_entries_from_signal, open_position_data,
        upnl_profit_pct, max_upnl_profit_pct, tp_order_counts,
        stop_loss_long, stop_loss_short, stop_loss_enabled=True,
        # --- Auto-hedge params ---
        auto_hedge_enabled=False,
        auto_hedge_ratio=0.3,
        auto_hedge_min_position_size=0.001,
        auto_hedge_price_diff_threshold=0.002,
        # --- NEW toggles ---
        disable_grid_on_hedge_side=False,
        hedge_with_grid=False,
        forcibly_close_hedge=True,
    ):
        """
        Main routine that:
        1) Checks stop-loss logic,
        2) Executes auto-hedge logic (new!),
        3) Manages grid orders, re-issues, additional entries, etc.

        Args:
            symbol (str): Symbol/pair, e.g. "BTCUSDT"
            grid_levels_long (list): Price levels for the long grid
            grid_levels_short (list): Price levels for the short grid
            long_grid_active (bool): Flag if long grid is active
            short_grid_active (bool): Flag if short grid is active
            long_pos_qty (float): Current long position size
            short_pos_qty (float): Current short position size
            current_price (float): Current market price
            dynamic_outer_price_distance_long (float): Outer distance for dynamic logic (long)
            dynamic_outer_price_distance_short (float): Outer distance for dynamic logic (short)
            ...
            stop_loss_long (float): Stop-loss percentage for longs
            stop_loss_short (float): Stop-loss percentage for shorts
            stop_loss_enabled (bool): Master toggle for stop-loss

            auto_hedge_enabled (bool): Master toggle for auto-hedge
            auto_hedge_ratio (float): Hedge size ratio relative to current position size
            auto_hedge_min_position_size (float): Don’t hedge if position is smaller than this
            auto_hedge_price_diff_threshold (float): Price-diff threshold (e.g. 0.002 = 0.2%)

            disable_grid_on_hedge_side (bool): If True, do not place grid orders on whichever side is hedged
            hedge_with_grid (bool): If True, do NOT automatically close the hedge if main position < min size or 0
        """

        logging.info(
            f"[AUTO-HEDGE] Received params - ratio: {auto_hedge_ratio}, "
            f"min_size: {auto_hedge_min_position_size}, threshold: {auto_hedge_price_diff_threshold}"
        )

        try:
            # -------------------------------
            # 1) STOP-LOSS LOGIC (Unchanged)
            # -------------------------------
            has_open_long_position = (long_pos_qty > 0)
            has_open_short_position = (short_pos_qty > 0)

            # Store previous state of long and short positions for comparison
            if not hasattr(self, 'previous_position_state'):
                self.previous_position_state = {}

            if symbol not in self.previous_position_state:
                self.previous_position_state[symbol] = {
                    'long': False,
                    'short': False,
                    'long_initial_entry': None,
                    'short_initial_entry': None
                }

            long_position_closed = (
                self.previous_position_state[symbol]['long']
                and not has_open_long_position
            )
            short_position_closed = (
                self.previous_position_state[symbol]['short']
                and not has_open_short_position
            )

            if long_position_closed:
                logging.info(f"[{symbol}] Long position closed. Resetting initial entry price for long.")
                self.previous_position_state[symbol]['long_initial_entry'] = None

            if short_position_closed:
                logging.info(f"[{symbol}] Short position closed. Resetting initial entry price for short.")
                self.previous_position_state[symbol]['short_initial_entry'] = None

            self.previous_position_state[symbol]['long'] = has_open_long_position
            self.previous_position_state[symbol]['short'] = has_open_short_position

            if has_open_long_position and not self.previous_position_state[symbol]['long_initial_entry']:
                self.previous_position_state[symbol]['long_initial_entry'] = long_pos_price
                logging.info(f"[{symbol}] Long position opened. Initial entry price for stop-loss: {long_pos_price}")

            if has_open_short_position and not self.previous_position_state[symbol]['short_initial_entry']:
                self.previous_position_state[symbol]['short_initial_entry'] = short_pos_price
                logging.info(f"[{symbol}] Short position opened. Initial entry price for stop-loss: {short_pos_price}")

            if stop_loss_enabled:
                stop_loss_price_long = (
                    self.previous_position_state[symbol]['long_initial_entry'] * (1 - stop_loss_long / 100)
                    if has_open_long_position else None
                )
                stop_loss_price_short = (
                    self.previous_position_state[symbol]['short_initial_entry'] * (1 + stop_loss_short / 100)
                    if has_open_short_position else None
                )

                logging.info(
                    f"[{symbol}] Calculated stop-loss prices: "
                    f"Long - {stop_loss_price_long}, Short - {stop_loss_price_short}"
                )

                # Long position SL check
                if has_open_long_position:
                    logging.info(
                        f"[{symbol}] Long Position Qty: {long_pos_qty}, "
                        f"Entry Price: {long_pos_price}, SL: {stop_loss_price_long}"
                    )
                    if current_price <= stop_loss_price_long:
                        logging.info(
                            f"[{symbol}] Long position hit stop-loss at {stop_loss_price_long}."
                        )
                        self.trigger_stop_loss(symbol, long_pos_qty, 'long', stop_loss_price_long, best_bid_price)

                # Short position SL check
                if has_open_short_position:
                    logging.info(
                        f"[{symbol}] Short Position Qty: {short_pos_qty}, "
                        f"Entry Price: {short_pos_price}, SL: {stop_loss_price_short}"
                    )
                    if current_price >= stop_loss_price_short:
                        logging.info(
                            f"[{symbol}] Short position hit stop-loss at {stop_loss_price_short}."
                        )
                        self.trigger_stop_loss(symbol, short_pos_qty, 'short', stop_loss_price_short, best_ask_price)
            else:
                logging.info("Stop-loss disabled")

            # -------------------------------------
            # 2) AUTO-HEDGE LOGIC (Newly Inserted)
            # -------------------------------------
            if not hasattr(self, 'hedge_positions'):
                self.hedge_positions = {}

            if symbol not in self.hedge_positions:
                self.hedge_positions[symbol] = {
                    'side': None,  # 'long' or 'short'
                    'qty': 0.0
                }

            current_hedge_side = self.hedge_positions[symbol]['side']

            if auto_hedge_enabled:
                try:
                    # If net LONG
                    if has_open_long_position and (long_pos_qty >= auto_hedge_min_position_size):
                        entry_price = long_pos_price
                        if (entry_price and abs(current_price / entry_price - 1) >= auto_hedge_price_diff_threshold):
                            desired_hedge_qty = long_pos_qty * auto_hedge_ratio
                            # Pass forcibly_close_hedge
                            await self.open_or_adjust_hedge_async(symbol, 'short', desired_hedge_qty, forcibly_close_hedge)
                        else:
                            logging.info(f"[AUTO-HEDGE] {symbol} price diff below threshold => no new hedge.")
                    # If net SHORT
                    elif has_open_short_position and (short_pos_qty >= auto_hedge_min_position_size):
                        entry_price = short_pos_price
                        if (entry_price and abs(current_price / entry_price - 1) >= auto_hedge_price_diff_threshold):
                            desired_hedge_qty = short_pos_qty * auto_hedge_ratio
                            # Pass forcibly_close_hedge
                            await self.open_or_adjust_hedge_async(symbol, 'long', desired_hedge_qty, forcibly_close_hedge)
                        else:
                            logging.info(f"[AUTO-HEDGE] {symbol} price diff below threshold => no new hedge.")
                    else:
                        # No net position or below min => close hedge unless hedge_with_grid is True
                        if self.hedge_positions[symbol]['side']:
                            if hedge_with_grid:
                                logging.info(f"[AUTO-HEDGE] {symbol}: 'hedge_with_grid=True' => keep existing hedge.")
                            else:
                                logging.info(f"[AUTO-HEDGE] {symbol}: No main pos => closing hedge.")
                                await self.close_hedge_position_async(symbol)
                        else:
                            logging.info(f"[AUTO-HEDGE] {symbol}: No main pos => no hedge to close.")
                except Exception as hedge_err:
                    logging.warning(f"[AUTO-HEDGE] Error for {symbol}: {hedge_err}")
            else:
                logging.info(f"[AUTO-HEDGE] Disabled for {symbol}.")
                if self.hedge_positions[symbol]['side']:
                    if hedge_with_grid:
                        logging.info(f"[AUTO-HEDGE] {symbol}: Hedge disabled but hedge_with_grid=True => keep hedge.")
                    else:
                        logging.info(f"[AUTO-HEDGE] {symbol}: Hedge disabled => closing existing hedge.")
                        await self.close_hedge_position_async(symbol)

            # -----------------------------------------------------------
            # 3) REMAINDER OF YOUR ORIGINAL GRID LOGIC (Unchanged Below)
            # -----------------------------------------------------------
            #open_symbols_long = self.get_open_symbols_long(open_position_data)
            #open_symbols_short = self.get_open_symbols_short(open_position_data)

            # logging.info(f"Open symbols long: {open_symbols_long}")
            # logging.info(f"Open symbols short: {open_symbols_short}")

            #length_of_open_symbols_long = len(open_symbols_long)
            #length_of_open_symbols_short = len(open_symbols_short)

            # logging.info(f"Length of open symbols long: {length_of_open_symbols_long}")
            # logging.info(f"Length of open symbols short: {length_of_open_symbols_short}")

            # Count unique open symbols across both long and short positions
            unique_open_symbols = len(set(open_symbols))

            logging.info(f"Unique open symbols: {unique_open_symbols}")

            # should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
            #     symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct
            # )

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': 0, 'short': 0}

            logging.info(f"Symbol format: {symbol}")
            logging.info(f"Test open orders: {open_orders}")

            has_open_long_order = any(
                order['info']['symbol'] == symbol
                and order['info']['side'].lower() == 'buy'
                and not order['info']['reduceOnly']
                for order in open_orders
            )
            has_open_short_order = any(
                order['info']['symbol'] == symbol
                and order['info']['side'].lower() == 'sell'
                and not order['info']['reduceOnly']
                for order in open_orders
            )

            logging.info(f"{symbol} Has open long order: {has_open_long_order}")
            logging.info(f"{symbol} Has open short order: {has_open_short_order}")

            # replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            # replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            current_time = time.time()

            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")
            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            # Additional logic for managing open symbols and checking trading permissions
            # open_symbols = list(set(all_open_symbols))
            logging.info(f"Open symbols: {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(
                f"Checking trading for symbol {symbol}. "
                f"Can trade: {trading_allowed}, In open_symbols: {symbol in open_symbols}, Allowed: {symbols_allowed}"
            )

            fresh_mfirsi_signal = await self.generate_l_signals_async(symbol)
            logging.info(f"Fresh MFIRSI signal for {symbol}: {fresh_mfirsi_signal}")

            # Derive boolean flags for clarity
            mfi_signal_long = (fresh_mfirsi_signal == "long")
            mfi_signal_short = (fresh_mfirsi_signal == "short")
            logging.info(f"MFIRSI SIGNAL FOR {symbol}: {mfirsi_signal}")

            async def issue_grid_safely(symbol: str, side: str, grid_levels: list, amounts: list):
                async with grid_lock_async:  # Lock to ensure no simultaneous grid issuance
                    try:
                        grid_set = (
                            self.active_long_grids if side == 'long'
                            else self.active_short_grids
                        )
                        order_side = 'buy' if side == 'long' else 'sell'

                        if self.has_active_grid(symbol, side, open_orders):
                            logging.warning(
                                f"[{symbol}] {side.capitalize()} grid already active or existing order found."
                            )
                            return

                        if isinstance(grid_levels, int):
                            grid_levels = [grid_levels]
                        assert isinstance(grid_levels, list), f"Expected grid_levels to be a list, got {type(grid_levels)}"

                        if isinstance(amounts, int):
                            amounts = [amounts] * len(grid_levels)
                        assert isinstance(amounts, list), f"Expected amounts to be a list, got {type(amounts)}"

                        if self.grid_cleared_status.get(symbol, {}).get(side, False):
                            logging.info(f"[{symbol}] Issuing new {side} grid orders.")
                            self.grid_cleared_status[symbol][side] = False
                            await self.issue_grid_orders_async(
                                symbol,
                                order_side,
                                grid_levels,
                                amounts,
                                (side == 'long'),
                                self.filled_levels[symbol][order_side]
                            )
                            grid_set.add(symbol)
                            logging.info(
                                f"[{symbol}] Successfully issued {side} grid orders."
                            )
                        else:
                            logging.warning(
                                f"[{symbol}] Attempted to issue {side} grid orders, but no clearance set."
                            )
                    except Exception as e:
                        logging.error(f"Exception in issue_grid_safely for {symbol} - {side}: {e}")

            current_hedge_side = self.hedge_positions[symbol]['side']
            skip_long_side = False
            skip_short_side = False
            if disable_grid_on_hedge_side:
                if current_hedge_side == 'long':
                    skip_long_side = True
                elif current_hedge_side == 'short':
                    skip_short_side = True

            # TODO: uncomment
            # replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
            #     symbol,
            #     long_pos_price,
            #     short_pos_price,
            #     long_pos_qty,
            #     short_pos_qty,
            #     dynamic_outer_price_distance_long=dynamic_outer_price_distance_long,
            #     dynamic_outer_price_distance_short=dynamic_outer_price_distance_short
            # )

            has_open_long_position = (long_pos_qty > 0)
            has_open_short_position = (short_pos_qty > 0)

            logging.info(
                f"{symbol} has long? {has_open_long_position}, short? {has_open_short_position}"
            )
            logging.info(
                f"[{symbol}] # of open symbols: {len(open_symbols)}, allowed: {symbols_allowed}"
            )

            # Place or re-issue grids if conditions allow
            if (unique_open_symbols <= symbols_allowed) or (symbol in open_symbols):
                fresh_signal = await self.generate_l_signals_async(symbol)
                try:
                    # Handling for Long Positions (skip if skip_long_side=True)
                    if not skip_long_side:
                        if (
                            fresh_signal.lower() == "long"
                            and long_mode
                            and not has_open_long_position
                            and not graceful_stop_long
                            and not self.has_active_grid(symbol, 'long', open_orders)
                            and symbol not in self.max_qty_reached_symbol_long
                        ):
                            logging.info(
                                f"[{symbol}] Creating new long position (signal=long)."
                            )

                            await self.clear_grid_async(symbol, 'buy')
                            modified_grid_levels_long = grid_levels_long.copy()
                            best_bid_price = await self.get_best_bid_price(symbol)
                            logging.info(
                                f"[{symbol}] Setting first level of modified grid to best_bid_price: {best_bid_price}"
                            )
                            modified_grid_levels_long[0] = best_bid_price

                            await issue_grid_safely(
                                symbol, 'long', modified_grid_levels_long, amounts_long
                            )

                            retry_counter = 0
                            max_retries = 100

                            while long_pos_qty < 0.00001 and retry_counter < max_retries:
                                await asyncio.sleep(3)

                                try:
                                    long_pos_qty = await self.get_position_qty_async(symbol, 'long')
                                except Exception as e:
                                    logging.error(
                                        f"[{symbol}] Error fetching long pos qty: {e}"
                                    )
                                    break

                                retry_counter += 1
                                logging.info(
                                    f"[{symbol}] Long pos qty after waiting: {long_pos_qty}, "
                                    f"retry {retry_counter}"
                                )

                                if (retry_counter % 10 == 0) and (long_pos_qty < 0.00001):
                                    logging.info(
                                        f"[{symbol}] Retrying long grid orders after {retry_counter} tries."
                                    )
                                    best_bid_price = await self.get_best_bid_price_async(symbol)
                                    await self.clear_grid_async(symbol, 'buy')
                                    modified_grid_levels_long[0] = best_bid_price
                                    await issue_grid_safely(
                                        symbol, 'long', modified_grid_levels_long, amounts_long
                                    )

                            logging.info(
                                f"[{symbol}] Long position filled or max retries reached."
                            )
                            self.last_signal_time[symbol] = current_time
                            self.last_mfirsi_signal[symbol] = "neutral"
                    else:
                        logging.info(
                            f"[{symbol}] skip_long_side=True => skipping new LONG grid logic (hedge side=long)"
                        )

                    # Handling for Short Positions (skip if skip_short_side=True)
                    if not skip_short_side:
                        if (
                            fresh_signal.lower() == "short"
                            and short_mode
                            and not has_open_short_position
                            and not graceful_stop_short
                            and not self.has_active_grid(symbol, 'short', open_orders)
                            and symbol not in self.max_qty_reached_symbol_short
                        ):
                            logging.info(
                                f"[{symbol}] Creating new short position (signal=short)."
                            )

                            await self.clear_grid_async(symbol, 'sell')
                            modified_grid_levels_short = grid_levels_short.copy()
                            best_ask_price = await self.get_best_ask_price_async(symbol)
                            logging.info(
                                f"[{symbol}] Setting first level of modified grid to best_ask_price: {best_ask_price}"
                            )
                            modified_grid_levels_short[0] = best_ask_price

                            await issue_grid_safely(
                                symbol, 'short', modified_grid_levels_short, amounts_short
                            )

                            retry_counter = 0
                            max_retries = 50

                            while short_pos_qty < 0.00001 and retry_counter < max_retries:
                                await asyncio.sleep(5)
                                try:
                                    short_pos_qty = await self.get_position_qty_async(symbol, 'short')
                                except Exception as e:
                                    logging.error(
                                        f"[{symbol}] Error fetching short pos qty: {e}"
                                    )
                                    break

                                retry_counter += 1
                                logging.info(
                                    f"[{symbol}] Short pos qty after waiting: {short_pos_qty}, "
                                    f"retry {retry_counter}"
                                )

                                if (retry_counter % 10 == 0) and (short_pos_qty < 0.00001):
                                    logging.info(
                                        f"[{symbol}] Retrying short grid orders after {retry_counter} tries."
                                    )
                                    best_ask_price = await self.get_best_ask_price_async(symbol)
                                    await self.clear_grid_async(symbol, 'sell')
                                    modified_grid_levels_short[0] = best_ask_price
                                    await issue_grid_safely(
                                        symbol,
                                        'short',
                                        modified_grid_levels_short,
                                        amounts_short
                                    )

                            logging.info(
                                f"[{symbol}] Short position filled or max retries reached."
                            )
                            self.last_signal_time[symbol] = current_time
                            self.last_mfirsi_signal[symbol] = "neutral"
                    else:
                        logging.info(
                            f"[{symbol}] skip_short_side=True => skipping new SHORT grid logic (hedge side=short)"
                        )

                except Exception as e:
                    logging.info(f"Exception caught in placing orders: {e}")
                    logging.info("Traceback: %s", traceback.format_exc())

            # -------------------------------------------------------
            # Additional entries from signal (MFI/RSI) if enabled
            # -------------------------------------------------------
            if additional_entries_from_signal:
                if symbol in open_symbols:
                    logging.info(f"Allowed symbol: {symbol}")

                    fresh_signal = await self.generate_l_signals_async(symbol)

                    logging.info(f"Fresh signal for {symbol}: {fresh_signal}")

                    if not hasattr(self, 'last_mfirsi_signal'):
                        self.last_mfirsi_signal = {}
                    if not hasattr(self, 'last_signal_time'):
                        self.last_signal_time = {}

                    if self.last_mfirsi_signal is None:
                        self.last_mfirsi_signal = {}
                    if self.last_signal_time is None:
                        self.last_signal_time = {}

                    if symbol not in self.last_mfirsi_signal:
                        self.last_mfirsi_signal[symbol] = "neutral"
                    if symbol not in self.last_signal_time:
                        self.last_signal_time[symbol] = 0

                    current_time = time.time()
                    last_signal_time = self.last_signal_time.get(symbol, 0)
                    time_since_last_signal = current_time - last_signal_time

                    if time_since_last_signal < 180:
                        logging.info(
                            f"[{symbol}] Waiting for signal cooldown. "
                            f"Time since last signal: {time_since_last_signal:.2f}s"
                        )
                        return

                    if fresh_signal.lower() != self.last_mfirsi_signal[symbol]:
                        logging.info(f"[{symbol}] MFIRSI signal changed to {fresh_signal}")
                        self.last_mfirsi_signal[symbol] = fresh_signal.lower()
                    else:
                        logging.info(f"[{symbol}] MFIRSI signal unchanged: {fresh_signal}")

                    try:
                        # Additional Long entries
                        if (
                            fresh_signal.lower() == "long"
                            and long_mode
                            and not self.auto_reduce_active_long.get(symbol, False)
                        ):
                            if (long_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_long):
                                if current_price <= long_pos_price:
                                    logging.info(
                                        f"[{symbol}] Adding to existing long (MFIRSI=long)."
                                    )
                                    if not self.has_active_grid(symbol, 'long', open_orders):
                                        await self.clear_grid_async(symbol, 'buy')
                                        modified_grid_levels_long = grid_levels_long.copy()
                                        modified_grid_levels_long[0] = best_bid_price
                                        if skip_long_side:
                                            logging.info(
                                                f"[{symbol}] skip_long_side=True => skipping add to LONG side."
                                            )
                                        else:
                                            await issue_grid_safely(
                                                symbol,
                                                'long',
                                                modified_grid_levels_long,
                                                amounts_long
                                            )
                                            # Retry logic, if needed
                                else:
                                    logging.info(
                                        f"[{symbol}] Price > long_pos_price => not adding to long."
                                    )

                        # Additional Short entries
                        elif (
                            fresh_signal.lower() == "short"
                            and short_mode
                            and not self.auto_reduce_active_short.get(symbol, False)
                        ):
                            if (short_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_short):
                                if current_price >= short_pos_price:
                                    logging.info(
                                        f"[{symbol}] Adding to existing short (MFIRSI=short)."
                                    )
                                    if not self.has_active_grid(symbol, 'short', open_orders):
                                        await self.clear_grid_async(symbol, 'sell')
                                        modified_grid_levels_short = grid_levels_short.copy()
                                        modified_grid_levels_short[0] = best_ask_price
                                        if skip_short_side:
                                            logging.info(
                                                f"[{symbol}] skip_short_side=True => skipping add to SHORT side."
                                            )
                                        else:
                                            await issue_grid_safely(
                                                symbol,
                                                'short',
                                                modified_grid_levels_short,
                                                amounts_short
                                            )
                                            # Retry logic, if needed
                                else:
                                    logging.info(
                                        f"[{symbol}] Price < short_pos_price => not adding to short."
                                    )
                            else:
                                logging.info(
                                    f"[{symbol}] Conditions not met for additional short entry."
                                )

                        elif fresh_signal.lower() == "neutral":
                            logging.info(f"[{symbol}] MFIRSI=neutral. No new grid orders.")

                        self.last_signal_time[symbol] = current_time
                        self.last_mfirsi_signal[symbol] = "neutral"

                    except Exception as e:
                        logging.info(f"Exception in placing entries {e}")
                        logging.info("Traceback: %s", traceback.format_exc())
                else:
                    logging.info("Additional entries from signal are disabled for this symbol.")
            else:
                logging.info("Additional entries from signal are disabled entirely.")

            await asyncio.sleep(5)

            # Check grid active
            logging.info(f"Symbol type for grid active check: {symbol}")
            long_grid_active, short_grid_active = self.check_grid_active(symbol, open_orders)
            logging.info(
                f"{symbol} Updated long grid active: {long_grid_active}, short grid active: {short_grid_active}"
            )

            # If no open orders, remove from active grids
            if not has_open_long_order and (symbol in self.active_long_grids):
                self.active_long_grids.discard(symbol)
                logging.info(f"[{symbol}] No open long orders => removed from active_long_grids")

            if not has_open_short_order and (symbol in self.active_short_grids):
                self.active_short_grids.discard(symbol)
                logging.info(f"[{symbol}] No open short orders => removed from active_short_grids")

            # If symbol is in open_symbols, attempt re-issuing grids for existing positions
            if symbol in open_symbols:
                # e.g. if (long_pos_qty>0 but no grid => place grid)
                if (
                    (long_pos_qty > 0 and not long_grid_active and not self.has_active_grid(symbol, 'long', open_orders))
                    or (short_pos_qty > 0 and not short_grid_active and not self.has_active_grid(symbol, 'short', open_orders))
                ):
                    logging.info(
                        f"[{symbol}] Open positions found without active grids. Issuing grid orders."
                    )

                    # Long Grid Logic
                    if (long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long):
                        if (not self.auto_reduce_active_long.get(symbol, False)) or entry_during_autoreduce:
                            if skip_long_side:
                                logging.info(
                                    f"[{symbol}] skip_long_side=True => skipping reissue of LONG grid."
                                )
                            else:
                                logging.info(
                                    f"[{symbol}] Placing long grid orders for existing open position."
                                )
                                await self.clear_grid_async(symbol, 'buy')

                                modified_grid_levels_long = grid_levels_long.copy()
                                best_bid_price = await self.get_best_bid_price(symbol)
                                modified_grid_levels_long[0] = best_bid_price * 0.995
                                logging.info(
                                    f"[{symbol}] Setting first level of modified long grid => best_bid_price*0.995 => {modified_grid_levels_long[0]}"
                                )
                                await issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

                    # Short Grid Logic
                    if (short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short):
                        if (not self.auto_reduce_active_short.get(symbol, False)) or entry_during_autoreduce:
                            if skip_short_side:
                                logging.info(
                                    f"[{symbol}] skip_short_side=True => skipping reissue of SHORT grid."
                                )
                            else:
                                logging.info(
                                    f"[{symbol}] Placing short grid orders for existing open position."
                                )
                                await self.clear_grid_async(symbol, 'sell')

                                modified_grid_levels_short = grid_levels_short.copy()
                                best_ask_price = await self.get_best_ask_price(symbol)
                                modified_grid_levels_short[0] = best_ask_price * 1.005
                                logging.info(
                                    f"[{symbol}] Setting first level of modified short grid => best_ask_price+0.5% => {modified_grid_levels_short[0]}"
                                )
                                await issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

                current_time = time.time()
                # Grid clearing logic if no positions
                if (
                    not long_pos_qty
                    and not short_pos_qty
                    and symbol in (self.active_long_grids | self.active_short_grids)
                ):
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if (current_time - last_cleared) > self.clear_interval:
                        logging.info(
                            f"[{symbol}] No open pos + time interval => cancel leftover grid orders."
                        )
                        await self.clear_grid_async(symbol, 'buy')
                        await self.clear_grid_async(symbol, 'sell')
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(
                            f"[{symbol}] No open pos, but time interval not passed => skip clearing."
                        )
            else:
                logging.info(
                    f"Symbol {symbol} not in open_symbols: {open_symbols} or not allowed => skip grid logic"
                )

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if (new_long_tp_min is not None) and (new_long_tp_max is not None):
                    self.next_long_tp_update = await self.update_quickscalp_tp_dynamic_async(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,
                        max_upnl_profit_pct=max_upnl_profit_pct,
                        short_pos_price=None,
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            # Update TP for short position
            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if (new_short_tp_min is not None) and (new_short_tp_max is not None):
                    self.next_short_tp_update = await self.update_quickscalp_tp_dynamic_async(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,
                        max_upnl_profit_pct=max_upnl_profit_pct,
                        short_pos_price=short_pos_price,
                        long_pos_price=None,
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            # Clear long grid if conditions are met
            if (
                has_open_long_order
                and ((long_pos_price is None) or (long_pos_price <= 0))
                and (not mfi_signal_long)
            ):
                if (symbol not in self.max_qty_reached_symbol_long):
                    current_time = time.time()
                    if symbol not in self.invalid_long_condition_time:
                        self.invalid_long_condition_time[symbol] = current_time
                        logging.info(
                            f"[{symbol}] Invalid long condition first met, waiting before clearing grid."
                        )
                    elif (current_time - self.invalid_long_condition_time[symbol]) >= 60:
                        await self.clear_grid_async(symbol, "buy")
                        logging.info(
                            f"[{symbol}] Cleared long grid after 1 min of invalid long_pos_price="
                            f"{long_pos_price} and no long signal."
                        )
                        del self.invalid_long_condition_time[symbol]
                else:
                    logging.info(
                        f"[{symbol}] In max_qty_reached_symbol_long => cannot replace grid"
                    )
            else:
                if symbol in self.invalid_long_condition_time:
                    del self.invalid_long_condition_time[symbol]

            # Clear short grid if conditions are met
            if (
                has_open_short_order
                and ((short_pos_price is None) or (short_pos_price <= 0))
                and (not mfi_signal_short)
            ):
                if (symbol not in self.max_qty_reached_symbol_short):
                    current_time = time.time()
                    if symbol not in self.invalid_short_condition_time:
                        self.invalid_short_condition_time[symbol] = current_time
                        logging.info(
                            f"[{symbol}] Invalid short condition first met, waiting before clearing grid."
                        )
                    elif (current_time - self.invalid_short_condition_time[symbol]) >= 60:
                        self.clear_grid(symbol, "sell")
                        logging.info(
                            f"[{symbol}] Cleared short grid after 1 min of invalid short_pos_price="
                            f"{short_pos_price} and no short signal."
                        )
                        del self.invalid_short_condition_time[symbol]
                else:
                    logging.info(
                        f"[{symbol}] In max_qty_reached_symbol_short => cannot replace grid"
                    )
            else:
                if symbol in self.invalid_short_condition_time:
                    del self.invalid_short_condition_time[symbol]

        except Exception as e:
            logging.info(f"[{symbol}] Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    # -----------------------------------------------------------------
    # HELPER METHODS FOR AUTO-HEDGING (Place them anywhere in your class)
    # -----------------------------------------------------------------

    async def open_or_adjust_hedge_async(self, symbol, hedge_side, desired_qty, forcibly_close_hedge=True):
        """
        Open or adjust a hedge position on the SAME symbol using post-only limit orders.
        (Requires Hedge Mode on Bybit to hold a separate hedge side.)

        1) If we already hold a hedge on the opposite side, close old hedge side if forcibly_close_hedge=True.
        If forcibly_close_hedge=False, skip closing the old hedge side and do nothing new.
        2) If we hold the same side, check how much to add or remove (difference).
        3) Place a post-only limit order on the correct side (buy=long or sell=short).
        """
        current_hedge_side = self.hedge_positions[symbol]['side']
        current_hedge_qty  = self.hedge_positions[symbol]['qty']

        if current_hedge_side and (current_hedge_side != hedge_side):
            if forcibly_close_hedge:
                logging.info(f"[AUTO-HEDGE] {symbol}: Hedge side mismatch => closing old side {current_hedge_side}")
                await self.close_hedge_position_async(symbol)
            else:
                logging.info(
                    f"[AUTO-HEDGE] {symbol}: Hedge side mismatch but forcibly_close_hedge=False => keeping old side {current_hedge_side}."
                )
                return  # do nothing, keep the old side

        qty_diff = desired_qty - current_hedge_qty
        if abs(qty_diff) < 1e-6:
            logging.info(f"[AUTO-HEDGE] {symbol}: Hedge already at {desired_qty:.4f}, no change needed.")
            return

        if hedge_side == 'short':
            position_idx = 2  # short side in Hedge Mode
            if qty_diff > 0:
                best_ask_price = await self.get_best_ask_price_async(symbol)
                final_qty = abs(qty_diff)
                logging.info(
                    f"[AUTO-HEDGE] Opening/Expanding short hedge on {symbol}, final qty={desired_qty:.4f}, ask={best_ask_price}"
                )
                await self.postonly_limit_order_bybit_async(
                    symbol=symbol,
                    side="sell",
                    amount=final_qty,
                    price=best_ask_price,
                    positionIdx=position_idx,
                    reduceOnly=False
                )
            else:
                best_bid_price = await self.get_best_bid_price_async(symbol)
                final_qty = abs(qty_diff)
                logging.info(
                    f"[AUTO-HEDGE] Reducing short hedge on {symbol} by {final_qty:.4f}, bid={best_bid_price}"
                )
                await self.postonly_limit_order_bybit_async(
                    symbol=symbol,
                    side="buy",
                    amount=final_qty,
                    price=best_bid_price,
                    positionIdx=position_idx,
                    reduceOnly=False
                )
            self.hedge_positions[symbol]['side'] = 'short'
            self.hedge_positions[symbol]['qty']  = desired_qty

        else:
            # hedge_side == 'long'
            position_idx = 1
            if qty_diff > 0:
                best_bid_price = await self.get_best_bid_price_async(symbol)
                final_qty = abs(qty_diff)
                logging.info(
                    f"[AUTO-HEDGE] Opening/Expanding long hedge on {symbol}, final qty={desired_qty:.4f}, bid={best_bid_price}"
                )
                await self.postonly_limit_order_bybit_async(
                    symbol=symbol,
                    side="buy",
                    amount=final_qty,
                    price=best_bid_price,
                    positionIdx=position_idx,
                    reduceOnly=False
                )
            else:
                best_ask_price = await self.get_best_ask_price_async(symbol)
                final_qty = abs(qty_diff)
                logging.info(
                    f"[AUTO-HEDGE] Reducing long hedge on {symbol} by {final_qty:.4f}, ask={best_ask_price}"
                )
                await self.postonly_limit_order_bybit_async(
                    symbol=symbol,
                    side="sell",
                    amount=final_qty,
                    price=best_ask_price,
                    positionIdx=position_idx,
                    reduceOnly=False
                )
            self.hedge_positions[symbol]['side'] = 'long'
            self.hedge_positions[symbol]['qty']  = desired_qty



    async def close_hedge_position_async(self, symbol):
        """
        Close any existing hedge on the SAME symbol using a post-only limit order.
        If hedge_side == 'short', we buy to close. If hedge_side == 'long', we sell to close.
        """
        if (symbol not in self.hedge_positions) or (not self.hedge_positions[symbol]['side']):
            logging.info(f"[AUTO-HEDGE] {symbol}: No existing hedge to close.")
            return

        hedge_side = self.hedge_positions[symbol]['side']
        hedge_qty  = self.hedge_positions[symbol]['qty']
        if hedge_qty <= 0:
            logging.info(f"[AUTO-HEDGE] {symbol}: Hedge qty <= 0 => nothing to close.")
            return

        if hedge_side == 'short':
            best_bid_price = await self.get_best_bid_price_async(symbol)
            logging.info(
                f"[AUTO-HEDGE] Closing short hedge on {symbol}, qty={hedge_qty:.4f}, bid={best_bid_price}"
            )
            await self.postonly_limit_order_bybit_async(
                symbol=symbol,
                side="buy",
                amount=hedge_qty,
                price=best_bid_price,
                positionIdx=2,
                reduceOnly=False
            )
        else:
            # hedge_side == 'long'
            best_ask_price = await self.get_best_ask_price_async(symbol)
            logging.info(
                f"[AUTO-HEDGE] Closing long hedge on {symbol}, qty={hedge_qty:.4f}, ask={best_ask_price}"
            )
            await self.postonly_limit_order_bybit_async(
                symbol=symbol,
                side="sell",
                amount=hedge_qty,
                price=best_ask_price,
                positionIdx=1,
                reduceOnly=False
            )

        self.hedge_positions[symbol]['side'] = None
        self.hedge_positions[symbol]['qty']  = 0.0

    async def lineargrid_base_async(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                        strength: float, min_outer_price_distance: float, max_outer_price_distance_long: float, max_outer_price_distance_short: float, reissue_threshold: float,
                        wallet_exposure_limit_long: float, wallet_exposure_limit_short: float, long_mode: bool,
                        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                        max_qty_percent_long: float, max_qty_percent_short: float, graceful_stop_long: bool, graceful_stop_short: bool,
                        additional_entries_from_signal: bool, open_position_data: list, drawdown_behavior: str, grid_behavior: str,
                        stop_loss_long: float, stop_loss_short: float, stop_loss_enabled: bool):
        try:
            long_pos_qty = long_pos_qty if long_pos_qty is not None else 0
            short_pos_qty = short_pos_qty if short_pos_qty is not None else 0

            # Ensure long_pos_qty and short_pos_qty are floats
            try:
                long_pos_qty = float(long_pos_qty)
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid value for long_pos_qty: {long_pos_qty}, Error: {e}")
                long_pos_qty = 0

            try:
                short_pos_qty = float(short_pos_qty)
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid value for short_pos_qty: {short_pos_qty}, Error: {e}")
                short_pos_qty = 0

            spread, current_price = await self.get_spread_and_price_async(symbol)
            if not spread:
                logging.warning(f"[{symbol}] no spread")
                return
            if not current_price:
                logging.warning(f"[{symbol}] no current price")
                return
            # dynamic_outer_price_distance = self.calculate_dynamic_outer_price_distance(spread, min_outer_price_distance, max_outer_price_distance)

            dynamic_outer_price_distance_long, dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(
                spread, min_outer_price_distance, max_outer_price_distance_long, max_outer_price_distance_short
            )

            # should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = await self.retry_api_call_async(self.exchange.get_open_orders_async, symbol)

            self.initialize_filled_levels(symbol)
            long_grid_active, short_grid_active = self.check_grid_active(symbol, open_orders)

            logging.info(f"Long grid active: {long_grid_active}")
            logging.info(f"Short grid active: {short_grid_active}")

            await self.check_and_manage_positions_async(
                long_pos_qty,
                short_pos_qty,
                symbol,
                total_equity,
                current_price,
                wallet_exposure_limit_long,
                wallet_exposure_limit_short,
                max_qty_percent_long,
                max_qty_percent_short
            )

            buffer_percentage_long, buffer_percentage_short = self.calculate_buffer_percentages(long_pos_qty, short_pos_qty, current_price, long_pos_price, short_pos_price, initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage)
            buffer_distance_long, buffer_distance_short = self.calculate_buffer_distances(current_price, buffer_percentage_long, buffer_percentage_short)

            order_book, best_ask_price, best_bid_price = await self.get_order_book_prices_async(symbol, current_price)
            # min_price, max_price, price_range, volume_histogram_long, volume_histogram_short = self.calculate_price_range_and_volume_histograms(order_book, current_price, max_outer_price_distance)

            logging.info(f"[{symbol}] calculate_price_range_and_volume_histograms start")
            min_price_long, max_price_long, price_range_long, volume_histogram_long, min_price_short, max_price_short, price_range_short, volume_histogram_short = self.calculate_price_range_and_volume_histograms(order_book, current_price, max_outer_price_distance_long, max_outer_price_distance_short)
            logging.info(f"[{symbol}] calculate_price_range_and_volume_histograms end")

            # Use the correct price range for each histogram
            volume_threshold_long, significant_levels_long = self.calculate_volume_thresholds_and_significant_levels(volume_histogram_long, price_range_long)
            volume_threshold_short, significant_levels_short = self.calculate_volume_thresholds_and_significant_levels(volume_histogram_short, price_range_short)

            # Call dbscan_classification if grid_behavior is set to "dbscanalgo"
            if grid_behavior == "dbscanalgo":
                initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)
                zigzag_length = 5  # Example value; replace with actual config value
                epsilon_deviation = 2.5  # Example value; replace with actual config value
                aggregate_range = 5  # Example value; replace with actual config value

                # Extract high, low, and volume data from order book
                highs = [float(order[0]) for order in order_book['bids']]  # Using bids for highs
                lows = [float(order[0]) for order in order_book['asks']]   # Using asks for lows
                volumes = [float(order[1]) for order in order_book['bids'] + order_book['asks']]  # Combining volumes from bids and asks

                # Create OHLCV format expected by dbscan_classification
                ohlcv_data = [{'high': high, 'low': low, 'volume': volume} for high, low, volume in zip(highs, lows, volumes)]

                # Correct function call with four arguments
                logging.info(f"[{symbol}] dbscan_classification start")
                support_resistance_levels = self.dbscan_classification(ohlcv_data, zigzag_length, epsilon_deviation, aggregate_range)
                logging.info(f"[{symbol}] dbscan_classification end")

                # initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)
                # Extract levels from the dbscan classification results
                grid_levels_long = [level['level'] for level in support_resistance_levels if level['level'] >= current_price]
                grid_levels_short = [level['level'] for level in support_resistance_levels if level['level'] <= current_price]

            else:
                initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)
                # grid_levels_long, grid_levels_short = self.calculate_grid_levels(long_pos_qty, short_pos_qty, levels, initial_entry_long, initial_entry_short, current_price, buffer_distance_long, buffer_distance_short, max_outer_price_distance)
                grid_levels_long, grid_levels_short = self.calculate_grid_levels(
                    long_pos_qty, short_pos_qty, levels,
                    initial_entry_long, initial_entry_short,
                    current_price,
                    buffer_distance_long, buffer_distance_short,
                    max_outer_price_distance_long, max_outer_price_distance_short
                )

            # adjusted_grid_levels_long = self.adjust_grid_levels(grid_levels_long, significant_levels_long, tolerance=0.01, min_outer_price_distance=min_outer_price_distance, max_outer_price_distance=max_outer_price_distance, current_price=current_price, levels=levels)
            # adjusted_grid_levels_short = self.adjust_grid_levels(grid_levels_short, significant_levels_short, tolerance=0.01, min_outer_price_distance=min_outer_price_distance, max_outer_price_distance=max_outer_price_distance, current_price=current_price, levels=levels)

            adjusted_grid_levels_long = self.adjust_grid_levels(
                grid_levels_long,
                significant_levels_long,
                tolerance=0.01,
                min_outer_price_distance=min_outer_price_distance,
                max_outer_price_distance_long=max_outer_price_distance_long,  # Update here
                max_outer_price_distance_short=max_outer_price_distance_short,  # Update here
                current_price=current_price,
                levels=levels
            )

            adjusted_grid_levels_short = self.adjust_grid_levels(
                grid_levels_short,
                significant_levels_short,
                tolerance=0.01,
                min_outer_price_distance=min_outer_price_distance,
                max_outer_price_distance_long=max_outer_price_distance_long,  # Update here
                max_outer_price_distance_short=max_outer_price_distance_short,  # Update here
                current_price=current_price,
                levels=levels
            )

            # grid_levels_long, grid_levels_short = self.finalize_grid_levels(adjusted_grid_levels_long, adjusted_grid_levels_short, levels, current_price, buffer_distance_long, buffer_distance_short, max_outer_price_distance, initial_entry_long, initial_entry_short)

            grid_levels_long, grid_levels_short = self.finalize_grid_levels(
                adjusted_grid_levels_long, adjusted_grid_levels_short,
                levels, current_price,
                buffer_distance_long, buffer_distance_short,
                max_outer_price_distance_long, max_outer_price_distance_short,
                initial_entry_long, initial_entry_short
            )

            qty_precision, min_qty = await self.get_precision_and_min_qty_async(symbol)

            # Apply drawdown behavior based on configuration
            if drawdown_behavior == "full_distribution":
                logging.info(f"Activating full distribution drawdown behavior for {symbol}")

                # Calculate order amounts for aggressive drawdown with strength
                amounts_long = self.calculate_order_amounts_aggressive_drawdown(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='buy', strength=strength, long_pos_qty=long_pos_qty)
                logging.info(f"[{symbol}] calculated long - {amounts_long}")

                amounts_short = self.calculate_order_amounts_aggressive_drawdown(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='sell', strength=strength, short_pos_qty=short_pos_qty
                )
                logging.info(f"[{symbol}] calculated short - {amounts_short}")

            elif drawdown_behavior == "progressive_drawdown":
                logging.info(f"Activating progressive drawdown behavior for {symbol}")

                # Calculate order amounts for progressive drawdown using progressive distribution
                amounts_long = self.calculate_order_amounts_progressive_distribution(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='buy', strength=strength, long_pos_qty=long_pos_qty
                )
                amounts_short = self.calculate_order_amounts_progressive_distribution(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='sell', strength=strength, short_pos_qty=short_pos_qty
                )

            else:
                logging.info(f"Applying standard grid behavior for {symbol}")

                # Calculate the total amounts without aggressive or progressive drawdown behaviors
                total_amount_long = self.calculate_total_amount_refactor(
                    symbol,
                    total_equity,
                    best_ask_price,
                    best_bid_price,
                    wallet_exposure_limit_long,
                    wallet_exposure_limit_short,
                    "buy",
                    levels,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    long_mode
                )

                total_amount_short = self.calculate_total_amount_refactor(
                    symbol,
                    total_equity,
                    best_ask_price,
                    best_bid_price,
                    wallet_exposure_limit_long,
                    wallet_exposure_limit_short,
                    "sell",
                    levels,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    short_mode
                )

                amounts_long = self.calculate_order_amounts_refactor(
                    symbol,
                    total_amount_long,
                    levels,
                    strength,
                    qty_precision,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    'buy'
                )

                amounts_short = self.calculate_order_amounts_refactor(
                    symbol,
                    total_amount_short,
                    levels,
                    strength,
                    qty_precision,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    'sell'
                )


            return await self.handle_grid_trades_async(
                symbol,
                grid_levels_long,
                grid_levels_short,
                long_grid_active,
                short_grid_active,
                long_pos_qty,
                short_pos_qty,
                current_price,
                dynamic_outer_price_distance_long,
                dynamic_outer_price_distance_short,
                min_outer_price_distance,
                buffer_percentage_long,
                buffer_percentage_short,
                adjusted_grid_levels_long,
                adjusted_grid_levels_short,
                levels,
                amounts_long,
                amounts_short,
                best_bid_price,
                best_ask_price,
                mfirsi_signal,
                open_orders,
                initial_entry_buffer_pct,
                reissue_threshold,
                entry_during_autoreduce,
                min_qty,
                open_symbols,
                symbols_allowed,
                long_mode,
                short_mode,
                long_pos_price,
                short_pos_price,
                graceful_stop_long,
                graceful_stop_short,
                min_buffer_percentage,
                max_buffer_percentage,
                additional_entries_from_signal,
                open_position_data,
                upnl_profit_pct,
                max_upnl_profit_pct,
                tp_order_counts,
                stop_loss_long,
                stop_loss_short,
                stop_loss_enabled,
            )

        except Exception as e:
            logging.error(f"[{symbol}] Error in executing gridstrategy: {e}")
            logging.error("Traceback: %s", traceback.format_exc())

    async def postonly_limit_order_bybit_async(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        """Directly places the order with the exchange."""
        params = {"reduceOnly": reduceOnly, "postOnly": True}
        order = await self.exchange.create_limit_order_bybit_async(symbol, side, amount, price, positionIdx=positionIdx, params=params)

        # Log and store the order ID if the order was placed successfully
        if order and 'id' in order:
            logging.info(f"Successfully placed post-only limit order for {symbol}. Order ID: {order['id']}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")
            if symbol not in self.order_ids:
                self.order_ids[symbol] = []
            self.order_ids[symbol].append(order['id'])
        else:
            logging.warning(f"Failed to place post-only limit order for {symbol}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")

        return order

    async def postonly_limit_order_bybit_nolimit_async(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly, "postOnly": True}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = await self.exchange.create_limit_order_bybit_async(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Nolimit postonly order result for {symbol}: {order}")
            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.info(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    async def liq_stop_loss_logic_async(self, long_pos_qty, long_pos_price, long_liquidation_price, short_pos_qty, short_pos_price, short_liquidation_price, liq_stoploss_enabled, symbol, liq_price_stop_pct):
        if liq_stoploss_enabled:
            try:
                current_price = await self.exchange.get_current_price_async(symbol)

                # Stop loss logic for long positions
                if long_pos_qty > 0 and long_liquidation_price:
                    # Convert to float if it's not None or empty string
                    long_liquidation_price = float(long_liquidation_price) if long_liquidation_price else None

                    if long_liquidation_price:
                        long_stop_loss_price = self.calculate_long_stop_loss_based_on_liq_price(
                            long_pos_price, long_liquidation_price, liq_price_stop_pct)
                        if long_stop_loss_price and current_price <= long_stop_loss_price:
                            # Place stop loss order for long position
                            logging.info(f"Placing long stop loss order for {symbol} at {long_stop_loss_price}")
                            await self.postonly_limit_order_bybit_nolimit_async(symbol, "sell", long_pos_qty, long_stop_loss_price, positionIdx=1, reduceOnly=True)

                # Stop loss logic for short positions
                if short_pos_qty > 0 and short_liquidation_price:
                    # Convert to float if it's not None or empty string
                    short_liquidation_price = float(short_liquidation_price) if short_liquidation_price else None

                    if short_liquidation_price:
                        short_stop_loss_price = self.calculate_short_stop_loss_based_on_liq_price(
                            short_pos_price, short_liquidation_price, liq_price_stop_pct)
                        if short_stop_loss_price and current_price >= short_stop_loss_price:
                            # Place stop loss order for short position
                            logging.info(f"Placing short stop loss order for {symbol} at {short_stop_loss_price}")
                            await self.postonly_limit_order_bybit_nolimit_async(symbol, "buy", short_pos_qty, short_stop_loss_price, positionIdx=2, reduceOnly=True)
            except Exception as e:
                logging.info(f"Exception caught in liquidation stop loss logic for {symbol}: {e}")

    async def stop_loss_logic_async(self, long_pos_qty, long_pos_price, short_pos_qty, short_pos_price, stoploss_enabled, symbol, stoploss_upnl_pct):
        if stoploss_enabled:
            try:
                # Initial stop loss calculation
                initial_short_stop_loss = self.calculate_quickscalp_short_stop_loss(short_pos_price, symbol, stoploss_upnl_pct) if short_pos_price else None
                initial_long_stop_loss = self.calculate_quickscalp_long_stop_loss(long_pos_price, symbol, stoploss_upnl_pct) if long_pos_price else None

                current_price = await self.exchange.get_current_price_async(symbol)
                order_book = await self.exchange.get_orderbook_async(symbol)
                current_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else None
                current_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else None

                # Calculate and set stop loss for long positions
                if long_pos_qty > 0 and long_pos_price and initial_long_stop_loss:
                    threshold_for_long = long_pos_price - (long_pos_price - initial_long_stop_loss) * 0.1
                    if current_price <= threshold_for_long:
                        adjusted_long_stop_loss = initial_long_stop_loss if current_price > initial_long_stop_loss else current_bid_price
                        logging.info(f"Setting long stop loss for {symbol} at {adjusted_long_stop_loss}")
                        await self.postonly_limit_order_bybit_nolimit_async(symbol, "sell", long_pos_qty, adjusted_long_stop_loss, positionIdx=1, reduceOnly=True)

                # Calculate and set stop loss for short positions
                if short_pos_qty > 0 and short_pos_price and initial_short_stop_loss:
                    threshold_for_short = short_pos_price + (initial_short_stop_loss - short_pos_price) * 0.1
                    if current_price >= threshold_for_short:
                        adjusted_short_stop_loss = initial_short_stop_loss if current_price < initial_short_stop_loss else current_ask_price
                        logging.info(f"Setting short stop loss for {symbol} at {adjusted_short_stop_loss}")
                        await self.postonly_limit_order_bybit_nolimit_async(symbol, "buy", short_pos_qty, adjusted_short_stop_loss, positionIdx=2, reduceOnly=True)
            except Exception as e:
                logging.info(f"Exception caught in stop loss functionality for {symbol}: {e}")
