import time
from decimal import Decimal, ROUND_HALF_UP
from ..strategy import Strategy
from typing import Tuple

class BybitHedgeGridStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager
        self.last_cancel_time = 0

    def limit_order(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def place_grid_orders(self, max_trade_qty, entry_amount, symbol, direction, start_price, end_price, positionIdx):
        try:
            price_precision = int(self.exchange.get_price_precision(symbol))

            if max_trade_qty < entry_amount:
                print("Max trade quantity is less than the minimum order size, cannot place orders.")
                return

            # Calculate the number of orders that can be placed with the provided max trade quantity
            num_orders = int(max_trade_qty / entry_amount)

            if num_orders <= 0:
                print("Insufficient funds to place any orders.")
                return

            prices = [start_price + i * (end_price - start_price) / (num_orders - 1) for i in range(num_orders)]
            amounts = [entry_amount for _ in range(num_orders)]
                
            for price, amt in zip(prices, amounts):
                # Round the price and ensure amount is not less than minimum required
                price = round(price, price_precision)  # Using fetched price precision to round
                amt = max(amt, 5)  # Ensuring the amount is not less than 5

                try:
                    print(f"Trying to place order with price: {price} and amount: {amt}") 
                    self.limit_order(symbol, direction, amt, price, positionIdx, reduceOnly=False)
                except Exception as e:
                    print(f"Exception while placing limit order: {e}")
        except Exception as e:
            print(f"Exception in grid order placement: {e}")


#self.limit_order(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)
    def get_open_take_profit_order_quantity(self, orders, side):
        for order in orders:
            if order['side'].lower() == side.lower() and order['reduce_only']:
                return order['qty'], order['id']
        return None, None

    def cancel_take_profit_orders(self, symbol, side):
        self.exchange.cancel_close_bybit(symbol, side)

    def calculate_short_take_profit(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            short_target_price = short_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            long_target_price = long_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume

        print("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)
        print("Set up exchange")

        while True:
            print(f"Bybit hedge grid strategy running")
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get API data
            data = self.manager.get_data()
            one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
            five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
            thirty_minute_distance = self.manager.get_asset_value(symbol, data, "30mSpread")
            trend = self.manager.get_asset_value(symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"Trend: {trend}")

            quote_currency = "USDT"
            dex_equity = self.exchange.get_balance_bybit(quote_currency)

            print(f"Total equity: {dex_equity}")

            current_price = self.exchange.get_current_price(symbol)
            market_data = self.exchange.get_market_data_bybit(symbol)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            price_precision = int(self.exchange.get_price_precision(symbol))
            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")
            print(f"Current price: {current_price}")
            print(f"Price precision: {price_precision}")

            leverage = float(market_data["leverage"]) if market_data["leverage"] !=0 else 50.0

            max_trade_qty = round(
                (float(dex_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )            
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            min_qty_bybit = market_data["min_qty"]
            print(f"Min qty: {min_qty_bybit}")

            if float(amount) < min_qty_bybit:
                print(f"The amount you entered ({amount}) is less than the minimum required by Bybit for {symbol}: {min_qty_bybit}.")
                break
            else:
                print(f"The amount you entered ({amount}) is valid for {symbol}")

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            print(f"{ma_3_low}")
            print(f"{ma_6_low}")

            position_data = self.exchange.get_positions_bybit(symbol)

            #print(f"Bybit pos data: {position_data}")

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]


            print(f"Short pos qty: {short_pos_qty}")
            print(f"Long pos qty: {long_pos_qty}")

            short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
            long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

            print(f"Long pos price {long_pos_price}")
            print(f"Short pos price {short_pos_price}")

            # Take profit calc
            short_take_profit = self.calculate_short_take_profit(short_pos_price, symbol)
            long_take_profit = self.calculate_long_take_profit(long_pos_price, symbol)

            # Dabbling with precision
            #price_precision, quantity_precision = self.exchange.get_symbol_precision_bybit(symbol)

            price_precision = int(self.exchange.get_price_precision(symbol))

            rounded_short_qty = None
            rounded_long_qty = None

            if short_pos_qty is not None:
                rounded_short_qty = round(short_pos_qty, price_precision)

            if long_pos_qty is not None:
                rounded_long_qty = round(long_pos_qty, price_precision)

            print(f"Rounded short qty: {rounded_short_qty}")
            print(f"Rounded long qty: {rounded_long_qty}")

            print(f"Short take profit: {short_take_profit}")
            print(f"Long take profit: {long_take_profit}")

            should_short = best_bid_price > ma_3_high
            should_long = best_bid_price < ma_3_high

            should_add_to_short = False
            should_add_to_long = False
            
            if short_pos_price is not None:
                should_add_to_short = short_pos_price < ma_6_low

            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_low

            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")
            
            open_orders = self.exchange.get_open_orders(symbol)

            # Call the get_open_take_profit_order_quantity function for the 'buy' side
            buy_qty, buy_id = self.get_open_take_profit_order_quantity(open_orders, 'buy')

            # Call the get_open_take_profit_order_quantity function for the 'sell' side
            sell_qty, sell_id = self.get_open_take_profit_order_quantity(open_orders, 'sell')

            # Print the results
            print("Buy Take Profit Order - Quantity: ", buy_qty, "ID: ", buy_id)
            print("Sell Take Profit Order - Quantity: ", sell_qty, "ID: ", sell_id)

            # Grid logic 
            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            
                            print(f"Placing initial long grid")
                            try:
                                #place_grid_orders(max_trade_qty, entry_amount, symbol, direction, start_price, end_price, positionIdx)
                                self.place_grid_orders(max_trade_qty, amount, symbol, "buy", best_bid_price - thirty_minute_distance, best_bid_price, 1)
                            except Exception as e:
                                print(f"Exception caught {e}")

                            print(f"Placed initial long grid")
                            time.sleep(0.05)
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:

                                print(f"Placed additional long grid")
                                self.place_grid_orders(max_trade_qty, amount, symbol, "buy", best_bid_price - thirty_minute_distance, best_bid_price, 1)
                                time.sleep(0.05)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:
                            
                            print(f"Placing initial short grid")
                            self.place_grid_orders(max_trade_qty, amount, symbol, "sell", best_ask_price + thirty_minute_distance, best_ask_price, 2)
                            print("Placed initial short grid")
                            time.sleep(0.05)
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                self.place_grid_orders(max_trade_qty, amount, symbol, "sell", best_ask_price + thirty_minute_distance, best_ask_price, 2)
                                time.sleep(0.05)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tp_qty, existing_long_tp_id = self.get_open_take_profit_order_quantity(open_orders, "close_long")
                if existing_long_tp_qty is None or existing_long_tp_qty != long_pos_qty:
                    try:
                        if existing_long_tp_id is not None:
                            self.cancel_take_profit_orders(symbol, "long")
                            print(f"Long take profit canceled")
                            time.sleep(0.05)

                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tp_qty, existing_short_tp_id = self.get_open_take_profit_order_quantity(open_orders, "close_short")
                if existing_short_tp_qty is None or existing_short_tp_qty != short_pos_qty:
                    try:
                        if existing_short_tp_id is not None:
                            self.cancel_take_profit_orders(symbol, "short")
                            print(f"Short take profit canceled")
                            time.sleep(0.05)

                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "buy", short_pos_qty, short_take_profit, positionIdx=2, reduce_only=True)
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            # Cancel entries
            current_time = time.time()
            if current_time - self.last_cancel_time >= 1800:  # Execute this block every 30 minutes or half an hour
                try:
                    if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                        self.exchange.cancel_all_entries_bybit(symbol)
                        print(f"Canceled entry orders for {symbol}")
                        time.sleep(0.05)
                except Exception as e:
                    print(f"An error occurred while canceling entry orders: {e}")

                self.last_cancel_time = current_time  # Update the last cancel time


            time.sleep(30)