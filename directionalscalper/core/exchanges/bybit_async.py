import json
import random
import time
from typing import List

import pandas as pd
from directionalscalper.core.exchanges.bybit import BybitExchange
import logging
import ccxt.async_support as ccxt_async
import traceback
from directionalscalper.core.strategies.logger import Logger
import asyncio

from rate_limit import RateLimitAsync

logging = Logger(logger_name="BybitExchangeAsync", filename="BybitExchangeAsync.log", stream=True)

class BybitExchangeAsync(BybitExchange):
    market_loaded_at = None
    MARKET_LOAD_TIMEOUT = 30

    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap', collateral_currency='USDT'):
        super().__init__(api_key, secret_key, passphrase, market_type)

        self.collateral_currency = collateral_currency
        self.rate_limiter_async = RateLimitAsync(10, 1)

        self.exchange_async = ccxt_async.bybit(
            {
                "apiKey": api_key,
                "secret": secret_key,
            }
        )

    async def setup_exchange_async(self, symbol) -> None:
        # values = {"position": False, "leverage": False}
        try:
            # Set the position mode to hedge
            await self.exchange_async.set_position_mode(hedged=True, symbol=symbol)
            # values["position"] = True
        except Exception as e:
            logging.info(f"An unknown error occurred in with set_position_mode: {e}")

    async def get_current_max_leverage_async(self, symbol):
        try:
            # Fetch leverage tiers for the symbol
            leverage_tiers = await self.exchange_async.fetch_market_leverage_tiers(symbol)

            # Process leverage tiers to find the maximum leverage
            max_leverage = max([tier['maxLeverage'] for tier in leverage_tiers])
            logging.info(f"Maximum leverage for symbol {symbol}: {max_leverage}")

            return max_leverage

        except Exception as e:
            logging.info(f"Error retrieving leverage tiers for {symbol}: {e}")
            return None

    async def set_leverage_async(self, leverage, symbol):
        try:
            await self.exchange_async.set_leverage(leverage, symbol)
            logging.info(f"Leverage set to {leverage} for symbol {symbol}")
        except Exception as e:
            logging.info(f"Error setting leverage: {e}")

    async def set_symbol_to_cross_margin_async(self, symbol, leverage):
        """
        Set a specific symbol's margin mode to cross with specified leverage.
        """
        try:
            response = await self.exchange_async.set_margin_mode('cross', symbol=symbol, params={'leverage': leverage})

            retCode = response.get('retCode') if isinstance(response, dict) else None

            if retCode == 110026:  # Margin mode is already set to cross
                logging.info(f"Symbol {symbol} is already set to cross margin mode. No changes made.")
                return {"status": "unchanged", "response": response}
            else:
                logging.info(f"Margin mode set to cross for symbol {symbol} with leverage {leverage}. Response: {response}")
                return {"status": "changed", "response": response}

        except Exception as e:
            logging.info(f"Failed to set margin mode or margin mode already set to cross for symbol {symbol} with leverage {leverage}: {e}")
            return {"status": "error", "message": str(e)}

    async def get_all_open_positions_async(self, retries=10, delay_factor=10, max_delay=60) -> List[dict]:
        for attempt in range(retries):
            try:
                all_positions = await self.exchange_async.fetch_positions(params={'paginate': True})
                cursor = all_positions[0]['info']['nextPageCursor']
                i = 0

                while True and i < 10:
                    i += 1
                    positions = await self.exchange_async.fetch_positions(params={'cursor': cursor, 'paginate': True})
                    if len(positions) == 0:
                        break
                    all_positions.extend(positions)
                    cursor = positions[0]['info']['nextPageCursor']

                open_positions = [position for position in all_positions if float(position.get('contracts', 0)) != 0]
                return open_positions
            except Exception as e:
                is_rate_limit_error = "Too many visits" in str(e) or (hasattr(e, 'response') and e.response.status_code == 403)

                if is_rate_limit_error and attempt < retries - 1:
                    delay = min(delay_factor * (attempt + 1), max_delay)  # Exponential delay with a cap
                    logging.info(f"Rate limit on get_all_open_positions_bybit hit, waiting for {delay} seconds before retrying...")
                    await self.exchange_async.sleep(delay * 1000)
                    continue
                else:
                    logging.info(f"Error fetching open positions: {e}")
                    return []

    async def get_balance_async(self):
        try:
            # Fetch the balance with params to specify the account type if needed
            balance_response = await self.exchange_async.fetch_balance({'type': 'swap'})

            # Log the raw response for debugging purposes
            # logging.info(f"Raw balance response from Bybit: {balance_response}")

            return self.parse_balance(balance_response)
        except Exception as e:
            logging.info(f"Error fetching balance from Bybit: {e}")
            return None, None

    def parse_balance(self, balance_response):
        total_balance = None
        available_balance = None

        if self.collateral_currency == 'all':
            if 'info' in balance_response:
                available_balance = balance_response['info']['result']['list'][0]['totalAvailableBalance']
                available_balance = float(available_balance)
                total_balance = balance_response['info']['result']['list'][0]['totalEquity']
        else:
            # Check for the required keys in the response
            if 'free' in balance_response and self.collateral_currency in balance_response['free']:
                # Return the available balance for the specified currency
                available_balance = float(balance_response['free'][self.collateral_currency])
            else:
                logging.warning(f"Available balance for {self.collateral_currency} not found in the response.")

             # Parse the balance
            if self.collateral_currency in balance_response['total']:
                total_balance = balance_response['total'][self.collateral_currency]
            else:
                logging.info(f"Balance for {self.collateral_currency} not found in the response.")

        return total_balance, available_balance

    async def fetch_ohlcv_async(self, symbol, timeframe='1d', limit=None, max_retries=100, base_delay=10, max_delay=60, params=None):
        """
        Fetch OHLCV data for the given symbol and timeframe.

        :param symbol: Trading symbol.
        :param timeframe: Timeframe string.
        :param limit: Limit the number of returned data points.
        :param max_retries: Maximum number of retries for API calls.
        :param base_delay: Base delay for exponential backoff.
        :param max_delay: Maximum delay for exponential backoff.
        :return: DataFrame with OHLCV data or an empty DataFrame on error.
        """
        retries = 0

        while retries < max_retries:
            try:
                async with self.rate_limiter_async:
                    # Fetch the OHLCV data from the exchange
                    ohlcv = await self.exchange_async.fetch_ohlcv(symbol, timeframe, limit=limit, params=params)  # Pass the limit parameter

                    # Create a DataFrame from the OHLCV data
                    return self.convert_ohlcv_to_df(ohlcv)

            except ccxt_async.RateLimitExceeded as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

            except ccxt_async.BadSymbol as e:
                # Handle the BadSymbol error gracefully and exit the loop
                logging.info(f"Bad symbol: {symbol}. Error: {e}")
                break  # Exit the retry loop as the symbol is invalid

            except ccxt_async.BaseError as e:
                # Log the error message
                logging.info(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
                logging.error(traceback.format_exc())
                return pd.DataFrame()  # Return empty DataFrame for other base errors

            except Exception as e:
                # Log the error message and traceback
                logging.info(f"Unexpected error occurred while fetching OHLCV data: {e}")
                logging.error(traceback.format_exc())

                # Handle specific error scenarios
                if isinstance(e, TypeError) and 'string indices must be integers' in str(e):
                    logging.info(f"TypeError occurred: {e}")
                    logging.info(f"Response content: {self.exchange.last_http_response}")

                    try:
                        response = json.loads(self.exchange.last_http_response)
                        logging.info(f"Parsed response into a dictionary: {response}")
                    except json.JSONDecodeError as json_error:
                        logging.info(f"Failed to parse response: {json_error}")

                return pd.DataFrame()  # Return empty DataFrame on unexpected errors

        logging.error(f"Failed to fetch OHLCV data after {max_retries} retries.")
        return pd.DataFrame()

    # NOTE: these are syncronous, but support full balance with all collateral assets
    def get_available_balance_bybit(self):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type
                balance_response = self.exchange.fetch_balance({'type': 'swap'})
                # Log the raw response for debugging purposes
                #logging.info(f"Raw available balance response from Bybit: {balance_response}")

                if self.collateral_currency == 'all' and 'info' in balance_response:
                    available_balance = balance_response['info']['result']['list'][0]['totalAvailableBalance']
                    return float(available_balance)

                # Check for the required keys in the response
                if 'free' in balance_response and self.collateral_currency in balance_response['free']:
                    # Return the available balance for the specified currency
                    return float(balance_response['free'][self.collateral_currency])
                else:
                    logging.warning(f"Available balance for {self.collateral_currency} not found in the response.")

            except Exception as e:
                logging.info(f"Error fetching available balance from Bybit: {e}")
        return None

    def get_futures_balance_bybit(self):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type if needed
                balance_response = self.exchange.fetch_balance({'type': 'swap'})
                # Log the raw response for debugging purposes
                #logging.info(f"Raw balance response from Bybit: {balance_response}")

                if self.collateral_currency == 'all' and 'info' in balance_response:
                    logging.info("quote is not set - pulling total balance from total equity")

                    total_balance = balance_response['info']['result']['list'][0]['totalWalletBalance']
                    return total_balance

                # Parse the balance
                if self.collateral_currency in balance_response['total']:
                    total_balance = balance_response['total'][self.collateral_currency]
                    return total_balance
                else:
                    logging.info(f"Balance for {self.collateral_currency} not found in the response.")
            except Exception as e:
                logging.info(f"Error fetching balance from Bybit: {e}")

        return None

    # NOTE: this is syncronous, but works properly if only 1 position is returned
    def get_positions_bybit(self, symbol, max_retries=100, retry_delay=5) -> dict:
        values = {
            "long": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
                'balance': 0
            },
            "short": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
                'balance': 0
            },
        }

        for i in range(max_retries):
            try:
                data = self.exchange.fetch_positions(symbol)
                positions = {
                    'long': next((position for position in data if position['side'] == 'long' and float(position.get('contracts', 0)) != 0), None),
                    'short': next((position for position in data if position['side'] == 'short' and float(position.get('contracts', 0)) != 0), None)
                }

                for side, position in positions.items():
                    if not position:
                        continue
                    values[side]["qty"] = float(position["contracts"])
                    values[side]["price"] = float(position["info"]["avgPrice"] or 0)
                    values[side]["realised"] = round(float(position["info"]["unrealisedPnl"] or 0), 4)
                    values[side]["cum_realised"] = round(float(position["info"]["cumRealisedPnl"] or 0), 4)
                    values[side]["upnl"] = round(float(position["info"]["unrealisedPnl"] or 0), 4)
                    values[side]["upnl_pct"] = round(float(position["percentage"] or 0), 4)
                    values[side]["liq_price"] = float(position["liquidationPrice"] or 0)
                    values[side]["entry_price"] = float(position["entryPrice"] or 0)
                    values[side]["balance"] = float(position["info"]["positionBalance"] or 0)
                break  # If the fetch was successful, break out of the loop
            except Exception as e:
                if i < max_retries - 1:  # If not the last attempt
                    logging.info(f"An unknown error occurred in get_positions_bybit(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.info(f"Failed to fetch positions after {max_retries} attempts: {e}")
                    raise e  # If it's still failing after max_retries, re-raise the exception.

        return values

    async def get_all_open_orders_async(self):
        """
        Fetch all open orders for all symbols from the Bybit API.

        :return: A list of open orders for all symbols.
        """
        try:
            all_open_orders = await self.exchange_async.fetch_open_orders(params={'paginate': True})
            return all_open_orders
        except Exception as e:
            print(f"An error occurred while fetching all open orders: {e}")
            return []

    async def cancel_order_by_id_async(self, order_id, symbol):
        try:
            # Call the updated cancel_order method
            result = await self.exchange_async.cancel_order(id=order_id, symbol=symbol)
            logging.info(f"Canceled order - ID: {order_id}, Response: {result}")
        except Exception as e:
            logging.info(f"Error occurred in cancel_order_by_id_async: {e}")
