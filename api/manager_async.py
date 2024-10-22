from __future__ import annotations
import asyncio

import fnmatch
from datetime import datetime, timedelta


from api.manager import Manager
from directionalscalper.core.utils import get_request_async
from directionalscalper.core.strategies.logger import Logger

logging = Logger(logger_name="ManagerAsync", filename="ManagerAsync.log", stream=True)

#log = logging.getLogger(__name__)


class ManagerAsync(Manager):
    async def fetch_data_from_url_async(self, url, max_retries: int = 5):
        current_time = datetime.now()
        if current_time <= self.data_cache_expiry:
            return self.data
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(60, delay)  # cap the delay to 60 seconds
            try:
                header, raw_json = await get_request_async(url=url)
                # Update the data cache and its expiry time
                self.data = raw_json
                self.data_cache_expiry = current_time + timedelta(seconds=self.cache_life_seconds)
                return raw_json
            except Exception as e:
                logging.error(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                await asyncio.sleep(delay)

        # Return cached data if all retries fail
        return self.data

    async def get_auto_rotate_symbols_async(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        if self.rotator_symbols_cache and not self.is_cache_expired():
            return self.rotator_symbols_cache

        symbols = []
        url = f"https://api.quantumvoid.org/volumedata/rotatorsymbols_{self.data_source_exchange.replace('_', '')}.json"

        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 30 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = await get_request_async(url=url)

                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} assets from API")

                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        usd_price = asset.get("Price", float('inf'))

                        logging.info(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
                            logging.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if whitelist and symbol not in whitelist:
                            logging.debug(f"Skipping {symbol} as it's not in whitelist")
                            continue

                        # Check against the max_usd_value, if provided
                        if max_usd_value is not None and usd_price > max_usd_value:
                            logging.debug(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
                            continue

                        logging.debug(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            symbols.append(symbol)

                    logging.info(f"Returning {len(symbols)} symbols")

                    # If successfully fetched, update the cache and its expiry time
                    if symbols:
                        self.rotator_symbols_cache = symbols
                        self.rotator_symbols_cache_expiry = datetime.now() + timedelta(seconds=self.cache_life_seconds)

                    return symbols

                else:
                    logging.warning("Unexpected data format. Expected a list of assets.")

            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                await asyncio.sleep(delay)

        # Return cached symbols if all retries fail
        logging.warning(f"Couldn't fetch rotator symbols after {max_retries} attempts. Using cached symbols.")
        return self.rotator_symbols_cache or []

    async def get_api_data_async(self, symbol):
        api_data_url = f"https://api.quantumvoid.org/volumedata/quantdatav2_{self.data_source_exchange.replace('_', '')}.json"
        data = await self.fetch_data_from_url_async(api_data_url)
        if not data:
            return None
        symbols = [asset.get("Asset", "") for asset in data if "Asset" in asset]

        # Fetch funding rate data from the new URL
        funding_data_url = f"https://api.quantumvoid.org/volumedata/funding_{self.data_source_exchange.replace('_', '')}.json"
        funding_data = await self.fetch_data_from_url_async(funding_data_url)

        #logging.info(f"Funding data: {funding_data}")

        api_data = {
            '1mVol': self.get_asset_value(symbol, data, "1mVol"),
            '5mVol': self.get_asset_value(symbol, data, "5mVol"),
            '1hVol': self.get_asset_value(symbol, data, "1hVol"),
            '1mSpread': self.get_asset_value(symbol, data, "1mSpread"),
            '5mSpread': self.get_asset_value(symbol, data, "5mSpread"),
            '30mSpread': self.get_asset_value(symbol, data, "30mSpread"),
            '1hSpread': self.get_asset_value(symbol, data, "1hSpread"),
            '4hSpread': self.get_asset_value(symbol, data, "4hSpread"),
            'MA Trend': self.get_asset_value(symbol, data, "MA Trend"),
            'HMA Trend': self.get_asset_value(symbol, data, "HMA Trend"),
            'MFI': self.get_asset_value(symbol, data, "MFI"),
            'ERI Trend': self.get_asset_value(symbol, data, "ERI Trend"),
            'Funding': self.get_asset_value(symbol, funding_data, "Funding"),  # Use funding_data instead of data
            'Symbols': symbols,
            'Top Signal 5m': self.get_asset_value(symbol, data, "Top Signal 5m"),
            'Bottom Signal 5m': self.get_asset_value(symbol, data, "Bottom Signal 5m"),
            'Top Signal 1m': self.get_asset_value(symbol, data, "Top Signal 1m"),
            'Bottom Signal 1m': self.get_asset_value(symbol, data, "Bottom Signal 1m"),
            'EMA Trend': self.get_asset_value(symbol, data, "EMA Trend")
        }
        return api_data
