import asyncio
import random
import ta as ta
import logging
import ccxt

from api.manager_async import ManagerAsync
from directionalscalper.core.strategies.base_strategy import BaseStrategy
from .logger import Logger


from rate_limit import RateLimitAsync


logging = Logger(logger_name="BaseStrategyAsync", filename="BaseStrategyAsync.log", stream=True)

class BaseStrategyAsync(BaseStrategy):
    def __init__(self, exchange, config, manager: 'ManagerAsync', symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.rate_limiter_async = RateLimitAsync(10, 1)

    async def is_funding_rate_acceptable_async(self, symbol: str) -> bool:
        """
        Check if the funding rate for a symbol is within the acceptable bounds defined by the MaxAbsFundingRate.

        :param symbol: The symbol for which the check is being made.
        :return: True if the funding rate is within acceptable bounds, False otherwise.
        """
        MaxAbsFundingRate = self.config.MaxAbsFundingRate

        logging.info(f"Max Abs Funding Rate: {self.config.MaxAbsFundingRate}")

        api_data = await self.manager.get_api_data_async(symbol)
        funding_rate = api_data['Funding']

        logging.info(f"Funding rate for {symbol} : {funding_rate}")

        # Check if funding rate is None
        if funding_rate is None:
            logging.warning(f"Funding rate for {symbol} is None.")
            return False

        # Check for longs and shorts combined
        return -MaxAbsFundingRate <= funding_rate <= MaxAbsFundingRate

    async def retry_api_call_async(self, function, *args, max_retries=100, base_delay=10, max_delay=60, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                async with self.rate_limiter_async:
                    return await function(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            except Exception as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Error occurred: {e}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        raise Exception(f"Failed to execute the API function after {max_retries} retries.")
