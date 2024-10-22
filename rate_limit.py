import time
import threading
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import logging

class RateLimit:
    def __init__(self, calls, period):
        self.calls = calls
        self.period = period
        self.lock = threading.Lock()
        self.call_times = deque()

    def __enter__(self):
        with self.lock:
            now = time.time()
            while self.call_times and now - self.call_times[0] > self.period:
                self.call_times.popleft()
            if len(self.call_times) >= self.calls:
                time_to_wait = self.period - (now - self.call_times[0])
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
            self.call_times.append(now)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

class RateLimitAsync:
    def __init__(self, calls, period):
        self.calls = calls
        self.period = period
        self.lock = asyncio.Lock()
        self.call_times = deque()

    async def __aenter__(self):
        async with self.lock:
            now = time.time()
            while self.call_times and now - self.call_times[0] > self.period:
                self.call_times.popleft()
            if len(self.call_times) >= self.calls:
                time_to_wait = self.period - (now - self.call_times[0])
                if time_to_wait > 0:
                    logging.info(f"Waiting {time_to_wait} in general rate limiter")
                    await asyncio.sleep(time_to_wait)
            self.call_times.append(now)

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass
