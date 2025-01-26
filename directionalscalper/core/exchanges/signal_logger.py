import os
import logging
import logging.handlers as handlers
from pathlib import Path
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import traceback
from logging import Logger as LoggerType
from collections import deque

class LogFilter(logging.Filter):
    """Filter that injects instance_id and symbol into all log records."""
    def __init__(self, instance_id: str | None, symbol: str | None):
        super().__init__()
        self.instance_id = instance_id
        self.symbol = symbol

    def filter(self, record):
        record.instance_id = self.instance_id
        record.symbol = self.symbol or "GLOBAL"
        return True

class SignalLogger:
    """A logger class for handling trading signals with per-symbol logging and history tracking."""

    MAX_HISTORY_SIZE = 1000
    LOG_FORMAT = '%(asctime)s [%(symbol)s] [%(levelname)s] [Instance: %(instance_id)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, base_dir: str = "logs"):
        """Initialize the SignalLogger.

        Args:
            base_dir (str): Base directory for storing logs and signal history.
        """
        try:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.loggers: Dict[str, LoggerType] = {}
            self.signal_history = deque(maxlen=self.MAX_HISTORY_SIZE)

            # Create shared file handlers
            self.signals_handler = self._create_handler(
                log_file="signals.log",
                max_bytes=10*1024*1024,
                backup_count=5
            )
            self.errors_handler = self._create_handler(
                log_file="signal_logger_errors.log",
                max_bytes=5*1024*1024,
                backup_count=3
            )

            # Set up error logger with its own instance_id
            self.error_logger = self._setup_logger(
                name="signal_logger_errors",
                handler=self.errors_handler,
                level=logging.ERROR,
                symbol=None,
                instance_id=None
            )
        except Exception as e:
            print(f"Failed to initialize SignalLogger: {e}")
            print(traceback.format_exc())
            raise

    def _create_handler(self, log_file: str, max_bytes: int, backup_count: int) -> handlers.RotatingFileHandler:
        """Create a rotating file handler with standard configuration.

        Args:
            log_file (str): Log file path
            max_bytes (int): Maximum size of log file before rotation
            backup_count (int): Number of backup files to keep

        Returns:
            handlers.RotatingFileHandler: Configured handler
        """
        handler = handlers.RotatingFileHandler(
            self.base_dir / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        formatter = logging.Formatter(self.LOG_FORMAT, self.DATE_FORMAT)
        handler.setFormatter(formatter)
        return handler

    def _setup_logger(self, name: str, handler: handlers.RotatingFileHandler,
                     level: int, symbol: Optional[str], instance_id: Optional[str]) -> LoggerType:
        """Set up a logger with standard configuration.

        Args:
            name (str): Logger name
            handler (handlers.RotatingFileHandler): Rotating file handler
            level (int): Logging level
            symbol (Optional[str]): Trading symbol

        Returns:
            LoggerType: Configured logger
        """
        logger = logging.getLogger(name)
        if logger.handlers:
            logger.handlers.clear()

        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        logger.addFilter(LogFilter(instance_id, symbol))
        return logger

    def get_logger(self, symbol: str) -> LoggerType:
        """Get or create a logger for a specific symbol.

        Args:
            symbol (str): Trading symbol to get logger for.

        Returns:
            LoggerType: Symbol-specific logger instance.
        """
        try:
            # Remove old logger if it exists (but keep the handler)
            if symbol in self.loggers:
                old_logger = self.loggers[symbol]
                old_logger.handlers.clear()  # Just remove handler reference, don't close it
                del self.loggers[symbol]

            # Create new logger with new instance_id, reusing the handler
            instance_id = str(uuid.uuid4())[:8]
            logger = self._setup_logger(
                name=f"signal_generator_{symbol}_{instance_id}",
                handler=self.signals_handler,
                level=logging.INFO,
                symbol=symbol,
                instance_id=instance_id
            )
            self.loggers[symbol] = logger
            return logger
        except Exception as e:
            self.error_logger.error(f"Failed to create logger for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())
            raise

    def log_signal(self, symbol: str, signal_data: Dict[str, Any], logger: LoggerType, level: str = "INFO") -> None:
        """Log signal data with instance tracking and store in history.

        Args:
            symbol (str): Trading symbol the signal is for.
            signal_data (Dict[str, Any]): Signal data to log.
            logger (LoggerType): Logger instance to use.
            level (str): Log level (default: "INFO").
        """
        try:
            signal_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                **signal_data
            }

            self.signal_history.append(signal_entry)
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(f"Signal: {json.dumps(signal_entry, default=str)}")
            self._save_signal_history()
        except Exception as e:
            self.error_logger.error(f"Failed to log signal for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())
            raise

    def _save_signal_history(self) -> None:
        """Save signal history to a JSON file."""
        try:
            history_file = self.base_dir / "signals_history.json"
            with open(history_file, 'w') as f:
                json.dump(list(self.signal_history), f, default=str, indent=2)
        except Exception as e:
            self.error_logger.error(f"Failed to save signal history: {e}")
            self.error_logger.error(traceback.format_exc())

    def cleanup(self) -> None:
        """Cleanup resources by closing all handlers."""
        try:
            # Save history before cleanup
            self._save_signal_history()

            # Clear all loggers (without closing shared handlers)
            for logger in self.loggers.values():
                logger.handlers.clear()
            self.loggers.clear()

            # Now close the shared handlers
            if hasattr(self, 'signals_handler'):
                self.signals_handler.close()
            if hasattr(self, 'errors_handler'):
                self.errors_handler.close()

            # Clear error logger handlers
            if hasattr(self, 'error_logger'):
                self.error_logger.handlers.clear()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print(traceback.format_exc())

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self.cleanup()

