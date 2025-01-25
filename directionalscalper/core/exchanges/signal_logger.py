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

class LogFilter(logging.Filter):
    """Filter that injects instance_id and symbol into all log records."""
    def __init__(self, instance_id, symbol=None):
        super().__init__()
        self.instance_id = instance_id
        self.symbol = symbol

    def filter(self, record):
        record.instance_id = self.instance_id
        record.symbol = self.symbol or "GLOBAL"
        return True

class SignalLogger:
    """A logger class for handling trading signals with per-symbol logging and history tracking."""

    def __init__(self, base_dir: str = "logs/signals"):
        """Initialize the SignalLogger.

        Args:
            base_dir (str): Base directory for storing logs and signal history.
        """
        try:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.instance_id = str(uuid.uuid4())[:8]  # Short UUID for readability
            self.loggers: Dict[str, LoggerType] = {}
            self.signal_history: Dict[str, List[Dict[str, Any]]] = {}
            self.log_filter = LogFilter(self.instance_id)  # Global filter without symbol

            # Set up error logging
            self.error_logger = self._setup_error_logger()
        except Exception as e:
            print(f"Failed to initialize SignalLogger: {e}")
            print(traceback.format_exc())
            raise

    def _setup_error_logger(self) -> LoggerType:
        """Set up a separate logger for internal errors.

        Returns:
            LoggerType: Logger for internal errors.
        """
        error_logger = logging.getLogger(f"signal_logger_errors_{self.instance_id}")
        if error_logger.handlers:
            error_logger.handlers.clear()

        error_log_path = self.base_dir / "signal_logger_errors.log"
        handler = handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        formatter = logging.Formatter(
            '%(asctime)s [%(symbol)s] [%(levelname)s] [Instance: %(instance_id)s] %(message)s'
        )
        handler.setFormatter(formatter)
        error_logger.addHandler(handler)
        error_logger.setLevel(logging.ERROR)
        error_logger.addFilter(self.log_filter)
        return error_logger

    def get_logger(self, symbol: str) -> LoggerType:
        """Get or create a logger for a specific symbol.

        Args:
            symbol (str): Trading symbol to get logger for.

        Returns:
            LoggerType: Symbol-specific logger instance.
        """
        try:
            # if symbol not in self.loggers:
            instance_id = str(uuid.uuid4())[:8]  # Short UUID for readability

            logger = logging.getLogger(f"signal_generator_{symbol}_{instance_id}")
            if logger.handlers:
                logger.handlers.clear()

            # Create formatter with symbol
            file_formatter = logging.Formatter(
                fmt='%(asctime)s [%(symbol)s] [%(levelname)s] [Instance: %(instance_id)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # File handler for symbol-specific logs
            symbol_log_path = self.base_dir / f"{symbol}.log"
            file_handler = handlers.RotatingFileHandler(
                symbol_log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(file_formatter)

            # Configure logger with symbol-specific filter
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.propagate = False
            logger.addFilter(LogFilter(instance_id, symbol))

            # self.loggers[symbol] = logger

            return logger

            # return self.loggers[symbol]
        except Exception as e:
            self.error_logger.error(f"Failed to create logger for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())
            raise

    def log_signal(self, symbol: str, signal_data: Dict[str, Any], logger: LoggerType, level: str = "INFO") -> None:
        """Log signal data with instance tracking and store in history.

        Args:
            symbol (str): Trading symbol the signal is for.
            signal_data (Dict[str, Any]): Signal data to log.
            level (str): Log level (default: "INFO").
        """
        try:
            # Add timestamp and instance ID to signal data
            signal_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                **signal_data
            }

            # Store in history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(signal_entry)

            # Keep only last 1000 signals per symbol
            if len(self.signal_history[symbol]) > 1000:
                self.signal_history[symbol] = self.signal_history[symbol][-1000:]

            # Log the signal
            log_method = getattr(logger, level.lower(), logger.info)  # Default to info if invalid level
            log_method(f"Signal: {json.dumps(signal_entry, default=str)}")

            # Save signal history to file periodically
            self._save_signal_history(symbol)
        except Exception as e:
            self.error_logger.error(f"Failed to log signal for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())
            raise

    def _save_signal_history(self, symbol: str) -> None:
        """Save signal history to a JSON file.

        Args:
            symbol (str): Trading symbol to save history for.
        """
        history_file = self.base_dir / f"{symbol}_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.signal_history[symbol], f, default=str, indent=2)
        except Exception as e:
            self.error_logger.error(f"Failed to save signal history for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())

    def get_signal_history(self, symbol: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get signal history for a symbol with optional limit.

        Args:
            symbol (str): Trading symbol to get history for.
            limit (Optional[int]): Maximum number of signals to return.

        Returns:
            List[Dict[str, Any]]: List of signal history entries.
        """
        try:
            history = self.signal_history.get(symbol, [])
            if limit:
                return history[-limit:]
            return history
        except Exception as e:
            self.error_logger.error(f"Failed to get signal history for {symbol}: {e}")
            self.error_logger.error(traceback.format_exc())
            return []

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear signal history for a specific symbol or all symbols.

        Args:
            symbol (Optional[str]): Symbol to clear history for. If None, clears all history.
        """
        try:
            if symbol:
                if symbol in self.signal_history:
                    self.signal_history[symbol] = []
                    history_file = self.base_dir / f"{symbol}_history.json"
                    if history_file.exists():
                        history_file.unlink()
            else:
                self.signal_history.clear()
                for file in self.base_dir.glob("*_history.json"):
                    file.unlink()
        except Exception as e:
            self.error_logger.error(f"Failed to clear history for {symbol or 'all symbols'}: {e}")
            self.error_logger.error(traceback.format_exc())

    def cleanup(self):
        """Cleanup resources by closing all handlers."""
        try:
            # Close error logger handlers
            for handler in self.error_logger.handlers:
                handler.close()
                self.error_logger.removeHandler(handler)

            # Close all symbol-specific logger handlers
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)
            self.loggers.clear()

        except Exception as e:
            print(f"Error during cleanup: {e}")
            print(traceback.format_exc())

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self.cleanup()
