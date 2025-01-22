from typing import Dict, Tuple, Union, Mapping, Any, TYPE_CHECKING

from ..strategies.logger import Logger

if TYPE_CHECKING:
    from directionalscalper.core.exchanges.exchange import Exchange

import logging
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
import pandas as pd
import numpy as np
import json
import traceback

logging = Logger(logger_name="SignalGenerator", filename="SignalGenerator.log", stream=True)

class SignalGenerator:
    def __init__(self, exchange: "Exchange"):
        self.exchange = exchange

    def generate_l_signals_from_data(self, ohlcv_1m, ohlcv_3m, symbol, neighbors_count=8, use_adx_filter=False, adx_threshold=20):
        """Generate trading signals using balanced approach between old and new implementations."""
        try:
            # Convert data to DataFrames if needed
            if isinstance(ohlcv_1m, list):
                df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                df_1m.set_index('timestamp', inplace=True)
            else:
                df_1m = ohlcv_1m.copy()

            if isinstance(ohlcv_3m, list):
                df_3m = pd.DataFrame(ohlcv_3m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'], unit='ms')
                df_3m.set_index('timestamp', inplace=True)
            else:
                df_3m = ohlcv_3m.copy()

            # Clean data
            for df in [df_1m, df_3m]:
                df.dropna(inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df = df[(df['high'] != 0) & (df['low'] != 0) & (df['close'] != 0) & (df['volume'] != 0)]

            # Calculate indicators using Exchange class methods where available
            df_1m = self._calculate_indicators(df_1m)

            # Calculate trend indicators
            df_3m = self.calculate_trend(df_3m)

            # Calculate recent momentum
            recent_momentum = (df_1m['close'].iloc[-1] / df_1m['close'].iloc[-3] - 1)

            # Calculate Price Position (20-period high/low range)
            highest_high = df_1m['high'].rolling(20).max()
            lowest_low = df_1m['low'].rolling(20).min()
            price_position = (df_1m['close'] - lowest_low) / (highest_high - lowest_low)

            # Calculate Candle Progress
            current_time = int(time.time() * 1000)
            last_candle_time = int(df_1m.index[-1].timestamp() * 1000)
            time_diff = (current_time - last_candle_time) / 1000
            candle_progress = min(max(time_diff / 60.0, 0.0), 1.0)  # For 1-minute candles

            # Prepare features for Lorentzian
            features = df_1m[['rsi', 'adx', 'cci', 'wt']].values[-100:]  # Last 100 candles
            feature_series = features[-1]
            feature_arrays = features[:-1]

            # Calculate Lorentzian distances and predictions
            y_train_series = np.where(df_1m['close'].shift(-4)[-100:] > df_1m['close'][-100:], 1, -1)
            y_train_series = y_train_series[:-1]

            predictions = []
            distances = []
            lastDistance = -1

            for i in range(len(feature_arrays)):
                if i % 4 == 0:
                    d = np.log(1 + np.abs(feature_series - feature_arrays[i])).sum()
                    if d >= lastDistance:
                        lastDistance = d
                        distances.append(d)
                        predictions.append(y_train_series[i])
                        if len(predictions) > neighbors_count:
                            lastDistance = distances[int(neighbors_count * 3 / 4)]
                            distances.pop(0)
                            predictions.pop(0)

            lorentzian_prediction = np.sum(predictions)

            # Detect market regime
            market_regime = self.detect_market_regime(df_1m, symbol)

            # Prepare signal data with all components
            signal_data = {
                "Close": float(df_1m['close'].iloc[-1]),
                "MACD": float(df_3m['macd'].iloc[-1]),
                "MACD_Signal": float(df_3m['macd_signal'].iloc[-1]),
                "EMA_Fast": float(df_3m['ema_fast'].iloc[-1]),
                "EMA_Slow": float(df_3m['ema_slow'].iloc[-1]),
                "RSI": float(df_1m['rsi'].iloc[-1]),
                "Volume": float(df_1m['volume'].iloc[-1]),
                "Avg_Volume": float(df_1m['volume'].rolling(20).mean().iloc[-1]),
                "Recent_Momentum": float(recent_momentum),
                "Price_Position": float(price_position.iloc[-1]),
                "Candle_Progress": float(candle_progress),
                "Lorentzian_Prediction": float(lorentzian_prediction),
                "Market_Regime": market_regime
            }

            # Calculate weighted signal
            weighted_signal = self.calculate_weighted_signal(signal_data, symbol)

            # Generate final signal with multi-timeframe confirmation
            new_signal = 'neutral'

            # Get trend confirmations
            is_uptrend_1m = df_1m['close'].iloc[-1] > df_1m['close'].rolling(8).mean().iloc[-1]
            is_uptrend_3m = df_3m['close'].iloc[-1] > df_3m['ema_fast'].iloc[-1] > df_3m['ema_slow'].iloc[-1]

            is_downtrend_1m = df_1m['close'].iloc[-1] < df_1m['close'].rolling(8).mean().iloc[-1]
            is_downtrend_3m = df_3m['close'].iloc[-1] < df_3m['ema_fast'].iloc[-1] < df_3m['ema_slow'].iloc[-1]

            # Volume confirmation
            volume_confirmed = float(signal_data["Volume"]) > float(signal_data["Avg_Volume"]) * 1.1

            # MACD confirmation
            macd_confirmed = (float(signal_data["MACD"]) > float(signal_data["MACD_Signal"]) and
                             float(signal_data["MACD"]) > 0) or \
                            (float(signal_data["MACD"]) < float(signal_data["MACD_Signal"]) and
                             float(signal_data["MACD"]) < 0)

            # RSI filters
            rsi = float(signal_data["RSI"])
            rsi_allows_long = rsi < 70  # Not overbought
            rsi_allows_short = rsi > 30  # Not oversold

            # Market regime based thresholds - align with log interpretation
            regime = str(signal_data["Market_Regime"])
            if regime == "trending":
                signal_threshold = 0.2  # Moderate threshold in trending markets
            elif regime == "volatile":
                signal_threshold = 0.3  # Strong threshold in volatile markets
            else:
                signal_threshold = 0.25  # Default threshold

            # Signal generation with multiple confirmation levels
            if weighted_signal > signal_threshold:
                # Strong signal with full confirmation
                if is_uptrend_3m and is_uptrend_1m and volume_confirmed and macd_confirmed and rsi_allows_long:
                    new_signal = 'long'
                # Good signal with partial confirmation
                elif is_uptrend_3m and (volume_confirmed or macd_confirmed) and rsi_allows_long:
                    if weighted_signal > signal_threshold * 1.2:  # 20% stronger signal required
                        new_signal = 'long'
                # Momentum signal in strong trend
                elif regime == "trending" and is_uptrend_3m and weighted_signal > signal_threshold * 1.3:
                    new_signal = 'long'

            elif weighted_signal < -signal_threshold:
                # Strong signal with full confirmation
                if is_downtrend_3m and is_downtrend_1m and volume_confirmed and macd_confirmed and rsi_allows_short:
                    new_signal = 'short'
                # Good signal with partial confirmation
                elif is_downtrend_3m and (volume_confirmed or macd_confirmed) and rsi_allows_short:
                    if weighted_signal < -signal_threshold * 1.2:  # 20% stronger signal required
                        new_signal = 'short'
                # Momentum signal in strong trend
                elif regime == "trending" and is_downtrend_3m and weighted_signal < -signal_threshold * 1.3:
                    new_signal = 'short'

            # Apply ADX filter if enabled
            if use_adx_filter and df_1m['adx'].iloc[-1] < adx_threshold:
                new_signal = 'neutral'

            # Signal buffer logic with proper buffering
            current_time = time.time()
            if symbol in self.exchange.last_signal:
                last_signal = self.exchange.last_signal[symbol]
                time_since_last = current_time - self.exchange.last_signal_time[symbol]

                # Determine buffer time based on regime
                buffer_time = 15  # Base buffer time
                if regime == "volatile":
                    buffer_time = 10  # Shorter buffer in volatile markets
                elif regime == "trending":
                    buffer_time = 20  # Longer buffer in trending markets

                if new_signal != 'neutral':  # Only buffer non-neutral signals
                    if new_signal == last_signal:  # Same signal
                        if time_since_last < buffer_time:
                            logging.info(f"[{symbol}] Signal buffered (regime: {regime}, buffer: {buffer_time}s)")
                            # Clean up memory before return
                            del df_1m, df_3m
                            return last_signal  # Return last signal during buffer period
                        else:
                            # Update time but keep the signal
                            self.exchange.last_signal_time[symbol] = current_time
                            del df_1m, df_3m
                            return new_signal
                    else:  # Different signal
                        # Update both signal and time
                        self.exchange.last_signal[symbol] = new_signal
                        self.exchange.last_signal_time[symbol] = current_time
                        del df_1m, df_3m
                        return new_signal
                else:  # Neutral signals aren't buffered
                    self.exchange.last_signal[symbol] = new_signal
                    self.exchange.last_signal_time[symbol] = current_time
                    del df_1m, df_3m
                    return new_signal
            else:  # First signal for this symbol
                self.exchange.last_signal[symbol] = new_signal
                self.exchange.last_signal_time[symbol] = current_time
                del df_1m, df_3m
                return new_signal

        except Exception as e:
            logging.error(f"[{symbol}] Error in generate_l_signals_from_data: {e}")
            logging.error(f"[{symbol}] {traceback.format_exc()}")
            # Clean up memory even on error
            try:
                del df_1m, df_3m
            except:
                pass
            return "neutral"

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators."""
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['adx'] = self.exchange.n_adx(df['high'], df['low'], df['close'], 14)
        df['cci'] = self.exchange.n_cci(df['high'], df['low'], df['close'], 20, 1)
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        df['wt'] = self.exchange.n_wt(hlc3, 10, 11)
        # atr_indicator = AverageTrueRange(
        #     high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True
        # )
        # df['atr'] = atr_indicator.average_true_range()
        # df['atr_pct'] = (df['atr'] / df['close']) * 100
        return df

    def calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA and MACD for trend detection."""
        df['ema_fast'] = EMAIndicator(df['close'], window=50, fillna=True).ema_indicator()
        df['ema_medium'] = EMAIndicator(df['close'], window=100, fillna=True).ema_indicator()
        df['ema_slow'] = EMAIndicator(df['close'], window=200, fillna=True).ema_indicator()
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        return df

    def detect_trend(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Detect if a strong uptrend or downtrend exists."""
        close, ema_fast, ema_medium, ema_slow = (
            df['close'].iloc[-1],
            df['ema_fast'].iloc[-1],
            df['ema_medium'].iloc[-1],
            df['ema_slow'].iloc[-1],
        )
        macd, macd_signal = df['macd'].iloc[-1], df['macd_signal'].iloc[-1]
        is_uptrend = close > ema_fast > ema_medium > ema_slow and macd > macd_signal
        is_downtrend = close < ema_fast < ema_medium < ema_slow and macd < macd_signal
        return is_uptrend, is_downtrend

    def detect_market_regime(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Enhanced market regime detection optimized for DCA crypto trading.
        Combines volatility analysis with trend consistency and direction changes.
        Returns 'volatile', 'trending', or 'ranging'
        """
        try:
            if len(df) < 20:  # Minimum required data
                return 'ranging'

            # 1. Volatility Analysis
            # Calculate ATR percentage for better volatility measurement
            atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = atr_indicator.average_true_range()
            df['atr_pct'] = (df['atr'] / df['close']) * 100

            current_volatility = df['atr_pct'].iloc[-1]
            recent_volatility = df['atr_pct'].rolling(5).mean().iloc[-1]  # 5-minute average
            baseline_volatility = df['atr_pct'].rolling(20).mean().iloc[-1]  # 20-minute baseline

            # 2. Price Changes and Direction Analysis
            price_changes = df['close'].pct_change()
            # Count direction changes in last 10 periods
            direction_changes = (np.diff(np.signbit(price_changes.iloc[-10:])) != 0).sum()

            # 3. Momentum Analysis
            mom_1min = price_changes.iloc[-1] * 100
            mom_3min = df['close'].pct_change(3).iloc[-1] * 100
            mom_5min = df['close'].pct_change(5).iloc[-1] * 100
            recent_momentum = (mom_1min * 0.5 + mom_3min * 0.3 + mom_5min * 0.2)

            # 4. Trend Analysis
            ema_fast = df['close'].ewm(span=8, adjust=False).mean()
            ema_med = df['close'].ewm(span=21, adjust=False).mean()
            ema_slow = df['close'].ewm(span=34, adjust=False).mean()

            # Trend consistency checks (from old implementation)
            price_above_fast = (df['close'] > ema_fast).rolling(10).mean().iloc[-1]
            fast_above_med = (ema_fast > ema_med).rolling(10).mean().iloc[-1]
            med_above_slow = (ema_med > ema_slow).rolling(10).mean().iloc[-1]
            trend_alignment = (price_above_fast + fast_above_med + med_above_slow) / 3

            # 5. Range Analysis
            high_low_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
            price_position = (df['close'] - df['low'].rolling(20).min()) / high_low_range
            range_touches = ((price_position < 0.2) | (price_position > 0.8)).rolling(20).sum().iloc[-1]

            # 6. Volume Analysis
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
            volume_trend = (df['volume'] > volume_sma).rolling(5).mean().iloc[-1]

            # Log regime metrics
            logging.info(f"""[{symbol}] Market Regime Metrics:
                Volatility:
                  Current ATR%: {current_volatility:.3f}%
                  Recent (5m): {recent_volatility:.3f}%
                  Baseline (20m): {baseline_volatility:.3f}%
                  Relative: {(current_volatility/baseline_volatility):.2f}x baseline
                Momentum:
                  1min: {mom_1min:.3f}%
                  3min: {mom_3min:.3f}%
                  5min: {mom_5min:.3f}%
                  Combined: {recent_momentum:.3f}
                Trend:
                  Price/Fast: {price_above_fast:.3f}
                  Fast/Med: {fast_above_med:.3f}
                  Med/Slow: {med_above_slow:.3f}
                  Alignment: {trend_alignment:.3f}
                Price Action:
                  Direction Changes: {direction_changes}
                  Range Touches: {range_touches}
                Volume:
                  Ratio: {volume_ratio:.3f}
                  Trend: {volume_trend:.3f}
            """)

            # Enhanced Regime Detection Logic
            # 1. Volatile Market (multiple conditions from both implementations)
            if (
                # High relative volatility with volume confirmation
                (current_volatility > baseline_volatility * 1.4 and volume_ratio > 1.2) or
                # Strong momentum with volume
                (abs(recent_momentum) > 0.8 and volume_ratio > 1.3) or
                # Sudden volatility spike
                (current_volatility > recent_volatility * 1.5) or
                # Many direction changes with high volume
                (direction_changes >= 5 and volume_ratio > 1.4) or
                # Absolute volatility threshold (from old implementation)
                (current_volatility > 2.0 and volume_trend > 0.7)
            ):
                return 'volatile'

            # 2. Trending Market (combined conditions)
            elif (
                # Strong trend alignment with momentum
                (trend_alignment > 0.7 and abs(recent_momentum) > 0.3) or
                # Very strong trend alignment
                (trend_alignment > 0.8 and volume_ratio > 1.1) or
                # Consistent price action with momentum
                (abs(mom_5min) > 0.5 and trend_alignment > 0.6) or
                # Clear trend with few direction changes
                (trend_alignment > 0.65 and direction_changes <= 2 and volume_trend > 0.6)
            ):
                return 'trending'

            # 3. Ranging Market (enhanced conditions)
            elif (
                # Multiple range touches with normal volatility
                (range_touches > 4 and current_volatility < baseline_volatility) or
                # Weak trend with low volatility
                (trend_alignment < 0.4 and current_volatility < recent_volatility) or
                # Low momentum with normal volume
                (abs(recent_momentum) < 0.2 and volume_ratio < 0.9) or
                # Many direction changes with low volatility
                (direction_changes >= 4 and current_volatility < baseline_volatility * 0.8)
            ):
                return 'ranging'

            # Default case
            return 'ranging'  # Default to ranging for safer DCA

        except Exception as e:
            logging.error(f"[{symbol}] Error in detect_market_regime: {e}")
            logging.error(f"[{symbol}] {traceback.format_exc()}")
            return 'ranging'  # Default to ranging on error

    def _get_regime_adjusted_weights(self, market_regime: str, symbol: str) -> Dict[str, float]:
        """Get weights adjusted for market regime, optimized for DCA crypto trading."""
        # Base weights optimized for DCA after removing volume
        base_weights = {
            "Trend": 0.40,          # Increased from 0.35 (primary trend)
            "Momentum": 0.30,       # Increased from 0.25 (MACD + Recent momentum)
            "Position": 0.15,       # Decreased from 0.20 (less focus on range)
            "RSI": 0.15            # Increased from 0.10 (better reversal detection)
        }

        # Optional Lorentzian component (0 to disable)
        lorentzian_weight = 0.10   # Unchanged ML influence

        if lorentzian_weight > 0:
            # Reduce other weights proportionally to add Lorentzian
            total_reduction = lorentzian_weight
            for k in base_weights:
                base_weights[k] *= (1 - total_reduction)
            base_weights["Lorentzian"] = lorentzian_weight

        weights = base_weights.copy()

        # Apply regime-specific adjustments
        if market_regime == "volatile":
            weights.update({
                "Trend": weights["Trend"] * 0.8,        # Reduce trend reliance in volatility
                "Momentum": weights["Momentum"] * 1.4,   # Increase momentum importance
                "Position": weights["Position"] * 1.2,   # Slightly more range focus
                "RSI": weights["RSI"] * 1.2             # More focus on extremes
            })
            if "Lorentzian" in weights:
                weights["Lorentzian"] *= 0.7            # Reduce ML in volatility

        elif market_regime == "trending":
            weights.update({
                "Trend": weights["Trend"] * 1.4,        # Strong trend following
                "Momentum": weights["Momentum"] * 1.2,   # Support momentum
                "Position": weights["Position"] * 0.6,   # Less focus on ranges
                "RSI": weights["RSI"] * 0.8             # Less reversal focus
            })
            if "Lorentzian" in weights:
                weights["Lorentzian"] *= 1.2            # Increase pattern recognition

        elif market_regime == "ranging":
            weights.update({
                "Trend": weights["Trend"] * 0.6,        # Reduce trend importance
                "Momentum": weights["Momentum"] * 0.8,   # Less momentum focus
                "Position": weights["Position"] * 1.8,   # Strong range focus
                "RSI": weights["RSI"] * 1.4             # Strong mean reversion
            })
            if "Lorentzian" in weights:
                weights["Lorentzian"] *= 1.0            # Normal pattern recognition

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Log the weights for debugging
        logging.debug(f"[{symbol}] Market Regime {market_regime} - Adjusted Weights:")
        for component, weight in normalized_weights.items():
            logging.debug(f"[{symbol}]   {component:10}: {weight:.3f}")

        return normalized_weights

    def _log_signal_components(self, symbol: str, components: Mapping[str, Union[float, Dict[str, float]]], weighted_signal: float, final_signal: float) -> None:
        """Enhanced logging of signal components and final results."""
        try:
            logging.info(f"[{symbol}] {'='*20} Signal Analysis {'='*20}")

            # Log raw values first
            raw_values = components.get("Raw_Values", {})
            if isinstance(raw_values, dict):  # Ensure raw_values is a dictionary
                logging.info(f"[{symbol}] Raw Values:")
                for name, value in raw_values.items():
                    if isinstance(value, (int, float)):  # Ensure value is numeric
                        logging.info(f"[{symbol}] {name:15}: {value:>8.4f}")
                components_without_raw = {k: v for k, v in components.items() if k != "Raw_Values"}
            else:
                components_without_raw = components

            # Log weighted components
            logging.info(f"[{symbol}] Weighted Components:")
            for component, value in components_without_raw.items():
                if isinstance(value, (int, float)):  # Ensure value is numeric
                    arrow = "↑" if value > 0 else "↓" if value < 0 else "→"
                    strength = abs(value) / 0.5 * 100  # Normalize to percentage
                    logging.info(f"[{symbol}] {component:15}: {value:>8.4f} {arrow} {strength:>5.1f}%")

            # Log signal progression
            logging.info(f"[{symbol}] Signal Progression:")
            logging.info(f"[{symbol}] {'Raw Weighted':15}: {weighted_signal:>8.4f}")
            logging.info(f"[{symbol}] {'Final Signal':15}: {final_signal:>8.4f}")

            # Log signal interpretation using consistent thresholds
            signal_strength = abs(final_signal)
            if signal_strength > 0.3:
                strength_desc = "Strong"
            elif signal_strength > 0.2:
                strength_desc = "Moderate"
            elif signal_strength > 0.1:
                strength_desc = "Weak"
            else:
                strength_desc = "Neutral"

            direction = "LONG" if final_signal > 0 else "SHORT" if final_signal < 0 else "NEUTRAL"
            if direction != "NEUTRAL":
                logging.info(f"[{symbol}] Signal Interpretation: {strength_desc} {direction}")
            else:
                logging.info(f"[{symbol}] Signal Interpretation: NEUTRAL")
            logging.info(f"[{symbol}] {'='*50}")

        except Exception as e:
            logging.error(f"[{symbol}] Error in _log_signal_components: {e}")
            logging.error(f"[{symbol}] {traceback.format_exc()}")

    def calculate_weighted_signal(self, data: Dict[str, Any], symbol: str) -> float:
        """Calculate weighted signal optimized for DCA crypto trading."""
        try:
            # Initialize components dictionary with explicit typing
            signal_components: Dict[str, float] = {}
            raw_values: Dict[str, float] = {}

            # 1. Trend Component (Simplified and enhanced with best features from both implementations)
            close = float(data["Close"])
            open_price = float(data.get("Open", close))
            ema_fast = float(data["EMA_Fast"])
            ema_slow = float(data["EMA_Slow"])
            ema_medium = float(data.get("EMA_Medium", (ema_fast + ema_slow) / 2))

            # Simple trend detection from old implementation
            is_uptrend = close > ema_fast > ema_medium > ema_slow
            is_downtrend = close < ema_fast < ema_medium < ema_slow

            # Enhanced trend strength calculation
            trend_strength = (ema_fast - ema_slow) / ema_slow
            price_to_ema = (close - ema_fast) / ema_fast

            # Simplified trend score with proven components from old implementation
            trend_score = (trend_strength * 0.5 + price_to_ema * 0.5)  # Equal weighting
            if is_uptrend:
                trend_score = max(trend_score * 1.2, 0.1)  # Minimum positive score in uptrend
            elif is_downtrend:
                trend_score = min(trend_score * 1.2, -0.1)  # Minimum negative score in downtrend

            signal_components["Trend"] = np.clip(trend_score * 100, -0.5, 0.5)

            # 2. Momentum Component (Enhanced with MACD confirmation)
            macd_diff = float(data["MACD"] - data["MACD_Signal"])
            macd = float(data["MACD"])
            macd_signal = float(data["MACD_Signal"])
            macd_norm = np.clip(macd_diff / (close * 0.001), -1, 1)
            recent_momentum = float(data.get("Recent_Momentum", 0))

            # MACD confirmation
            macd_confirmed = (macd > macd_signal and macd_diff > 0) or (macd < macd_signal and macd_diff < 0)

            momentum_score = (macd_norm * 0.7 + recent_momentum * 100 * 0.3)
            if macd_confirmed:
                momentum_score *= 1.2

            signal_components["Momentum"] = np.clip(momentum_score * 0.5, -0.5, 0.5)

            # 3. Position Component (Enhanced with dynamic zones)
            pos = float(data.get("Price_Position", 0.5))
            atr_pct = float(data.get("ATR_Pct", 1.0))  # ATR as percentage of price

            # Dynamic thresholds based on volatility
            upper_threshold = min(0.8, 0.75 + atr_pct * 0.1)
            lower_threshold = max(0.2, 0.25 - atr_pct * 0.1)

            if pos > upper_threshold:
                pos_score = -0.4
            elif pos > upper_threshold - 0.1:
                pos_score = -0.2
            elif pos < lower_threshold:
                pos_score = 0.4
            elif pos < lower_threshold + 0.1:
                pos_score = 0.2
            else:
                pos_score = (0.5 - pos) * 0.4
            signal_components["Position"] = pos_score

            # 4. RSI Component (Enhanced with trend alignment)
            rsi = float(data["RSI"])
            if rsi > 70:
                rsi_score = -0.4 * ((rsi - 70) / 30)
                if is_uptrend:  # Less bearish in strong uptrend
                    rsi_score *= 0.7
            elif rsi < 30:
                rsi_score = 0.4 * ((30 - rsi) / 30)
                if is_downtrend:  # Less bullish in strong downtrend
                    rsi_score *= 0.7
            else:
                rsi_score = (50 - rsi) / 50 * 0.2
            signal_components["RSI"] = rsi_score

            # 5. Volume Component (Minimized for DCA grid trading)
            volume = float(data["Volume"])
            avg_volume = float(data["Avg_Volume"])
            candle_progress = float(data.get("Candle_Progress", 1.0))
            volume_ratio = 0.0  # Initialize volume_ratio

            # Volume scoring with minimal impact for DCA
            if avg_volume > 0:
                # Calculate volume ratio relative to expected volume for current candle progress
                expected_volume = avg_volume * candle_progress
                volume_ratio = volume / expected_volume if expected_volume > 0 else 0.0

                # Only consider extreme volume conditions
                if volume_ratio > 2.0:  # Extremely high volume (2x expected)
                    vol_score = 0.2  # Small positive score
                elif volume_ratio < 0.2:  # Extremely low volume (less than 20% expected)
                    vol_score = -0.1  # Very small negative score
                else:
                    vol_score = 0.0  # Neutral for normal volume

                # Minimal weight for volume component
                volume_weight_multiplier = min(0.2, candle_progress)  # Maximum 20% weight
            else:
                vol_score = 0.0
                volume_weight_multiplier = 0.1

            signal_components["Volume"] = vol_score
            signal_components["Volume_Weight_Mult"] = volume_weight_multiplier

            # 6. Lorentzian Component (Enhanced with trend alignment)
            if "Lorentzian_Prediction" in data:
                lorentzian_pred = float(data["Lorentzian_Prediction"])
                neighbors_count = float(data.get("Neighbors_Count", 8))
                max_pred = neighbors_count * 0.8
                lorentzian_norm = np.clip(lorentzian_pred / max_pred, -0.4, 0.4)

                # Enhance Lorentzian influence when aligned with trend
                if (lorentzian_norm > 0 and trend_score > 0) or (lorentzian_norm < 0 and trend_score < 0):
                    lorentzian_norm *= 1.2

                signal_components["Lorentzian"] = lorentzian_norm

            # Get regime-adjusted weights
            regime = str(data["Market_Regime"])
            weights = self._get_regime_adjusted_weights(regime, symbol)

            # Adjust volume weight based on candle progress
            # if "Volume" in weights:
            #     weights["Volume"] *= signal_components.pop("Volume_Weight_Mult")

            # Calculate weighted signal
            weighted_signal = sum(weights[k] * signal_components[k] for k in weights.keys() if k in signal_components)

            # Apply regime-based scaling with volume confirmation
            if regime == "volatile":
                if abs(signal_components["Volume"]) > 0.15:  # Only in extreme volume conditions
                    final_signal = np.clip(weighted_signal * 1.1, -1, 1)  # Slightly stronger signal
                else:
                    final_signal = weighted_signal  # No modification
            else:
                final_signal = weighted_signal

            # Strong signal enhancement when multiple components align
            component_signs = [np.sign(v) for v in signal_components.values() if abs(v) > 0.1]
            if len(component_signs) >= 4 and all(s == np.sign(final_signal) for s in component_signs):
                # Additional check for trend confirmation and candle progress
                if (final_signal > 0 and is_uptrend) or (final_signal < 0 and is_downtrend):
                    # Scale enhancement based on candle progress
                    enhancement = 1.0 + (0.3 * min(candle_progress, 1.0))  # Max 30% boost, scales with progress
                    final_signal *= enhancement
                else:
                    # Scale enhancement based on candle progress
                    enhancement = 1.0 + (0.1 * min(candle_progress, 1.0))  # Max 10% boost, scales with progress
                    final_signal *= enhancement
                final_signal = np.clip(final_signal, -1, 1)

            # Store raw values for logging
            raw_values.update({
                "Close": close,
                "Open": open_price,
                "EMA_Fast": ema_fast,
                "EMA_Medium": ema_medium,
                "EMA_Slow": ema_slow,
                "MACD": macd,
                "MACD_Signal": macd_signal,
                "RSI": rsi,
                "Volume_Ratio": volume_ratio,
                "Volume_vs_Avg": (volume / avg_volume) if avg_volume > 0 else 0.0,
                "Candle_Progress": candle_progress,
                "Volume_Weight": weights.get("Volume", 0.0),
                "Is_Strong_Uptrend": is_uptrend,
                "Is_Strong_Downtrend": is_downtrend,
                "MACD_Confirmed": macd_confirmed,
                "Component_Alignment": len([s for s in component_signs if s == np.sign(final_signal)]),
                "Total_Components": len(component_signs),
                "Candle_Direction": 1 if close >= open_price else -1,
                "Candle_Size": abs(close - open_price) / open_price * 100
            })

            # Log components and final signal
            self._log_signal_components(symbol, {**signal_components, "Raw_Values": raw_values}, weighted_signal, final_signal)

            return final_signal

        except Exception as e:
            logging.error(f"[{symbol}] Error in calculate_weighted_signal: {e}")
            logging.error(traceback.format_exc())
            return 0.0

