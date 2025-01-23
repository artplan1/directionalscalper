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
            # Convert list to DataFrame if needed
            if isinstance(ohlcv_1m, list):
                logging.info(f"[{symbol}] Converting 1m data to DataFrame")

                df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                df_1m.set_index('timestamp', inplace=True)
            else:
                df_1m = ohlcv_1m.copy()

            if isinstance(ohlcv_3m, list):
                logging.info(f"[{symbol}] Converting 3m data to DataFrame")

                df_3m = pd.DataFrame(ohlcv_3m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'], unit='ms')
                df_3m.set_index('timestamp', inplace=True)
            else:
                df_3m = ohlcv_3m.copy()

            logging.info(f"[{symbol}] Data structure after conversion:")
            logging.info(f"1m columns: {df_1m.columns.tolist()}")
            logging.info(f"1m index type: {type(df_1m.index)}")
            logging.info(f"1m data shape: {df_1m.shape}, 3m data shape: {df_3m.shape}")

            # Calculate indicators
            try:
                logging.info(f"[{symbol}] Starting indicator calculations")

                # Calculate indicators
                df_1m = self._calculate_indicators(df_1m)
                df_3m = self.calculate_trend(df_3m)

                try:
                    logging.info(f"[{symbol}] 3m indicators calculated successfully")
                    logging.info(f"[{symbol}] EMA ranges - Fast: {df_3m['ema_fast'].min():.4f} to {df_3m['ema_fast'].max():.4f}")
                    logging.info(f"[{symbol}] EMA ranges - Medium: {df_3m['ema_medium'].min():.4f} to {df_3m['ema_medium'].max():.4f}")
                    logging.info(f"[{symbol}] EMA ranges - Slow: {df_3m['ema_slow'].min():.4f} to {df_3m['ema_slow'].max():.4f}")
                    logging.info(f"[{symbol}] MACD range: {df_3m['macd'].min():.4f} to {df_3m['macd'].max():.4f}")
                except Exception as e:
                    logging.error(f"[{symbol}] 3m indicator calculation failed: {e}")
                    return 'neutral'

                # Verify no NaN values in critical columns
                critical_1m_cols = ['rsi', 'adx', 'cci', 'wt', 'atr', 'atr_pct']
                critical_3m_cols = ['ema_fast', 'ema_medium', 'ema_slow', 'macd', 'macd_signal']

                # Check latest values
                latest_1m = df_1m[critical_1m_cols].iloc[-1]
                latest_3m = df_3m[critical_3m_cols].iloc[-1]

                nan_1m = latest_1m.isna().sum()
                nan_3m = latest_3m.isna().sum()

                if nan_1m > 0 or nan_3m > 0:
                    logging.error(f"[{symbol}] NaN values in latest bar - 1m: {nan_1m}, 3m: {nan_3m}")
                    logging.error(f"[{symbol}] 1m values: {latest_1m.to_dict()}")
                    logging.error(f"[{symbol}] 3m values: {latest_3m.to_dict()}")

                    # Log the last few rows of price data
                    logging.error(f"[{symbol}] Last 5 rows of 1m price data:")
                    logging.error(df_1m[['open', 'high', 'low', 'close']].tail().to_string())
                    return 'neutral'

                logging.info(f"[{symbol}] All indicators calculated successfully")

            except Exception as e:
                logging.error(f"[{symbol}] Error calculating indicators: {e}")
                logging.error(f"[{symbol}] {traceback.format_exc()}")
                return 'neutral'

            # Detect trend
            is_strong_uptrend, is_strong_downtrend = self.detect_trend(df_3m)

            logging.info(f"[{symbol}] 3m Trend Analysis:")
            logging.info(f"[{symbol}] Close: {df_3m['close'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] EMA Fast: {df_3m['ema_fast'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] EMA Medium: {df_3m['ema_medium'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] EMA Slow: {df_3m['ema_slow'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] MACD: {df_3m['macd'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] MACD Signal: {df_3m['macd_signal'].iloc[-1]:.4f}")
            logging.info(f"[{symbol}] Strong Uptrend: {is_strong_uptrend}")
            logging.info(f"[{symbol}] Strong Downtrend: {is_strong_downtrend}")

            # Calculate market regime from 1m data
            market_regime = self._detect_market_regime(df_1m, df_3m, symbol)
            # logging.info(f"[{symbol}] Detecting market regime - atr_pct: {df_1m['atr_pct'].iloc[-1]}")

            logging.info(f"[{symbol}] Market Regime: {market_regime}")
            latest_atr_pct = df_1m['atr_pct'].iloc[-1]
            logging.info(f"[{symbol}] ATR %: {latest_atr_pct:.4f}")

            # Calculate trend quality once
            trend_quality = self._calculate_trend_quality(df_3m, df_1m)

            logging.info(f"[{symbol}] EMA Separations - Fast-Med: {abs(df_3m['ema_fast'].iloc[-1] - df_3m['ema_medium'].iloc[-1]) / df_3m['close'].iloc[-1]:.6f}, Med-Slow: {abs(df_3m['ema_medium'].iloc[-1] - df_3m['ema_slow'].iloc[-1]) / df_3m['close'].iloc[-1]:.6f}")
            logging.info(f"[{symbol}] Trend Quality: {trend_quality:.3f}")

            # Prepare data for weighted signal calculation
            signal_data = {
                "Close": df_3m['close'].iloc[-1],
                'MACD': df_3m['macd'].iloc[-1],
                'MACD_Signal': df_3m['macd_signal'].iloc[-1],
                'EMA_Fast': df_3m['ema_fast'].iloc[-1],
                'EMA_Medium': df_3m['ema_medium'].iloc[-1],
                'EMA_Slow': df_3m['ema_slow'].iloc[-1],
                'ATR': df_1m['atr'].iloc[-1],
                'MACD_Range': df_3m['macd'].abs().max(),  # Add MACD range
                'trend_quality': trend_quality  # Add trend quality to signal data
            }

            # Calculate prediction if using nearest neighbor analysis
            if neighbors_count > 0:
                features = df_1m[['rsi', 'adx', 'cci', 'wt']].values
                feature_series = features[-1]
                feature_arrays = features[:-1]  # All except last point

                # Calculate Lorentzian distances and predictions
                y_train_series = np.where(df_1m['close'].shift(-4) > df_1m['close'], 1, -1)[:-1]

                predictions = []
                distances = []
                lastDistance = -1

                # Improved sampling and distance calculation
                for i in range(len(feature_arrays)):
                    if i % 4 == 0:  # Sample every 4th point for noise reduction
                        # Lorentzian distance with feature scaling
                        diff = feature_series - feature_arrays[i]
                        scaled_diff = diff / (np.abs(diff).mean() + 1e-10)  # Prevent division by zero
                        d = np.log(1 + np.abs(scaled_diff)).sum()

                        if d >= lastDistance:
                            lastDistance = d
                            distances.append(d)
                            predictions.append(y_train_series[i])

                            if len(predictions) > neighbors_count:
                                # Dynamic threshold at 75th percentile
                                lastDistance = distances[int(neighbors_count * 3 / 4)]
                                distances.pop(0)
                                predictions.pop(0)

                # Calculate weighted prediction based on distances
                if len(predictions) > 0:
                    weights = 1 / (np.array(distances) + 1e-10)  # Distance-based weights
                    weights = weights / weights.sum()  # Normalize weights
                    prediction = np.sum(predictions * weights)  # Simple weighted average
                else:
                    prediction = 0  # Neutral if no predictions

                signal_data.update({
                    "Prediction": prediction,
                    "Max_Prediction": 1.0  # Since prediction is naturally in [-1, 1]
                })

            # Get market regime and volatility metrics
            current_volatility = df_1m['atr_pct'].iloc[-1]

            # Calculate momentum from 1m data for signal generation
            price_momentum = df_1m['close'].pct_change(3).iloc[-1]  # Simple 3-minute momentum

            # Get weights adjusted for regime and market conditions
            weights = self._get_regime_adjusted_weights(market_regime, symbol, current_volatility, price_momentum, signal_data)

            # Apply prediction weight adjustments for optimal conditions
            weights = self._adjust_prediction_weights(df_1m, df_3m, weights)

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            logging.info(f"""[{symbol}] Weight Distribution:
                [{symbol}] Market Regime: {market_regime}
                [{symbol}] MACD: {weights['MACD']:.3f}
                [{symbol}] EMA Fast: {weights['EMA_Fast']:.3f}
                [{symbol}] EMA Medium: {weights['EMA_Medium']:.3f}
                [{symbol}] ATR: {weights['ATR']:.3f}
                [{symbol}] Volatility: {current_volatility:.3f}%
                [{symbol}] Price Momentum: {price_momentum:.4f}
            """)

            # Calculate weighted signal with adjusted weights
            weighted_signal = self._calculate_weighted_signal(signal_data, symbol, weights)

            logging.info(f"[{symbol}] Weighted Signal: {weighted_signal:.3f}")

            # More aggressive thresholds for minute-scale trading
            base_threshold = 0.12  # Reduced from 0.15 for faster response

            # Get recent momentum with noise filtering
            smooth_changes = df_1m['close'].pct_change().ewm(span=3, adjust=False).mean()
            recent_momentum = abs(smooth_changes.rolling(3).sum().iloc[-1])

            # Dynamic threshold adjustment
            if market_regime == "volatile":
                threshold = base_threshold * 1.2  # More conservative in volatile markets
            elif market_regime == "trending":
                # In trending markets, adjust based on trend alignment
                if (weighted_signal > 0 and is_strong_uptrend) or (weighted_signal < 0 and is_strong_downtrend):
                    threshold = base_threshold * 0.85  # More aggressive when aligned with trend
                else:
                    threshold = base_threshold * 1.1  # More conservative when against trend
            elif market_regime == "ranging":
                threshold = base_threshold * 0.9  # More aggressive in ranging markets
            else:  # normal regime
                if (weighted_signal > 0 and is_strong_uptrend) or (weighted_signal < 0 and is_strong_downtrend):
                    threshold = base_threshold * 0.85  # Still aggressive for strong trend even in normal regime
                elif current_volatility < 0.2:  # Low volatility environment
                    threshold = base_threshold * 0.9
                    if recent_momentum > 0.003:  # Strong momentum in low vol
                        threshold *= 0.9  # Even more aggressive
                elif current_volatility > 0.8:  # High volatility environment
                    threshold = base_threshold * 1.1
                else:
                    threshold = base_threshold

            logging.info(f"""[{symbol}] Threshold Analysis:
                [{symbol}] Base Threshold: {base_threshold:.3f}
                [{symbol}] Market Regime: {market_regime}
                [{symbol}] Current Volatility: {current_volatility:.4f}
                [{symbol}] Recent Momentum: {recent_momentum:.4f}
                [{symbol}] Final Threshold: {threshold:.3f}
                [{symbol}] Weighted Signal: {weighted_signal:.3f}
            """)

            if weighted_signal > threshold:
                new_signal = "long"
            elif weighted_signal < -threshold:
                new_signal = "short"
            else:
                new_signal = "neutral"

            logging.info(f"[{symbol}] New Signal: {new_signal}")

            # Signal buffer logic with proper buffering
            current_time = time.time()
            if symbol in self.exchange.last_signal:
                last_signal = self.exchange.last_signal[symbol]
                time_since_last = current_time - self.exchange.last_signal_time[symbol]

                # Determine buffer time based on regime and signal strength
                buffer_time = 8  # Base buffer time
                if market_regime == "volatile":
                    buffer_time = 12  # Longer buffer to protect against whipsaws
                elif market_regime == "trending":
                    # Check for strong trend alignment
                    if (weighted_signal > 0 and is_strong_uptrend) or (weighted_signal < 0 and is_strong_downtrend):
                        # Dynamic buffer based on trend quality and signal strength
                        quality_factor = trend_quality * (0.8 + min(0.4, abs(weighted_signal)))
                        buffer_time = 6 + (quality_factor * 8)  # Range: 6-14s based on quality
                    else:
                        buffer_time = 6  # Shorter buffer when trend is uncertain
                elif market_regime == "ranging":
                    buffer_time = 4  # Shortest buffer to catch reversals quickly

                # Adjust buffer time based on signal strength with more granular thresholds
                signal_strength = abs(weighted_signal)
                if signal_strength > 0.25:  # Very strong signal
                    buffer_time *= 1.3
                elif signal_strength > 0.20:  # Strong signal
                    buffer_time *= 1.2
                elif signal_strength > 0.15:  # Moderate signal
                    buffer_time *= 1.1
                elif signal_strength < 0.10:  # Weak signal
                    buffer_time *= 0.9

                # Additional adjustments based on trend quality changes
                if trend_quality < 0.3:  # Weak trend quality
                    buffer_time *= 0.8  # Reduce buffer time
                elif trend_quality > 0.7:  # Strong trend quality
                    buffer_time *= 1.2  # Increase buffer time

                # Cap maximum buffer time
                buffer_time = min(buffer_time, 15)  # Maximum 15 seconds

                logging.info(f"""[{symbol}] Buffer Analysis:
                    [{symbol}] Base Buffer: 8.0s
                    [{symbol}] Market Regime: {market_regime}
                    [{symbol}] Trend Quality: {trend_quality:.3f}
                    [{symbol}] Signal Strength: {signal_strength:.3f}
                    [{symbol}] Final Buffer: {buffer_time:.1f}s
                """)

                if new_signal != 'neutral':  # Only buffer non-neutral signals
                    if new_signal == last_signal:  # Same signal
                        if time_since_last < buffer_time:
                            logging.info(f"[{symbol}] Signal buffered (regime: {market_regime}, buffer: {buffer_time}s)")
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
        # Original indicators
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['adx'] = self.exchange.n_adx(df['high'], df['low'], df['close'], 14)
        df['cci'] = self.exchange.n_cci(df['high'], df['low'], df['close'], 20, 1)
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        df['wt'] = self.exchange.n_wt(hlc3, 10, 11)
        atr_indicator = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True
        )
        df['atr'] = atr_indicator.average_true_range()
        df['atr_pct'] = (df['atr'] / df['close']) * 100

        # Add EMAs for regime detection (1m timeframe)
        df['regime_ema_fast'] = EMAIndicator(df['close'], window=8, fillna=True).ema_indicator()
        df['regime_ema_medium'] = EMAIndicator(df['close'], window=21, fillna=True).ema_indicator()
        df['regime_ema_slow'] = EMAIndicator(df['close'], window=55, fillna=True).ema_indicator()

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
        """Detect if a strong uptrend or downtrend exists with crypto-specific enhancements."""
        # Get latest values
        close = df['close'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_medium = df['ema_medium'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]

        # Calculate EMA slopes for momentum
        ema_fast_slope = (df['ema_fast'].iloc[-1] - df['ema_fast'].iloc[-2]) / df['ema_fast'].iloc[-2]
        ema_medium_slope = (df['ema_medium'].iloc[-1] - df['ema_medium'].iloc[-2]) / df['ema_medium'].iloc[-2]

        # Calculate EMA compression
        ema_ranges = [ema_fast, ema_medium, ema_slow]
        ema_compression = (max(ema_ranges) - min(ema_ranges)) / close

        # Enhanced trend conditions with flexibility
        is_uptrend = (
            close > ema_fast and                      # Price above fast EMA
            ema_fast_slope > 0 and                    # Rising fast EMA
            ema_medium_slope > 0 and                  # Rising medium EMA
            macd > macd_signal and                    # MACD confirmation
            (close > ema_medium or                    # Either price above medium EMA
             (ema_compression < 0.001 and             # Or EMAs are compressed
              ema_fast > ema_medium))                 # with fast above medium
        )

        is_downtrend = (
            close < ema_fast and                      # Price below fast EMA
            ema_fast_slope < 0 and                    # Falling fast EMA
            ema_medium_slope < 0 and                  # Falling medium EMA
            macd < macd_signal and                    # MACD confirmation
            (close < ema_medium or                    # Either price below medium EMA
             (ema_compression < 0.001 and             # Or EMAs are compressed
              ema_fast < ema_medium))                 # with fast below medium
        )

        # Ensure trends are mutually exclusive
        if is_uptrend and is_downtrend:
            is_uptrend = False
            is_downtrend = False

        return is_uptrend, is_downtrend

    def _detect_market_regime(self, df: pd.DataFrame, df_3m: pd.DataFrame, symbol: str) -> str:
        """Detect market regime based on volatility and momentum patterns."""
        try:
            # Volatility metrics (keep on 1m for accuracy)
            atr_pct = df['atr_pct'].iloc[-1]
            recent_volatility = df['atr_pct'].rolling(10).mean().iloc[-1]
            baseline_volatility = df['atr_pct'].rolling(30).mean().iloc[-1]
            vol_ratio = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1

            # Price metrics from 3m data for trend consistency
            close = df_3m['close'].iloc[-1]
            ema_fast = df_3m['ema_fast'].iloc[-1]
            ema_medium = df_3m['ema_medium'].iloc[-1]
            ema_slow = df_3m['ema_slow'].iloc[-1]

            # Calculate momentum from both timeframes
            # Short-term momentum from 1m for quick moves
            smooth_changes_1m = df['close'].pct_change().ewm(span=3, adjust=False).mean()
            short_momentum = smooth_changes_1m.rolling(3).sum().iloc[-1]  # 3m momentum

            # Medium-term momentum from 3m for trend alignment
            medium_momentum = df_3m['close'].pct_change(3).iloc[-1]  # 9m momentum

            # Trend alignment check using 3m EMAs
            trend_alignment = 0
            if close > ema_fast > ema_medium > ema_slow:
                trend_alignment = 1
            elif close < ema_fast < ema_medium < ema_slow:
                trend_alignment = -1

            # Count direction changes with minimum move filter (keep on 1m for accuracy)
            min_move = atr_pct * 0.1  # 10% of ATR as minimum significant move
            direction_changes = (
                (smooth_changes_1m.rolling(8)
                 .apply(lambda x: ((x > min_move) != (x < -min_move)).sum())
                 .iloc[-1])
            )

            # Log all metrics for analysis
            logging.info(f"""[{symbol}] Regime Detection Metrics:
                [{symbol}] Current ATR%: {atr_pct:.4f}
                [{symbol}] Recent Volatility (10m): {recent_volatility:.4f}
                [{symbol}] Baseline Volatility (30m): {baseline_volatility:.4f}
                [{symbol}] Volatility Ratio: {vol_ratio:.4f}
                [{symbol}] Short Momentum (1m): {short_momentum:.4f}
                [{symbol}] Medium Momentum (3m): {medium_momentum:.4f}
                [{symbol}] Trend Alignment: {trend_alignment}
                [{symbol}] Direction Changes (8m): {direction_changes}
                [{symbol}] Close: {close:.6f}
                [{symbol}] EMAs - Fast: {ema_fast:.6f}, Medium: {ema_medium:.6f}, Slow: {ema_slow:.6f}
            """)

            # Regime detection with adaptive thresholds
            # Calculate adaptive thresholds based on historical volatility
            vol_threshold = max(1.2, baseline_volatility / recent_volatility * 1.1)
            momentum_threshold_1m = max(0.005, atr_pct * 0.1)  # 10% of current volatility
            momentum_threshold_3m = max(0.01, atr_pct * 0.2)   # 20% of current volatility

            # 1. Volatile: Adaptive volatility ratio OR strong momentum
            if (vol_ratio > vol_threshold or
                abs(short_momentum) > momentum_threshold_1m or
                abs(medium_momentum) > momentum_threshold_3m):
                return 'volatile'

            # 2. Trending: Strong trend alignment with momentum confirmation
            elif ((trend_alignment != 0 and abs(medium_momentum) > momentum_threshold_3m * 0.5) or
                  (close < ema_fast < ema_medium < ema_slow and short_momentum < -momentum_threshold_1m * 0.3) or
                  (close > ema_fast > ema_medium > ema_slow and short_momentum > momentum_threshold_1m * 0.3)):
                return 'trending'

            # 3. Ranging: Adaptive low momentum thresholds
            elif (abs(medium_momentum) < momentum_threshold_3m * 0.15 and
                  abs(short_momentum) < momentum_threshold_1m * 0.2 and
                  direction_changes >= 3 and
                  vol_ratio < 0.9):
                return 'ranging'

            # 4. Normal: Default state
            return 'normal'

        except Exception as e:
            logging.error(f"[{symbol}] Error in detect_market_regime: {e}")
            logging.error(f"[{symbol}] {traceback.format_exc()}")
            return 'normal'  # Default to normal regime on error

    def _get_regime_adjusted_weights(self, market_regime: str, symbol: str, current_volatility: float, price_momentum: float, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Get weights adjusted for market regime and current market conditions."""
        # Base weights for signal generation
        weights = {
            "MACD": 0.25,       # Momentum and trend confirmation
            "EMA_Fast": 0.30,   # Immediate price action
            "EMA_Medium": 0.25, # Trend structure
            "ATR": 0.20,        # Volatility awareness
            "Prediction": 0.0   # Removed for simplicity
        }

        # Apply regime-specific adjustments
        if market_regime == "volatile":
            weights["MACD"] *= 0.7        # Reduce momentum
            weights["EMA_Fast"] *= 1.4    # Quick price action
            weights["EMA_Medium"] *= 1.2  # Trend awareness
            weights["ATR"] *= 1.5         # Volatility awareness

        elif market_regime == "ranging":
            weights["MACD"] *= 0.6        # Minimal momentum
            weights["EMA_Fast"] *= 1.3    # Price action
            weights["EMA_Medium"] *= 1.2  # Moderate trend
            weights["ATR"] *= 1.4         # Volatility awareness

        elif market_regime == "trending":
            weights["MACD"] *= 1.2        # Momentum importance
            weights["EMA_Fast"] *= 1.2    # Price action
            weights["EMA_Medium"] *= 1.3  # Trend confirmation
            weights["ATR"] *= 0.9         # Less volatility focus

            # Simple trend alignment check
            if ((price_momentum < 0 and signal_data["MACD"] < signal_data["MACD_Signal"]) or
                (price_momentum > 0 and signal_data["MACD"] > signal_data["MACD_Signal"])):
                weights["MACD"] *= 1.2
                weights["EMA_Medium"] *= 1.1

        else:  # normal regime
            if current_volatility > 1.5:
                weights["ATR"] *= 1.2
                weights["EMA_Fast"] *= 1.1
            elif current_volatility < 0.5:
                weights["MACD"] *= 0.9
                weights["ATR"] *= 1.1
                weights["EMA_Fast"] *= 1.2

        logging.info(f"""[{symbol}] Pre-normalized Weight Distribution:
            [{symbol}] Market Regime: {market_regime}
            [{symbol}] MACD: {weights['MACD']:.3f}
            [{symbol}] EMA Fast: {weights['EMA_Fast']:.3f}
            [{symbol}] EMA Medium: {weights['EMA_Medium']:.3f}
            [{symbol}] ATR: {weights['ATR']:.3f}
            [{symbol}] Volatility: {current_volatility:.3f}%
            [{symbol}] Price Momentum: {price_momentum:.4f}
        """)

        return weights

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

    def _calculate_weighted_signal(self, data: Dict[str, Any], symbol: str, weights: Dict[str, float]) -> float:
        """
        Calculate weighted trading signal optimized for minute-scale crypto scalping.
        Returns a value between -1 (strong sell) and 1 (strong buy).
        """
        try:
            # Get trend quality from signal data
            trend_quality = data['trend_quality']

            # 1. Price Momentum Component
            price_momentum = (data["Close"] - data["EMA_Fast"]) / data["EMA_Fast"]
            price_momentum = max(min(price_momentum, 1), -1)  # Bound between -1 and 1

            # 2. MACD Component
            macd_diff = data["MACD"] - data["MACD_Signal"]
            # Use MACD range for normalization, using half range for better sensitivity
            macd_range = data["MACD_Range"]
            if macd_range > 0:
                macd_signal = macd_diff / (macd_range * 0.5)  # Use half range for stronger signals
            else:
                macd_signal = 0
            macd_signal = max(min(macd_signal, 1), -1)  # Bound between -1 and 1

            # Log MACD details for analysis
            logging.info(f"[{symbol}] MACD Details - Diff: {macd_diff:.6f}, Range: {macd_range:.6f}, Signal: {macd_signal:.3f}")

            # 3. EMA Trend Component
            if data["Close"] < data["EMA_Fast"] < data["EMA_Medium"] < data["EMA_Slow"]:
                ema_trend = -1.0  # Perfect bearish alignment
            elif data["Close"] > data["EMA_Fast"] > data["EMA_Medium"] > data["EMA_Slow"]:
                ema_trend = 1.0   # Perfect bullish alignment
            elif data["Close"] < data["EMA_Fast"] < data["EMA_Medium"]:
                ema_diff = data["Close"] - data["EMA_Fast"]
                ema_trend = max(ema_diff / (data["ATR"] * 2), -1)  # Scale by 2 ATR
            elif data["Close"] > data["EMA_Fast"] > data["EMA_Medium"]:
                ema_diff = data["Close"] - data["EMA_Fast"]
                ema_trend = min(ema_diff / (data["ATR"] * 2), 1)  # Scale by 2 ATR
            else:
                ema_diff = data["Close"] - data["EMA_Medium"]
                ema_trend = ema_diff / (data["ATR"] * 2)  # Scale by 2 ATR
            ema_trend = max(min(ema_trend, 1), -1)  # Ensure bounds

            # First check for perfect technical alignment with meaningful separation
            if abs(ema_trend) == 1.0 and trend_quality >= 0.8:  # At least 80% of minimum separation
                # Adjust transfer percentages based on trend quality
                separation_factor = min(1.0, trend_quality)

                # Transfer from EMA_Fast to EMA_Medium
                fast_transfer = weights["EMA_Fast"] * (0.9 * separation_factor)
                weights["EMA_Fast"] *= (1 - 0.9 * separation_factor)
                weights["EMA_Medium"] += fast_transfer

                # Reduce ATR influence proportionally to separation
                if weights["ATR"] > 0.1:
                    atr_transfer = (weights["ATR"] - 0.1) * separation_factor
                    weights["ATR"] -= atr_transfer
                    weights["EMA_Medium"] += atr_transfer

                # If prediction aligns with perfect trend, transfer from MACD instead
                if "Prediction" in data and np.sign(data["Prediction"]) == np.sign(ema_trend):
                    pred_strength = abs(data["Prediction"])
                    pred_keep_pct = min(0.8, max(0.5, pred_strength * 0.7))

                    # Transfer from MACD proportionally to separation
                    macd_transfer = weights["MACD"] * (0.8 * separation_factor)
                    weights["MACD"] *= (1 - 0.8 * separation_factor)
                    weights["EMA_Medium"] += macd_transfer

                    logging.info(f"[{symbol}] Reduced MACD weight with separation factor {separation_factor:.3f}")
                    logging.info(f"[{symbol}] New MACD weight: {weights['MACD']:.3f}")
                    logging.info(f"[{symbol}] Added to EMA_Medium weight: +{macd_transfer:.3f}")

                    weights["Prediction"] *= pred_keep_pct
                    weights["EMA_Medium"] += weights["Prediction"] * (1 - pred_keep_pct)

                else:  # Prediction contradicts or doesn't exist
                    pred_transfer = weights["Prediction"] * (0.95 * separation_factor)
                    weights["Prediction"] *= (1 - 0.95 * separation_factor)
                    weights["EMA_Medium"] += pred_transfer

                    macd_transfer = weights["MACD"] * (0.8 * separation_factor)
                    weights["MACD"] *= (1 - 0.8 * separation_factor)
                    weights["EMA_Medium"] += macd_transfer

                logging.info(f"[{symbol}] Adjusted weights for perfect trend signal (separation factor: {separation_factor:.3f})")
                logging.info(f"[{symbol}] New prediction weight: {weights['Prediction']:.3f}")
                logging.info(f"[{symbol}] New EMA_Medium weight: {weights['EMA_Medium']:.3f}")
                logging.info(f"[{symbol}] New EMA_Fast weight: {weights['EMA_Fast']:.3f}")
                logging.info(f"[{symbol}] New ATR weight: {weights['ATR']:.3f}")

            # Then check for strong trend signals with meaningful separation
            elif abs(ema_trend) > 0.5 and trend_quality >= 0.5:  # At least 50% of minimum separation
                # Calculate transfer percentage based on trend strength
                trend_strength = abs(ema_trend)
                # For very strong trends (> 0.8), transfer up to 85%
                # For moderate trends (> 0.5), transfer minimum 65%
                transfer_pct = min(0.85, max(0.65, trend_strength * 0.8))

                # If MACD is weak, transfer more of its weight
                if abs(macd_signal) < 0.1:
                    macd_transfer = weights["MACD"] * 0.6  # Transfer 60% of MACD weight (up from 40%)
                    weights["MACD"] *= 0.4
                    weights["EMA_Medium"] += macd_transfer

                    logging.info(f"[{symbol}] Reduced MACD weight due to weak signal")
                    logging.info(f"[{symbol}] New MACD weight: {weights['MACD']:.3f}")
                    logging.info(f"[{symbol}] Added to EMA_Medium weight: +{macd_transfer:.3f}")

                # If prediction contradicts strong trend and is weak (< 0.2), transfer more weight
                if "Prediction" in data and np.sign(data["Prediction"]) != np.sign(ema_trend):
                    pred_strength = abs(data["Prediction"])
                    if pred_strength < 0.2:  # Weak contradicting prediction
                        transfer_pct = min(0.9, transfer_pct + 0.15)  # Boost transfer percentage

                    pred_transfer = weights["Prediction"] * transfer_pct
                    weights["Prediction"] *= (1 - transfer_pct)
                    weights["EMA_Medium"] += pred_transfer

                    logging.info(f"[{symbol}] Adjusted weights for strong trend signal")
                    logging.info(f"[{symbol}] Trend strength: {trend_strength:.3f}, Transfer percentage: {transfer_pct:.1%}")
                    logging.info(f"[{symbol}] Prediction strength: {pred_strength:.3f}")
                    logging.info(f"[{symbol}] New prediction weight: {weights['Prediction']:.3f}")
                    logging.info(f"[{symbol}] New EMA_Medium weight: {weights['EMA_Medium']:.3f}")

            # 4. Volatility Component (normalized to positive range [0, 0.3])
            volatility_signal = min(data["ATR"] / data["Close"] * 3, 0.3)  # Increased range and sensitivity

            # 5. Normalize prediction if available
            prediction = 0
            if "Prediction" in data:
                prediction = data["Prediction"]  # Already in [-1, 1] range
                logging.info(f"[{symbol}] Prediction: {prediction:.3f}")

            # 6. Calculate weighted signal
            weighted_signal = (
                price_momentum * weights["EMA_Fast"] +
                macd_signal * weights["MACD"] +
                ema_trend * weights["EMA_Medium"] +
                volatility_signal * weights["ATR"] +
                prediction * weights["Prediction"]
            )

            # 7. Bound final signal
            final_signal = max(min(weighted_signal, 1.0), -1.0)

            # 8. Log components and weights
            logging.info(f"[{symbol}] Signal Components")
            logging.info(f"[{symbol}] Price Momentum ({weights['EMA_Fast']:.2f}): {price_momentum:.3f}")
            logging.info(f"[{symbol}] MACD ({weights['MACD']:.2f}): {macd_signal:.3f}")
            logging.info(f"[{symbol}] EMA Trend ({weights['EMA_Medium']:.2f}): {ema_trend:.3f}")
            logging.info(f"[{symbol}] Volatility ({weights['ATR']:.2f}): {volatility_signal:.3f}")
            logging.info(f"[{symbol}] Prediction ({weights['Prediction']:.2f}): {prediction:.3f}")
            logging.info(f"[{symbol}] Final: {final_signal:.3f}")
            logging.info(f"[{symbol}] Price: {data['Close']:.2f}")

            return final_signal

        except Exception as e:
            logging.error(f"[{symbol}] Error in calculate_weighted_signal: {e}")
            return 0.0

    def _is_emas_packed(self, df_3m, threshold=0.0015):
        """Check if EMAs are tightly packed within threshold."""
        latest = df_3m.iloc[-1]
        ema_values = [
            latest['ema_fast'],
            latest['ema_medium'],
            latest['ema_slow']
        ]
        max_diff = max(ema_values) - min(ema_values)
        return (max_diff / latest['close']) < threshold

    def _check_low_volatility(self, df_1m, atr_threshold=0.005):
        """Check if volatility is below threshold."""
        return df_1m['atr_pct'].iloc[-1] < atr_threshold

    def _adjust_prediction_weights(self, df_1m, df_3m, weights):
        """
        Adjust weights based on market conditions.
        For Lorentzian predictions (already in [-1, 1]), we use more conservative adjustments.
        """
        adjusted_weights = weights.copy()

        if self._check_low_volatility(df_1m) and self._is_emas_packed(df_3m):
            volatility_factor = 1 - (df_1m['atr_pct'].iloc[-1] / 0.005)

            # Calculate EMA pack factor
            latest = df_3m.iloc[-1]
            ema_values = [latest['ema_fast'], latest['ema_medium'], latest['ema_slow']]
            max_diff = max(ema_values) - min(ema_values)
            ema_pack_factor = 1 - (max_diff / latest['close'] / 0.0015)

            # More conservative increase (up to 25% more instead of 50%)
            increase = min(volatility_factor, ema_pack_factor) * 0.25

            # Adjust weights
            extra_weight = adjusted_weights["Prediction"] * increase
            adjusted_weights["Prediction"] *= (1 + increase)

            # Reduce other weights proportionally
            reduction_per_weight = extra_weight / (len(adjusted_weights) - 1)
            for key in adjusted_weights:
                if key != "Prediction":
                    adjusted_weights[key] -= reduction_per_weight

            logging.info(f"Adjusted weights due to optimal conditions:")
            logging.info(f"Volatility factor: {volatility_factor:.3f}")
            logging.info(f"EMA pack factor: {ema_pack_factor:.3f}")
            logging.info(f"Weight increase: {increase:.3f}")

        return adjusted_weights

    def _calculate_trend_quality(self, df_3m: pd.DataFrame, df_1m: pd.DataFrame) -> float:
        """Calculate trend quality based on EMA separations and trend structure, optimized for DCA grid trading."""
        try:
            close = df_3m['close'].iloc[-1]
            ema_fast = df_3m['ema_fast'].iloc[-1]
            ema_medium = df_3m['ema_medium'].iloc[-1]
            ema_slow = df_3m['ema_slow'].iloc[-1]

            # Calculate EMA separations using ATR for normalization
            atr = df_1m['atr'].iloc[-1]
            if atr > 0:
                ema_fast_med_sep = abs(ema_fast - ema_medium) / atr
                ema_med_slow_sep = abs(ema_medium - ema_slow) / atr
                price_ema_dist = abs(close - ema_fast) / atr
            else:
                ema_fast_med_sep = 0
                ema_med_slow_sep = 0
                price_ema_dist = 0

            # Price-EMA distance factor (tighter for grid levels)
            dist_factor = min(1.0, price_ema_dist / 1.5)  # Reduced from 2.0 to 1.5 ATR units

            # Enhanced separation quality calculation with grid-optimized thresholds
            min_sep = min(ema_fast_med_sep, ema_med_slow_sep)
            max_sep = max(ema_fast_med_sep, ema_med_slow_sep)
            separation_quality = min_sep / max_sep if max_sep > 0 else 0.0

            # Separation ratio penalty (adjusted for grid trading)
            separation_ratio = ema_fast_med_sep / ema_med_slow_sep if ema_med_slow_sep > 0 else 0.0
            if separation_ratio > 3.0 or separation_ratio < 0.33:  # Tighter bounds for grid levels
                separation_quality *= 0.6  # 40% penalty for very uneven separations
            elif separation_ratio > 2.0 or separation_ratio < 0.5:  # More conservative thresholds
                separation_quality *= 0.8  # 20% penalty for moderately uneven separations

            # Add grid-specific boost for optimal separation
            if 0.75 <= separation_ratio <= 1.5:  # Tighter ideal range for grid trading
                separation_quality *= 1.3  # 30% boost for ideal grid spacing

            # Enhanced alignment score with grid-optimized distance consideration
            if (close > ema_fast > ema_medium > ema_slow):  # Bullish alignment
                alignment_score = 1.0
                if dist_factor > 0.7:  # Reduced from 0.8 for tighter grid levels
                    alignment_score *= 0.85
                elif dist_factor < 0.3:  # Increased from 0.2 for more grid opportunities
                    alignment_score *= 1.2  # Increased boost for tight alignment
            elif (close < ema_fast < ema_medium < ema_slow):  # Bearish alignment
                alignment_score = 1.0
                if dist_factor > 0.7:
                    alignment_score *= 0.85
                elif dist_factor < 0.3:
                    alignment_score *= 1.2
            elif ((close > ema_fast > ema_medium and ema_medium >= ema_slow) or
                  (close < ema_fast < ema_medium and ema_medium <= ema_slow)):
                alignment_score = 0.7
            else:
                alignment_score = 0.3

            # Enhanced momentum quality using ATR for normalization
            momentum_atr = (df_3m['close'].iloc[-1] - df_3m['close'].iloc[-4]) / (atr * 3) if atr > 0 else 0
            momentum_quality = min(1.0, abs(momentum_atr))  # Normalized by ATR

            # Momentum consistency check with grid-optimized thresholds
            short_momentum = (df_3m['close'].iloc[-1] - df_3m['close'].iloc[-3]) / (atr * 2) if atr > 0 else 0
            med_momentum = (df_3m['close'].iloc[-1] - df_3m['close'].iloc[-5]) / (atr * 4) if atr > 0 else 0

            if abs(momentum_atr) < 0.15:  # Reduced from 0.2 for more sensitive momentum detection
                momentum_quality *= 0.8  # Less severe penalty
            elif np.sign(short_momentum) != np.sign(med_momentum):
                momentum_quality *= 0.9  # Less severe penalty for inconsistency

            # Final weighted calculation with grid-optimized weights
            trend_quality = (
                0.40 * separation_quality +    # Increased weight for separation
                0.40 * alignment_score +       # Equal weight for alignment
                0.20 * momentum_quality        # Same weight for momentum
            )

            # Enhanced volatility adjustments using ATR percentage
            volatility = df_1m['atr_pct'].iloc[-1]  # ATR% is already price-independent
            if volatility > 0.3:  # Increased from 0.25% for wider grid levels
                trend_quality *= (1 - min(0.3, (volatility - 0.3) / 0.3))  # Less aggressive penalty
            elif volatility < 0.1:  # Increased from 0.08% for wider grid levels
                boost_factor = (0.1 - volatility) / 0.1
                if momentum_quality > 0.5:  # Reduced threshold for grid trading
                    trend_quality *= (1 + boost_factor * 0.2)  # Reduced boost for stability
                else:
                    trend_quality *= (1 + boost_factor * 0.1)  # Minimal boost for weak momentum

            # Additional boost for perfect alignment with good separation
            if alignment_score >= 1.0 and separation_quality > 0.6:  # Reduced from 0.7 for grid trading
                trend_quality *= 1.2  # Reduced from 1.3 for more stable grid signals

            return max(0.0, min(1.0, trend_quality))

        except Exception as e:
            logging.error(f"Error calculating trend quality: {e}")
            return 0.0

