from typing import Dict, Tuple, Any, TYPE_CHECKING
from logging import Logger as LoggerType

from .signal_logger import SignalLogger


if TYPE_CHECKING:
    from directionalscalper.core.exchanges.exchange import Exchange

import time
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
import pandas as pd
import numpy as np
import traceback
from ..strategies.logger import Logger

logging = Logger(logger_name="SignalGenerator", filename="SignalGenerator.log", stream=True)

class SignalGenerator:
    def __init__(self, exchange: "Exchange"):
        self.exchange = exchange
        self.signal_logger = SignalLogger()

    def cleanup(self):
        """Cleanup resources by closing the signal logger."""
        try:
            if hasattr(self, 'signal_logger'):
                self.signal_logger.cleanup()
        except Exception as e:
            print(f"Error during SignalGenerator cleanup: {e}")
            print(traceback.format_exc())

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self.cleanup()

    def generate(self, ohlcv_1m, ohlcv_3m, symbol, neighbors_count=8, use_adx_filter=False, adx_threshold=20):
        """Generate trading signals using balanced approach between old and new implementations."""
        self.signal_logger.cleanup()

        logger = self.signal_logger.get_logger(symbol)

        try:
            # Convert list to DataFrame if needed
            if isinstance(ohlcv_1m, list):
                logger.info("Converting 1m data to DataFrame")

                df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                df_1m.set_index('timestamp', inplace=True)
            else:
                df_1m = ohlcv_1m.copy()

            if isinstance(ohlcv_3m, list):
                logger.info("Converting 3m data to DataFrame")

                df_3m = pd.DataFrame(ohlcv_3m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'], unit='ms')
                df_3m.set_index('timestamp', inplace=True)
            else:
                df_3m = ohlcv_3m.copy()

            logger.info("Starting signal generation")

            # Log last 3 candles information
            try:
                last_3_candles = df_1m[['open', 'high', 'low', 'close', 'volume']].tail(3)
                if len(last_3_candles) > 0:
                    logger.info("Last 3 candles (1m):")
                    prev_close = None
                    for idx, candle in last_3_candles.iterrows():
                        try:
                            candle_body = abs(candle['close'] - candle['open'])
                            candle_total_range = candle['high'] - candle['low']
                            body_to_range_ratio = (candle_body / candle_total_range * 100) if candle_total_range != 0 else 0
                            direction = "Bullish" if candle['close'] > candle['open'] else "Bearish"

                            # Calculate change from previous candle
                            change_str = ""
                            if prev_close is not None:
                                change_pct = ((candle['close'] - prev_close) / prev_close) * 100
                                change_str = f" | Chg: {change_pct:+.2f}%"

                            # Check for high volatility candle
                            vol_note = ""
                            if body_to_range_ratio > 80:
                                vol_note = " [Strong]"
                            elif candle_total_range/candle['open'] > 0.003:  # 0.3% range
                                vol_note = " [High Range]"

                            # Safe timestamp formatting
                            try:
                                if isinstance(idx, pd.Timestamp):
                                    timestamp = idx.strftime('%H:%M:%S')
                                else:
                                    timestamp = str(idx)
                            except:
                                timestamp = str(idx)

                            logger.info(f"{timestamp} - {direction}{vol_note} | O: {candle['open']:.4f} H: {candle['high']:.4f} L: {candle['low']:.4f} C: {candle['close']:.4f} | Body/Range: {body_to_range_ratio:.1f}%{change_str} | Vol: {candle['volume']:.2f}")
                            prev_close = candle['close']
                        except Exception as e:
                            logger.error(f"Error processing candle at {idx}: {str(e)}")
                else:
                    logger.warning("No recent candles available for analysis")
            except Exception as e:
                logger.error(f"Error analyzing recent candles: {str(e)}")

            # Calculate indicators
            try:
                logger.info("Starting indicator calculations")

                # Calculate indicators
                df_1m = self._calculate_indicators(df_1m)
                df_3m = self.calculate_trend(df_3m)

                try:
                    logger.info("3m indicators calculated successfully")
                    logger.info(f"EMA ranges - Fast: {df_3m['ema_fast'].min():.4f} to {df_3m['ema_fast'].max():.4f}")
                    logger.info(f"EMA ranges - Medium: {df_3m['ema_medium'].min():.4f} to {df_3m['ema_medium'].max():.4f}")
                    logger.info(f"EMA ranges - Slow: {df_3m['ema_slow'].min():.4f} to {df_3m['ema_slow'].max():.4f}")
                    logger.info(f"MACD range: {df_3m['macd'].min():.4f} to {df_3m['macd'].max():.4f}")
                except Exception as e:
                    logger.error(f"3m indicator calculation failed: {e}")
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
                    logger.error(f"NaN values in latest bar - 1m: {nan_1m}, 3m: {nan_3m}")
                    logger.error(f"1m values: {latest_1m.to_dict()}")
                    logger.error(f"3m values: {latest_3m.to_dict()}")

                    # Log the last few rows of price data
                    logger.error("Last 5 rows of 1m price data:")
                    logger.error(df_1m[['open', 'high', 'low', 'close']].tail().to_string())
                    return 'neutral'

                logger.info("All indicators calculated successfully")

            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                logger.error(traceback.format_exc())
                return 'neutral'

            # Detect trend
            is_strong_uptrend, is_strong_downtrend = self.detect_trend(df_3m, logger=logger)

            logger.info("3m Trend Analysis:")
            logger.info(f"Strong Uptrend: {is_strong_uptrend}")
            logger.info(f"Strong Downtrend: {is_strong_downtrend}")

            # Calculate market regime from 1m data
            market_regime = self._detect_market_regime(df_1m, df_3m, symbol, logger=logger)
            # logging.info(f"[{symbol}] Detecting market regime - atr_pct: {df_1m['atr_pct'].iloc[-1]}")

            logger.info(f"Market Regime: {market_regime}")
            latest_atr_pct = df_1m['atr_pct'].iloc[-1]
            logger.info(f"ATR %: {latest_atr_pct:.4f}")

            # Calculate trend quality once
            trend_quality = self._calculate_trend_quality(df_3m, df_1m, logger=logger)

            logger.info(f"EMA Separations - Fast-Med: {abs(df_3m['ema_fast'].iloc[-1] - df_3m['ema_medium'].iloc[-1]) / df_3m['close'].iloc[-1]:.6f}, Med-Slow: {abs(df_3m['ema_medium'].iloc[-1] - df_3m['ema_slow'].iloc[-1]) / df_3m['close'].iloc[-1]:.6f}")
            logger.info(f"Trend Quality: {trend_quality:.3f}")

            # Prepare data for weighted signal calculation
            signal_data = {
                "Close": df_3m['close'].iloc[-2],  # Use completed candle
                'MACD': df_3m['macd'].iloc[-2],
                'MACD_Signal': df_3m['macd_signal'].iloc[-2],
                'EMA_Fast': df_3m['ema_fast'].iloc[-2],
                'EMA_Medium': df_3m['ema_medium'].iloc[-2],
                'EMA_Slow': df_3m['ema_slow'].iloc[-2],
                'ATR': df_3m['atr'].iloc[-2],  # Switch to 3m ATR for stability
                'MACD_Range': df_3m['macd'].abs().max(),
                'trend_quality': trend_quality,
                # Use completed candles for pattern analysis
                'last_three_closes': df_3m['close'].iloc[-4:-1].values.tolist(),  # -4 to -1 to get completed candles
                'last_three_volumes': df_3m['volume'].iloc[-4:-1].values.tolist()
            }

            # Log completed vs current candle for analysis
            logger.info("Candle Analysis:")
            logger.info(f"Last Complete (3m) - Close: {df_3m['close'].iloc[-2]:.4f}, Volume: {df_3m['volume'].iloc[-2]:.2f}")
            logger.info(f"Current (3m) - Close: {df_3m['close'].iloc[-1]:.4f}, Volume: {df_3m['volume'].iloc[-1]:.2f}")

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
            weights = self._get_regime_adjusted_weights(market_regime, symbol, current_volatility, price_momentum, signal_data, logger=logger)

            # Apply prediction weight adjustments for optimal conditions
            weights = self._adjust_prediction_weights(df_1m, df_3m, weights, logger=logger)

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            logger.info("Weight Distribution:")
            logger.info(f"Market Regime: {market_regime}")
            logger.info(f"MACD: {weights['MACD']:.3f}")
            logger.info(f"EMA Fast: {weights['EMA_Fast']:.3f}")
            logger.info(f"EMA Medium: {weights['EMA_Medium']:.3f}")
            logger.info(f"ATR: {weights['ATR']:.3f}")
            logger.info(f"Volatility: {current_volatility:.3f}%")
            logger.info(f"Price Momentum: {price_momentum:.4f}")

            # Calculate weighted signal with adjusted weights
            weighted_signal = self._calculate_weighted_signal(signal_data, symbol, weights, logger=logger)

            logger.info(f"Weighted Signal: {weighted_signal:.3f}")

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

            logger.info("Threshold Analysis:")
            logger.info(f"Base Threshold: {base_threshold:.3f}")
            logger.info(f"Market Regime: {market_regime}")
            logger.info(f"Current Volatility: {current_volatility:.4f}")
            logger.info(f"Recent Momentum: {recent_momentum:.4f}")
            logger.info(f"Final Threshold: {threshold:.3f}")
            logger.info(f"Weighted Signal: {weighted_signal:.3f}")

            if weighted_signal > threshold:
                new_signal = "long"
            elif weighted_signal < -threshold:
                new_signal = "short"
            else:
                new_signal = "neutral"

            logger.info(f"[{symbol}] New Signal: {new_signal}")

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

                # Apply volatility adjustment
                if market_regime == "volatile" and current_volatility > 2.0:
                    buffer_time *= 0.8  # Reduce buffer time in highly volatile conditions

                logger.info("Buffer Analysis:")
                logger.info("Base Buffer: 8.0s")
                logger.info(f"Market Regime: {market_regime}")
                logger.info(f"Trend Quality: {trend_quality:.3f}")
                logger.info(f"Signal Strength: {signal_strength:.3f}")
                logger.info(f"Final Buffer: {buffer_time:.1f}s")

                signal_data['signal'] = new_signal
                signal_data['buffer_time'] = buffer_time
                signal_data['market_regime'] = market_regime
                signal_data['trend_quality'] = trend_quality
                signal_data['signal_strength'] = signal_strength
                signal_data['threshold'] = threshold

                if new_signal != 'neutral':  # Only buffer non-neutral signals
                    if new_signal == last_signal:  # Same signal
                        if time_since_last < buffer_time:
                            logger.info(f"Signal buffered (regime: {market_regime}, buffer: {buffer_time}s)")
                            new_signal = last_signal  # Return last signal during buffer period
                        else:
                            # Update time but keep the signal
                            self.exchange.last_signal_time[symbol] = current_time

                    else:  # Different signal
                        # Update both signal and time
                        self.exchange.last_signal[symbol] = new_signal
                        self.exchange.last_signal_time[symbol] = current_time

                else:  # Neutral signals aren't buffered
                    self.exchange.last_signal[symbol] = new_signal
                    self.exchange.last_signal_time[symbol] = current_time

            else:  # First signal for this symbol
                self.exchange.last_signal[symbol] = new_signal
                self.exchange.last_signal_time[symbol] = current_time

            self.signal_logger.log_signal(symbol, signal_data, logger=logger)

            del df_1m, df_3m

            logging.info(f"New signal: {new_signal}")

            return new_signal
        except Exception as e:
            logger.error(f"Error in generate_l_signals_from_data: {e}")
            logger.error(f"{traceback.format_exc()}")
            # # Clean up memory even on error
            # try:
            #     del df_1m, df_3m
            # except:
            #     pass
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

        # Add ATR calculation for 3m timeframe
        atr_indicator = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True
        )
        df['atr'] = atr_indicator.average_true_range()
        df['atr_pct'] = (df['atr'] / df['close']) * 100

        return df

    def detect_trend(self, df: pd.DataFrame, logger: LoggerType) -> Tuple[bool, bool]:
        """Detect if a strong uptrend or downtrend exists with crypto-specific enhancements."""
        # Get latest completed candle values
        close = df['close'].iloc[-2]
        ema_fast = df['ema_fast'].iloc[-2]
        ema_medium = df['ema_medium'].iloc[-2]
        ema_slow = df['ema_slow'].iloc[-2]
        macd = df['macd'].iloc[-2]
        macd_signal = df['macd_signal'].iloc[-2]

        # Calculate EMA slopes using completed candles
        ema_fast_slope = (df['ema_fast'].iloc[-2] - df['ema_fast'].iloc[-3]) / df['ema_fast'].iloc[-3]
        ema_medium_slope = (df['ema_medium'].iloc[-2] - df['ema_medium'].iloc[-3]) / df['ema_medium'].iloc[-3]

        # Calculate EMA compression using completed candles
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

        # Log trend detection using completed candles
        logger.info("Trend Detection (Using Completed Candles):")
        logger.info(f"Close: {close:.4f}")
        logger.info(f"EMAs - Fast: {ema_fast:.4f}, Medium: {ema_medium:.4f}, Slow: {ema_slow:.4f}")
        logger.info(f"EMA Slopes - Fast: {ema_fast_slope:.6f}, Medium: {ema_medium_slope:.6f}")
        logger.info(f"MACD: {macd:.6f}, Signal: {macd_signal:.6f}")
        logger.info(f"EMA Compression: {ema_compression:.6f}")

        # Ensure trends are mutually exclusive
        if is_uptrend and is_downtrend:
            is_uptrend = False
            is_downtrend = False

        return is_uptrend, is_downtrend

    def _detect_market_regime(self, df: pd.DataFrame, df_3m: pd.DataFrame, symbol: str, logger: LoggerType) -> str:
        """Detect market regime based on volatility and momentum patterns."""
        try:
            # Base calculations on completed candles
            atr_pct = df['atr_pct'].iloc[-2]  # Completed candle
            recent_volatility = df_3m['atr_pct'].rolling(4).mean().iloc[-2]
            baseline_volatility = df_3m['atr_pct'].rolling(10).mean().iloc[-2]
            vol_ratio = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1

            # Early warning checks using current candle for immediate detection
            current_atr = df['atr_pct'].iloc[-1]
            current_close = df['close'].iloc[-1]
            last_close = df['close'].iloc[-2]
            current_volume = df['volume'].iloc[-1]  # Keep current for warnings
            completed_volume = df['volume'].iloc[-2]  # Use for regime confirmation
            avg_volume = df['volume'].rolling(10).mean().iloc[-2]

            # Detect potential volatility events with dynamic thresholds
            volatility_warning = False
            vol_spike_threshold = max(1.5, 1 + (atr_pct * 2.5))
            if current_atr > atr_pct * vol_spike_threshold:
                volatility_warning = True
                logger.info(f"Volatility Warning: Current ATR {current_atr:.4f} vs Complete {atr_pct:.4f}")

            # Detect potential trend reversal with volume confirmation
            reversal_warning = False
            price_change = abs(current_close - last_close) / last_close
            volume_surge = current_volume > avg_volume * 2.0  # Use current for immediate detection
            if price_change > atr_pct * 1.5 and volume_surge:
                reversal_warning = True
                logger.info(f"Reversal Warning: Price change {price_change:.4f} with {current_volume/avg_volume:.1f}x volume")

            # Price metrics from 3m data for trend consistency
            close = df_3m['close'].iloc[-2]
            ema_fast = df_3m['ema_fast'].iloc[-2]
            ema_medium = df_3m['ema_medium'].iloc[-2]
            ema_slow = df_3m['ema_slow'].iloc[-2]

            # Calculate momentum from both timeframes using completed candles
            smooth_changes_1m = df['close'].pct_change().ewm(span=3, adjust=False).mean()
            short_momentum = smooth_changes_1m.rolling(3).sum().iloc[-2]
            medium_momentum = df_3m['close'].pct_change(3).iloc[-2]

            # EMA separation and compression checks
            ema_fast_med_separation = abs(ema_fast - ema_medium) / close
            ema_med_slow_separation = abs(ema_medium - ema_slow) / close
            ema_compression = max(ema_fast_med_separation, ema_med_slow_separation)

            # Trend alignment check using 3m EMAs
            trend_alignment = 0
            if close > ema_fast > ema_medium > ema_slow:
                trend_alignment = 1
            elif close < ema_fast < ema_medium < ema_slow:
                trend_alignment = -1

            # Count direction changes with minimum move filter
            min_move = atr_pct * 0.15  # Increased for crypto's larger moves
            direction_changes = (
                (smooth_changes_1m.rolling(8)
                 .apply(lambda x: ((x > min_move) != (x < -min_move)).sum())
                 .iloc[-2])
            )

            # Log all metrics for analysis
            logger.info("Regime Detection Metrics (Using Completed Candles):")
            logger.info(f"Current ATR% (1m): {atr_pct:.4f}")
            logger.info(f"Recent Volatility (12m): {recent_volatility:.4f}")
            logger.info(f"Baseline Volatility (30m): {baseline_volatility:.4f}")
            logger.info(f"Volatility Ratio: {vol_ratio:.4f}")
            logger.info(f"Short Momentum (1m): {short_momentum:.4f}")
            logger.info(f"Medium Momentum (3m): {medium_momentum:.4f}")
            logger.info(f"Trend Alignment: {trend_alignment}")
            logger.info(f"Direction Changes (8m): {direction_changes}")
            logger.info(f"EMA Compression: {ema_compression:.6f}")

            # Compare with current candle
            current_momentum = smooth_changes_1m.rolling(3).sum().iloc[-1]
            logger.info("Current vs Last Complete Candle:")
            logger.info(f"ATR% - Current: {current_atr:.4f}, Complete: {atr_pct:.4f}")
            logger.info(f"Momentum - Current: {current_momentum:.4f}, Complete: {short_momentum:.4f}")

            # Regime detection with adaptive thresholds
            vol_threshold = max(1.2, baseline_volatility / recent_volatility * 1.2)  # Good for crypto
            momentum_threshold_1m = max(0.004, atr_pct * 0.12)  # Increased for crypto's rapid moves
            momentum_threshold_3m = max(0.008, atr_pct * 0.20)  # Increased for medium-term momentum

            # Adjust thresholds if warnings are active
            if volatility_warning:
                vol_threshold *= 0.85  # More sensitive to volatility
                momentum_threshold_1m *= 1.5  # Increased from 1.3 for crypto
                momentum_threshold_3m *= 1.5  # Increased from 1.3 for crypto

            if reversal_warning:
                momentum_threshold_1m *= 1.6  # Increased from 1.4 for crypto
                momentum_threshold_3m *= 1.6  # Increased from 1.4 for crypto

            # 1. Volatile regime detection - Use current candle for immediate response
            if (
                (vol_ratio > vol_threshold and current_volume > avg_volume * 1.8) or  # Use current for spikes
                (abs(short_momentum) > momentum_threshold_1m and
                 abs(medium_momentum) > momentum_threshold_3m * 0.8) or
                atr_pct > 0.5 or
                (current_atr > atr_pct * 2.0 and current_volume > avg_volume * 1.8)  # Use current for spikes
            ):
                return 'volatile'

            # 2. Trending regime detection - Use completed candle for confirmation
            elif (
                abs(trend_alignment) == 1 and
                abs(medium_momentum) > momentum_threshold_3m * 0.15 and
                (
                    # Either good EMA structure with moderate volume
                    (completed_volume > avg_volume * 1.0 and  # Reduced from 1.2
                     ema_compression > 0.003) or
                    # Or strong momentum with any volume
                    (abs(medium_momentum) > momentum_threshold_3m * 0.3)
                ) and
                direction_changes <= 2
            ):
                # Log volume comparison for analysis
                logger.info("Volume Analysis for Trend Detection:")
                logger.info(f"Completed Volume: {completed_volume:.2f}")
                logger.info(f"Average Volume: {avg_volume:.2f}")
                logger.info(f"Volume Ratio: {completed_volume/avg_volume:.2f}x")
                return 'trending'

            # 3. Normal regime detection - Use completed candle for stability
            elif (
                0.75 < vol_ratio < 1.25 and
                ema_compression > 0.0035 and
                abs(short_momentum) < momentum_threshold_1m * 0.7 and
                abs(medium_momentum) < momentum_threshold_3m * 0.8 and
                direction_changes < 2 and
                0.7 < completed_volume / avg_volume < 1.3  # Use completed for normal confirmation
            ):
                return 'normal'

            # All other conditions are considered ranging (default state)
            return 'ranging'

        except Exception as e:
            logger.error(f"Error in detect_market_regime: {e}")
            logger.error(f"{traceback.format_exc()}")
            return 'ranging'  # Default to ranging on error

    def _get_regime_adjusted_weights(self, market_regime: str, symbol: str, current_volatility: float, price_momentum: float, signal_data: Dict[str, Any], logger: LoggerType) -> Dict[str, float]:
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
            weights["MACD"] *= 0.9        # Increased from 0.7 for stronger momentum influence
            weights["EMA_Fast"] *= 1.2    # Reduced from 1.3 to avoid trend bias
            weights["EMA_Medium"] *= 1.1  # Reduced from 1.2
            weights["ATR"] *= 1.4         # Reduced from 1.6 for less volatility focus

            # Additional volatility adjustments
            if current_volatility > 3.0:  # Extra high volatility
                weights["MACD"] *= 1.3    # Further increase MACD importance
                weights["ATR"] *= 1.1     # Slight increase in volatility awareness
                weights["EMA_Fast"] *= 0.9  # Reduce fast EMA influence

        elif market_regime == "ranging":
            weights["MACD"] *= 0.5        # Reduced momentum importance
            weights["EMA_Fast"] *= 1.4    # Increased for price action
            weights["EMA_Medium"] *= 1.3  # Increased for trend structure
            weights["ATR"] *= 1.5         # Increased for volatility awareness

        elif market_regime == "trending":
            weights["MACD"] *= 1.3        # Increased momentum importance
            weights["EMA_Fast"] *= 1.1    # Slightly reduced for less noise
            weights["EMA_Medium"] *= 1.4  # Increased for stronger trend following
            weights["ATR"] *= 0.8         # Reduced volatility focus

            # Enhanced trend alignment check with price levels
            current_close = signal_data["Close"]
            current_ema_fast = signal_data["EMA_Fast"]
            if ((price_momentum < 0 and signal_data["MACD"] < signal_data["MACD_Signal"] and current_close < current_ema_fast) or
                (price_momentum > 0 and signal_data["MACD"] > signal_data["MACD_Signal"] and current_close > current_ema_fast)):
                weights["MACD"] *= 1.3
                weights["EMA_Medium"] *= 1.2

        else:  # normal regime
            if current_volatility > 1.5:
                weights["ATR"] *= 1.2
                weights["EMA_Fast"] *= 1.1
            elif current_volatility < 0.5:
                weights["MACD"] *= 0.9
                weights["ATR"] *= 1.1
                weights["EMA_Fast"] *= 1.2

        logger.info("Pre-normalized Weight Distribution:")
        logger.info(f"Market Regime: {market_regime}")
        logger.info(f"MACD: {weights['MACD']:.3f}")
        logger.info(f"EMA Fast: {weights['EMA_Fast']:.3f}")
        logger.info(f"EMA Medium: {weights['EMA_Medium']:.3f}")
        logger.info(f"ATR: {weights['ATR']:.3f}")
        logger.info(f"Volatility: {current_volatility:.3f}%")
        logger.info(f"Price Momentum: {price_momentum:.4f}")

        return weights

    def _calculate_weighted_signal(self, data: Dict[str, Any], symbol: str, weights: Dict[str, float], logger: LoggerType) -> float:
        try:
            trend_quality = data['trend_quality']

            # Enhanced trend quality factors
            strong_trend = trend_quality > 0.7
            weak_trend = trend_quality < 0.3

            # Price Momentum with enhanced breakout detection
            price_momentum = (data["Close"] - data["EMA_Fast"]) / data["EMA_Fast"]

            # Enhanced momentum for failed breakouts with trend consideration
            if abs(price_momentum) > 0.001:
                last_three_closes = data.get("last_three_closes", [])
                if len(last_three_closes) >= 3:
                    last_three_volumes = data.get("last_three_volumes", [])
                    # Failed breakout detection with volume confirmation
                    if (price_momentum < 0 and
                        last_three_closes[1] > last_three_closes[0] and
                        last_three_closes[2] < last_three_closes[1] and
                        last_three_volumes[2] < last_three_volumes[1]):
                        momentum_multiplier = 1.2
                        if weak_trend:  # Stronger adjustment in weak trends
                            momentum_multiplier = 1.4
                        price_momentum *= momentum_multiplier
                    elif (price_momentum > 0 and
                          last_three_closes[1] < last_three_closes[0] and
                          last_three_closes[2] > last_three_closes[1] and
                          last_three_volumes[2] < last_three_volumes[1]):
                        momentum_multiplier = 1.2
                        if weak_trend:  # Stronger adjustment in weak trends
                            momentum_multiplier = 1.4
                        price_momentum *= momentum_multiplier

            # Ensure momentum sign matches price movement direction
            price_direction = np.sign(data["Close"] - data["EMA_Fast"])
            if abs(price_momentum) > 0:
                price_momentum = abs(price_momentum) * price_direction

            price_momentum = max(min(price_momentum, 1), -1)

            # MACD with trend quality consideration
            macd_diff = data["MACD"] - data["MACD_Signal"]
            macd_range = data["MACD_Range"]
            if macd_range > 0:
                macd_signal = macd_diff / (macd_range * 0.5)
                if strong_trend and np.sign(macd_signal) == np.sign(price_momentum):
                    macd_signal *= 1.3  # Stronger boost in strong trends
                elif weak_trend:
                    macd_signal *= 0.7  # Reduce MACD influence in weak trends
            else:
                macd_signal = 0
            macd_signal = max(min(macd_signal, 1), -1)

            # Enhanced EMA trend component with momentum alignment
            if data["Close"] < data["EMA_Fast"] < data["EMA_Medium"] < data["EMA_Slow"]:
                ema_trend = -1.0
                if strong_trend and np.sign(macd_signal) < 0:  # Only boost if MACD aligns
                    ema_trend *= 1.2
            elif data["Close"] > data["EMA_Fast"] > data["EMA_Medium"] > data["EMA_Slow"]:
                ema_trend = 1.0
                if strong_trend and np.sign(macd_signal) > 0:  # Only boost if MACD aligns
                    ema_trend *= 1.2
            else:
                ema_diff = data["Close"] - data["EMA_Fast"]
                ema_trend = ema_diff / (data["ATR"] * 2)
                # Reduce trend influence when MACD disagrees
                if np.sign(ema_trend) != np.sign(macd_signal):
                    ema_trend *= 0.7
            ema_trend = max(min(ema_trend, 1), -1)

            # Volatility component with trend consideration
            volatility_signal = min(data["ATR"] / data["Close"] * 3.2, 0.35)
            if strong_trend:
                volatility_signal *= 0.7  # Reduce volatility impact in strong trends
            elif weak_trend:
                volatility_signal *= 1.3  # Increase volatility awareness in weak trends

            # Get prediction if available
            prediction = data.get("Prediction", 0)

            # Calculate weighted signal with enhanced trend quality influence
            weighted_signal = (
                price_momentum * weights["EMA_Fast"] * (1 + trend_quality * 0.2) +  # Increased from 0.1
                macd_signal * weights["MACD"] * (1 + trend_quality * 0.15) +        # Added trend quality factor
                ema_trend * weights["EMA_Medium"] * (1 + trend_quality * 0.25) +    # Increased from 0.15
                volatility_signal * weights["ATR"] * (1 - trend_quality * 0.3) +    # Increased from 0.2
                prediction * weights["Prediction"]                                   # Prediction component
            )

            # Enhanced signal alignment boost
            components = [price_momentum, macd_signal, ema_trend]
            if all(abs(c) > 0.2 for c in components) and all(np.sign(c) == np.sign(components[0]) for c in components):
                alignment_boost = 1.2 * (1 + trend_quality * 0.2)  # Increased from 1.1 and 0.1
                # Add prediction to alignment check if it exists and is significant
                if abs(prediction) > 0.2 and np.sign(prediction) == np.sign(components[0]):
                    alignment_boost *= 1.1  # Additional 10% boost for aligned prediction
                weighted_signal *= alignment_boost

            final_signal = max(min(weighted_signal, 1.0), -1.0)

            # Enhanced logging
            logger.info("Signal Analysis:")
            logger.info(f"Trend Quality: {trend_quality:.3f} ({'Strong' if strong_trend else 'Weak' if weak_trend else 'Normal'})")
            logger.info(f"Price Momentum: {price_momentum:.3f} (Weight: {weights['EMA_Fast']:.2f})")
            logger.info(f"MACD Signal: {macd_signal:.3f} (Weight: {weights['MACD']:.2f})")
            logger.info(f"EMA Trend: {ema_trend:.3f} (Weight: {weights['EMA_Medium']:.2f})")
            logger.info(f"Volatility: {volatility_signal:.3f} (Weight: {weights['ATR']:.2f})")
            logger.info(f"Prediction: {prediction:.3f} (Weight: {weights['Prediction']:.2f})")
            logger.info(f"Final Signal: {final_signal:.3f}")

            return final_signal

        except Exception as e:
            logger.error(f"Error in calculate_weighted_signal: {e}")
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

    def _adjust_prediction_weights(self, df_1m, df_3m, weights, logger: LoggerType):
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

            logger.info("Adjusted weights due to optimal conditions:")
            logger.info(f"Volatility factor: {volatility_factor:.3f}")
            logger.info(f"EMA pack factor: {ema_pack_factor:.3f}")
            logger.info(f"Weight increase: {increase:.3f}")

        return adjusted_weights

    def _calculate_trend_quality(self, df_3m: pd.DataFrame, df_1m: pd.DataFrame, logger: LoggerType) -> float:
        """Calculate trend quality based on EMA separations and trend structure, optimized for DCA grid trading."""
        try:
            # Use completed candles
            close = df_3m['close'].iloc[-2]
            ema_fast = df_3m['ema_fast'].iloc[-2]
            ema_medium = df_3m['ema_medium'].iloc[-2]
            ema_slow = df_3m['ema_slow'].iloc[-2]

            # Calculate EMA separations using ATR for normalization (use completed candle)
            atr = df_1m['atr'].iloc[-2]
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
            logger.error(f"Error calculating trend quality: {e}")
            return 0.0

