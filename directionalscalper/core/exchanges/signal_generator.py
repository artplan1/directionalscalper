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
            market_regime = self._detect_market_regime(df_1m, symbol)
            # logging.info(f"[{symbol}] Detecting market regime - atr_pct: {df_1m['atr_pct'].iloc[-1]}")

            logging.info(f"[{symbol}] Market Regime: {market_regime}")
            latest_atr_pct = df_1m['atr_pct'].iloc[-1]
            logging.info(f"[{symbol}] ATR %: {latest_atr_pct:.4f}")

            # Prepare data for weighted signal calculation
            signal_data = {
                "Close": df_3m['close'].iloc[-1],
                'MACD': df_3m['macd'].iloc[-1],
                'MACD_Signal': df_3m['macd_signal'].iloc[-1],
                'EMA_Fast': df_3m['ema_fast'].iloc[-1],
                'EMA_Medium': df_3m['ema_medium'].iloc[-1],
                'ATR': df_1m['atr'].iloc[-1],
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

            # Base weights for minute-scale crypto scalping
            base_weights = {
                "MACD": 0.35,      # Reduced trend dependency
                "EMA_Fast": 0.20,  # Less trend weight
                "EMA_Medium": 0.10,# Minimal baseline trend
                "ATR": 0.25,       # Increased volatility weight
                "Prediction": 0.10 # Increased ML influence
            }

            # Get market regime and volatility metrics
            # market_regime = self.detect_market_regime(df_1m)
            current_volatility = df_1m['atr_pct'].iloc[-1]
            price_momentum = df_1m['close'].pct_change(3).iloc[-1]  # 3-minute momentum

            # Adjust weights based on market regime
            weights = base_weights.copy()

            if market_regime == "volatile":
                if current_volatility > 3.0:  # High volatility regime
                    weights["MACD"] *= 1.5     # Strong emphasis on momentum
                    weights["EMA_Fast"] *= 1.2 # Quick trend confirmation
                    weights["EMA_Medium"] *= 0.5  # Reduce longer-term influence
                    weights["ATR"] *= 1.3      # Higher volatility awareness
                    weights["Prediction"] *= 0.4  # Reduce ML in high volatility

                    if abs(price_momentum) > 0.005:  # Strong 3-min momentum (0.5%)
                        weights["MACD"] *= 1.2       # Further increase momentum weight
                        weights["EMA_Fast"] *= 1.3   # Stronger trend following
                else:  # Moderate volatility
                    weights["MACD"] *= 1.3
                    weights["EMA_Fast"] *= 1.1
                    weights["ATR"] *= 1.2

            elif market_regime == "ranging":
                # Ranging market - focus on reversals
                weights["MACD"] *= 1.2      # Catch momentum shifts
                weights["EMA_Fast"] *= 0.8  # Reduce trend following
                weights["EMA_Medium"] *= 0.6 # Minimal baseline trend influence
                weights["ATR"] *= 1.4       # Watch for breakouts
                weights["Prediction"] *= 1.2 # Increase ML for range trading

                if current_volatility < 1.0:  # Low volatility ranging
                    weights["MACD"] *= 0.8    # Reduce momentum sensitivity
                    weights["ATR"] *= 1.5     # Higher breakout sensitivity

            elif market_regime == "trending":
                # Clear trend - follow the momentum
                weights["MACD"] *= 1.3
                weights["EMA_Fast"] *= 1.4
                weights["EMA_Medium"] *= 1.2
                weights["ATR"] *= 0.8
                weights["Prediction"] *= 0.7

                if abs(price_momentum) > 0.003:  # Strong trend momentum
                    weights["MACD"] *= 1.2
                    weights["EMA_Fast"] *= 1.3

            else:  # normal regime
                # Balanced weights with slight momentum bias
                if current_volatility > 1.5:  # Above average volatility
                    weights["MACD"] *= 1.2
                    weights["ATR"] *= 1.1
                elif current_volatility < 0.5:  # Below average volatility
                    weights["MACD"] *= 0.9
                    weights["ATR"] *= 1.3
                    weights["Prediction"] *= 1.2

            # Quick momentum adjustment for all regimes
            if abs(price_momentum) > 0.008:  # Strong momentum (0.8% in 3 mins)
                weights["MACD"] *= 1.3
                weights["EMA_Fast"] *= 1.2
                weights["EMA_Medium"] *= 0.7

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            logging.info(f"""[{symbol}] Minute-Scale Regime Weights:
                [{symbol}] Market Regime: {market_regime}
                [{symbol}] Current Volatility: {current_volatility:.3f}%
                [{symbol}] 3-min Momentum: {price_momentum:.3f}%
                [{symbol}] Weights: {json.dumps(weights)}
            """)

            # Calculate weighted signal with adjusted weights
            weighted_signal = self._calculate_weighted_signal(signal_data, symbol, weights)

            logging.info(f"[{symbol}] Weighted Signal: {weighted_signal:.3f}")

            # More aggressive thresholds for minute-scale trading
            if weighted_signal > 0.15:    # Reduced threshold for faster entries
                new_signal= "long"
            elif weighted_signal < -0.15:  # Reduced threshold for faster entries
                new_signal= "short"
            else:
                new_signal= "neutral"

            # Signal buffer logic with proper buffering
            current_time = time.time()
            if symbol in self.exchange.last_signal:
                last_signal = self.exchange.last_signal[symbol]
                time_since_last = current_time - self.exchange.last_signal_time[symbol]

                # Determine buffer time based on regime
                buffer_time = 8  # Base buffer time
                if market_regime == "volatile":
                    buffer_time = 5  # Shorter buffer in volatile markets
                elif market_regime == "trending":
                    buffer_time = 12  # Longer buffer in trending markets

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
        is_uptrend = (close > ema_fast and ema_fast > ema_medium) or (macd > macd_signal)
        is_downtrend = (close < ema_fast and ema_fast < ema_medium) or (macd < macd_signal)
        return is_uptrend, is_downtrend

    def _detect_market_regime(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Detect market regime optimized for minute-scale crypto trading.
        Returns: 'volatile', 'trending', 'ranging', or 'normal'
        """
        try:
            # Volatility metrics
            atr_pct = df['atr_pct'].iloc[-1]  # Current ATR as percentage
            recent_volatility = df['atr_pct'].rolling(5).mean().iloc[-1]  # 5-minute average
            baseline_volatility = df['atr_pct'].rolling(20).mean().iloc[-1]  # 20-minute baseline

            # Momentum and trend metrics
            price_changes = df['close'].pct_change()
            recent_momentum = price_changes.rolling(5).sum().iloc[-1]  # 5-minute momentum

            # Trend strength indicators
            ema_fast = df['close'].ewm(span=8, adjust=False).mean()
            ema_medium = df['close'].ewm(span=21, adjust=False).mean()

            # Calculate trend consistency
            price_above_fast = (df['close'] > ema_fast).rolling(10).mean().iloc[-1]
            fast_above_medium = (ema_fast > ema_medium).rolling(10).mean().iloc[-1]

            # Direction changes (ranging detection)
            direction_changes = (price_changes.rolling(10)
                               .apply(lambda x: ((x > 0) != (x.shift(1) > 0)).sum())
                               .iloc[-1])

            # Log regime detection metrics
            logging.info(f"""Market Regime Metrics:
                Current ATR%: {atr_pct:.4f}
                Recent Volatility (5m): {recent_volatility:.4f}
                Baseline Volatility (20m): {baseline_volatility:.4f}
                Recent Momentum: {recent_momentum:.4f}
                Direction Changes: {direction_changes}
                Price Above Fast EMA: {price_above_fast:.4f}
                Fast Above Medium EMA: {fast_above_medium:.4f}
            """)

            # Detect regime with improved volatility analysis
            if (recent_volatility > baseline_volatility * 1.2 and atr_pct > recent_volatility) or \
               abs(recent_momentum) > 0.015:  # More sensitive volatile conditions
                return 'volatile'

            elif (price_above_fast > 0.8 and fast_above_medium > 0.8 and recent_momentum > 0 and \
                  recent_volatility > baseline_volatility * 0.8) or \
                 (price_above_fast < 0.2 and fast_above_medium < 0.2 and recent_momentum < 0 and \
                  recent_volatility > baseline_volatility * 0.8):
                # Strong trend conditions with sufficient volatility
                return 'trending'

            elif direction_changes >= 4 and recent_volatility < baseline_volatility * 0.8:
                # Ranging conditions with lower recent volatility
                return 'ranging'

            else:
                return 'normal'

        except Exception as e:
            logging.error(f"Error in detect_market_regime: {e}")
            return 'normal'  # Default to normal regime on error

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

    def _calculate_weighted_signal(self, data: Dict[str, Any], symbol: str, weights: Dict[str, float]) -> float:
        """
        Calculate weighted trading signal optimized for minute-scale crypto scalping.
        Returns a value between -1 (strong sell) and 1 (strong buy).
        """
        try:
            # 1. Price Momentum Component
            price_momentum = (data["Close"] - data["EMA_Fast"]) / data["EMA_Fast"]

            # 2. MACD Component
            macd_signal = (data["MACD"] - data["MACD_Signal"]) / abs(data["MACD_Signal"])
            macd_signal = max(min(macd_signal, 1), -1)  # Bound between -1 and 1

            # 3. EMA Trend Component
            if data["Close"] > data["EMA_Fast"] > data["EMA_Medium"]:
                ema_trend = min((data["Close"] - data["EMA_Fast"]) / data["EMA_Fast"], 1)
            elif data["Close"] < data["EMA_Fast"] < data["EMA_Medium"]:
                ema_trend = max((data["Close"] - data["EMA_Fast"]) / data["EMA_Fast"], -1)
            else:
                ema_trend = (data["Close"] - data["EMA_Medium"]) / data["EMA_Medium"]

            # 4. Volatility Component
            volatility_signal = min(data["ATR"] / data["Close"], 0.1)

            # 5. Calculate weighted signal
            weighted_signal = (
                price_momentum * weights["EMA_Fast"] +
                macd_signal * weights["MACD"] +
                ema_trend * weights["EMA_Medium"] +
                volatility_signal * weights["ATR"]
            )

            # 6. Add prediction if available
            if "Prediction" in data and "Max_Prediction" in data and data["Max_Prediction"] != 0:
                prediction = (data["Prediction"] / data["Max_Prediction"]) * 2 - 1
                weighted_signal += prediction * weights["Prediction"]

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


