"""
Technical Indicators
Comprehensive technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta
from loguru import logger


class TechnicalIndicators:
    """Calculate technical indicators for market analysis"""

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        if df.empty or len(df) < 50:
            return df

        result = df.copy()

        try:
            # Trend Indicators
            result = self._add_trend_indicators(result)

            # Momentum Indicators
            result = self._add_momentum_indicators(result)

            # Volatility Indicators
            result = self._add_volatility_indicators(result)

            # Volume Indicators
            result = self._add_volume_indicators(result)

            # Custom Indicators
            result = self._add_custom_indicators(result)

            logger.debug(f"Calculated {len(result.columns) - 6} technical indicators")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return result

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators (highly optimized - only most effective)"""

        # Moving Averages - Keep only essential ones
        # SMA 50 for medium-term trend identification
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

        # EMAs for short-term trend (faster reaction than SMA)
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)

        # MACD - Essential trend-following indicator
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # ADX - Trend strength (essential for filters)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (highly optimized - only most effective)"""

        # RSI - Most important momentum indicator (industry standard)
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

        # Stochastic Oscillator - Complements RSI for overbought/oversold
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators (highly optimized - only most effective)"""
        from src.config import config

        # ATR - Critical for risk management and dynamic SL/TP
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=config.atr_period)

        # Bollinger Bands - Essential volatility indicator
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators (highly optimized - only most effective)"""

        # On-Balance Volume - Best volume momentum indicator
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Volume Weighted Average Price - Essential for institutional levels
        df['vwap'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )

        return df

    def _add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators (highly optimized - only most effective)"""

        # High-Low spread - Candle range relative to price
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Close position in range - Where price closed within the candle
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Price vs SMA50 - Deviation from medium-term trend
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']

        # Trend strength - How strong is the trend relative to volatility
        df['trend_strength'] = abs(df['close'] - df['sma_50']) / df['atr']

        return df

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect candlestick patterns

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary of detected patterns
        """
        if len(df) < 3:
            return {}

        patterns = {}

        try:
            # Get last few candles
            current = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) > 2 else prev

            # Doji
            body = abs(current['close'] - current['open'])
            range_val = current['high'] - current['low']
            patterns['doji'] = body < (range_val * 0.1) if range_val > 0 else False

            # Hammer
            lower_wick = min(current['open'], current['close']) - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])
            patterns['hammer'] = (lower_wick > body * 2) and (upper_wick < body * 0.5)

            # Shooting Star
            patterns['shooting_star'] = (upper_wick > body * 2) and (lower_wick < body * 0.5)

            # Engulfing patterns
            if current['close'] > current['open'] and prev['close'] < prev['open']:
                patterns['bullish_engulfing'] = (
                    current['open'] <= prev['close'] and current['close'] >= prev['open']
                )
            else:
                patterns['bullish_engulfing'] = False

            if current['close'] < current['open'] and prev['close'] > prev['open']:
                patterns['bearish_engulfing'] = (
                    current['open'] >= prev['close'] and current['close'] <= prev['open']
                )
            else:
                patterns['bearish_engulfing'] = False

            # Morning Star / Evening Star
            if len(df) > 2:
                patterns['morning_star'] = (
                    prev2['close'] < prev2['open'] and  # First bearish
                    abs(prev['close'] - prev['open']) < body * 0.3 and  # Small body
                    current['close'] > current['open'] and  # Last bullish
                    current['close'] > (prev2['open'] + prev2['close']) / 2
                )

                patterns['evening_star'] = (
                    prev2['close'] > prev2['open'] and  # First bullish
                    abs(prev['close'] - prev['open']) < body * 0.3 and  # Small body
                    current['close'] < current['open'] and  # Last bearish
                    current['close'] < (prev2['open'] + prev2['close']) / 2
                )

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels

        Args:
            df: DataFrame with OHLCV data
            window: Lookback window

        Returns:
            Dictionary with support and resistance levels
        """
        if len(df) < window:
            return {'support': 0, 'resistance': 0}

        try:
            recent = df.tail(window)

            support = recent['low'].min()
            resistance = recent['high'].max()

            return {
                'support': float(support),
                'resistance': float(resistance),
                'mid_point': float((support + resistance) / 2)
            }

        except Exception as e:
            logger.error(f"Error calculating S/R levels: {e}")
            return {'support': 0, 'resistance': 0}
