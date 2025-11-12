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
        """Add trend indicators (optimized - reduced redundancy)"""

        # Simple Moving Averages (reduced from 4 to 2 - less correlation)
        # Removed: sma_7 (too noisy), sma_100 (redundant with sma_50)
        df['sma_25'] = ta.trend.sma_indicator(df['close'], window=25)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

        # Exponential Moving Averages (reduced from 4 to 2 - less correlation)
        # Removed: ema_50, ema_200 (redundant with SMAs)
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (optimized - reduced redundancy)"""

        # RSI (reduced from 3 to 1 - RSI14 is standard and sufficient)
        # Removed: rsi_6 (too noisy), rsi_21 (redundant)
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

        # ROC (Rate of Change)
        df['roc'] = ta.momentum.roc(df['close'], window=12)

        # TSI (True Strength Index)
        df['tsi'] = ta.momentum.tsi(df['close'])

        # Ultimate Oscillator
        df['uo'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])

        # Awesome Oscillator
        df['ao'] = ta.momentum.awesome_oscillator(df['high'], df['low'])

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        from src.config import config
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pband'] = bollinger.bollinger_pband()

        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=config.atr_period)

        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=config.atr_period)
        df['kc_high'] = keltner.keltner_channel_hband()
        df['kc_mid'] = keltner.keltner_channel_mband()
        df['kc_low'] = keltner.keltner_channel_lband()

        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_high'] = donchian.donchian_channel_hband()
        df['dc_mid'] = donchian.donchian_channel_mband()
        df['dc_low'] = donchian.donchian_channel_lband()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""

        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Chaikin Money Flow
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])

        # Force Index
        df['fi'] = ta.volume.force_index(df['close'], df['volume'])

        # Ease of Movement
        df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'])

        # Volume Price Trend
        df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])

        # Negative Volume Index
        df['nvi'] = ta.volume.negative_volume_index(df['close'], df['volume'])

        # Volume Weighted Average Price (VWAP)
        df['vwap'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )

        return df

    def _add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators"""

        # Price momentum
        df['momentum_1'] = df['close'].pct_change(1)
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # Volatility
        df['volatility_7'] = df['close'].pct_change().rolling(window=7).std()
        df['volatility_14'] = df['close'].pct_change().rolling(window=14).std()
        df['volatility_30'] = df['close'].pct_change().rolling(window=30).std()

        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Volume momentum
        df['volume_change'] = df['volume'].pct_change(1)

        # Price vs SMA
        df['price_vs_sma20'] = (df['close'] - df['sma_25']) / df['sma_25']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']

        # Trend strength
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
