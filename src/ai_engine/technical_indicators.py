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
        Calculate support and resistance levels using pivot points and swing analysis

        Args:
            df: DataFrame with OHLCV data
            window: Lookback window

        Returns:
            Dictionary with support and resistance levels
        """
        if len(df) < window:
            return {'support': 0, 'resistance': 0}

        try:
            recent = df.tail(window * 2)  # Usar ventana más amplia para mejor detección

            # Método 1: Pivot Points (más preciso que simple min/max)
            # Detectar swing lows (soportes potenciales)
            swing_lows = []
            for i in range(5, len(recent) - 5):
                current_low = recent.iloc[i]['low']
                # Verificar si es un swing low (mínimo local)
                is_swing_low = True
                for j in range(1, 6):
                    if current_low >= recent.iloc[i-j]['low'] or current_low >= recent.iloc[i+j]['low']:
                        is_swing_low = False
                        break
                if is_swing_low:
                    swing_lows.append(current_low)

            # Detectar swing highs (resistencias potenciales)
            swing_highs = []
            for i in range(5, len(recent) - 5):
                current_high = recent.iloc[i]['high']
                # Verificar si es un swing high (máximo local)
                is_swing_high = True
                for j in range(1, 6):
                    if current_high <= recent.iloc[i-j]['high'] or current_high <= recent.iloc[i+j]['high']:
                        is_swing_high = False
                        break
                if is_swing_high:
                    swing_highs.append(current_high)

            # Usar swing points si están disponibles, de lo contrario fallback a min/max
            if swing_lows:
                support = max(swing_lows[-3:]) if len(swing_lows) >= 3 else max(swing_lows)
            else:
                support = recent['low'].min()

            if swing_highs:
                resistance = min(swing_highs[-3:]) if len(swing_highs) >= 3 else min(swing_highs)
            else:
                resistance = recent['high'].max()

            return {
                'support': float(support),
                'resistance': float(resistance),
                'mid_point': float((support + resistance) / 2)
            }

        except Exception as e:
            logger.error(f"Error calculating S/R levels: {e}")
            return {'support': 0, 'resistance': 0}

    @staticmethod
    def detect_price_reaction_at_level(
        df: pd.DataFrame,
        level: float,
        level_type: str,  # 'support' or 'resistance'
        tolerance_percent: float = 0.3
    ) -> Dict[str, any]:
        """
        Detecta si el precio ha reaccionado a un nivel de soporte o resistencia

        Una reacción válida requiere:
        1. El precio tocó el nivel (dentro de tolerancia)
        2. Patrón de vela de rechazo (hammer, engulfing, etc.)
        3. Confirmación posterior (el precio se aleja del nivel)

        Args:
            df: DataFrame with OHLCV data
            level: Support or resistance level to check
            level_type: 'support' or 'resistance'
            tolerance_percent: Tolerance for level touch (default 0.3%)

        Returns:
            Dict with reaction details:
            {
                'has_reaction': bool,
                'confidence': float,
                'pattern': str,
                'candles_ago': int,
                'reason': str
            }
        """
        if df.empty or len(df) < 5 or level == 0:
            return {
                'has_reaction': False,
                'confidence': 0.0,
                'pattern': None,
                'candles_ago': 0,
                'reason': 'Insufficient data or invalid level'
            }

        try:
            # Analizar las últimas 10 velas para detectar reacción reciente
            lookback = min(10, len(df))
            recent_candles = df.tail(lookback)

            # Calcular rango de tolerancia para el nivel
            tolerance = level * (tolerance_percent / 100)
            level_low = level - tolerance
            level_high = level + tolerance

            # Buscar toque del nivel
            for i in range(len(recent_candles) - 1, -1, -1):
                candle = recent_candles.iloc[i]
                candles_ago = len(recent_candles) - 1 - i

                # Verificar si la vela tocó el nivel
                touched = False
                if level_type == 'support':
                    # Para soporte, verificar si el low tocó el nivel
                    if level_low <= candle['low'] <= level_high:
                        touched = True
                elif level_type == 'resistance':
                    # Para resistencia, verificar si el high tocó el nivel
                    if level_low <= candle['high'] <= level_high:
                        touched = True

                if touched:
                    # Verificar patrón de rechazo
                    pattern_result = TechnicalIndicators._check_rejection_pattern(
                        recent_candles, i, level_type
                    )

                    if pattern_result['has_pattern']:
                        # Verificar confirmación (velas posteriores se alejaron del nivel)
                        confirmation = TechnicalIndicators._check_reaction_confirmation(
                            recent_candles, i, level, level_type
                        )

                        if confirmation['confirmed']:
                            return {
                                'has_reaction': True,
                                'confidence': pattern_result['confidence'] * confirmation['strength'],
                                'pattern': pattern_result['pattern_name'],
                                'candles_ago': candles_ago,
                                'reason': f"{pattern_result['pattern_name']} at {level_type} ({candles_ago} candles ago), confirmation: {confirmation['strength']:.2f}"
                            }

            return {
                'has_reaction': False,
                'confidence': 0.0,
                'pattern': None,
                'candles_ago': 0,
                'reason': f'No reaction detected at {level_type} level {level:.5f}'
            }

        except Exception as e:
            logger.error(f"Error detecting price reaction: {e}")
            return {
                'has_reaction': False,
                'confidence': 0.0,
                'pattern': None,
                'candles_ago': 0,
                'reason': f'Error: {str(e)}'
            }

    @staticmethod
    def _check_rejection_pattern(df: pd.DataFrame, candle_idx: int, level_type: str) -> Dict[str, any]:
        """
        Verifica si hay un patrón de rechazo en la vela especificada

        Patrones de rechazo alcistas (en soporte):
        - Hammer: mecha inferior larga, cuerpo pequeño arriba
        - Bullish Engulfing: vela alcista que envuelve a bajista previa
        - Morning Star: patrón de 3 velas de reversión alcista

        Patrones de rechazo bajistas (en resistencia):
        - Shooting Star: mecha superior larga, cuerpo pequeño abajo
        - Bearish Engulfing: vela bajista que envuelve a alcista previa
        - Evening Star: patrón de 3 velas de reversión bajista
        """
        if candle_idx >= len(df) - 1:
            return {'has_pattern': False, 'confidence': 0.0, 'pattern_name': None}

        try:
            current = df.iloc[candle_idx]
            next_candle = df.iloc[candle_idx + 1] if candle_idx + 1 < len(df) else current
            prev = df.iloc[candle_idx - 1] if candle_idx > 0 else current

            body = abs(current['close'] - current['open'])
            range_val = current['high'] - current['low']

            if range_val == 0:
                return {'has_pattern': False, 'confidence': 0.0, 'pattern_name': None}

            if level_type == 'support':
                # Patrones alcistas (rechazo de soporte)

                # Hammer: mecha inferior >= 1.5x cuerpo, mecha superior pequeña (OPCIÓN 2: más flexible)
                lower_wick = min(current['open'], current['close']) - current['low']
                upper_wick = current['high'] - max(current['open'], current['close'])

                if lower_wick >= body * 1.5 and upper_wick < body * 0.5:
                    # Verificar confirmación con siguiente vela alcista
                    confirmation = next_candle['close'] > next_candle['open']
                    confidence = 0.85 if confirmation else 0.65
                    return {
                        'has_pattern': True,
                        'confidence': confidence,
                        'pattern_name': 'Hammer (Support Rejection)'
                    }

                # Bullish Engulfing
                if (current['close'] > current['open'] and
                    prev['close'] < prev['open'] and
                    current['open'] <= prev['close'] and
                    current['close'] >= prev['open']):
                    return {
                        'has_pattern': True,
                        'confidence': 0.80,
                        'pattern_name': 'Bullish Engulfing'
                    }

                # Vela alcista fuerte (cuerpo > 50% del rango) (OPCIÓN 2: más flexible)
                if current['close'] > current['open'] and body / range_val > 0.5:
                    return {
                        'has_pattern': True,
                        'confidence': 0.60,
                        'pattern_name': 'Strong Bullish Candle'
                    }

            elif level_type == 'resistance':
                # Patrones bajistas (rechazo de resistencia)

                # Shooting Star: mecha superior >= 1.5x cuerpo, mecha inferior pequeña (OPCIÓN 2: más flexible)
                upper_wick = current['high'] - max(current['open'], current['close'])
                lower_wick = min(current['open'], current['close']) - current['low']

                if upper_wick >= body * 1.5 and lower_wick < body * 0.5:
                    # Verificar confirmación con siguiente vela bajista
                    confirmation = next_candle['close'] < next_candle['open']
                    confidence = 0.85 if confirmation else 0.65
                    return {
                        'has_pattern': True,
                        'confidence': confidence,
                        'pattern_name': 'Shooting Star (Resistance Rejection)'
                    }

                # Bearish Engulfing
                if (current['close'] < current['open'] and
                    prev['close'] > prev['open'] and
                    current['open'] >= prev['close'] and
                    current['close'] <= prev['open']):
                    return {
                        'has_pattern': True,
                        'confidence': 0.80,
                        'pattern_name': 'Bearish Engulfing'
                    }

                # Vela bajista fuerte (cuerpo > 50% del rango) (OPCIÓN 2: más flexible)
                if current['close'] < current['open'] and body / range_val > 0.5:
                    return {
                        'has_pattern': True,
                        'confidence': 0.60,
                        'pattern_name': 'Strong Bearish Candle'
                    }

            return {'has_pattern': False, 'confidence': 0.0, 'pattern_name': None}

        except Exception as e:
            logger.error(f"Error checking rejection pattern: {e}")
            return {'has_pattern': False, 'confidence': 0.0, 'pattern_name': None}

    @staticmethod
    def _check_reaction_confirmation(
        df: pd.DataFrame,
        reaction_idx: int,
        level: float,
        level_type: str
    ) -> Dict[str, any]:
        """
        Verifica si hay confirmación de la reacción
        (las velas posteriores se alejaron del nivel)
        """
        try:
            # Necesitamos al menos 1-2 velas después de la reacción
            candles_after = len(df) - reaction_idx - 1

            if candles_after < 1:
                return {'confirmed': False, 'strength': 0.0}

            # Analizar movimiento después de la reacción
            reaction_candle = df.iloc[reaction_idx]
            latest_candle = df.iloc[-1]

            if level_type == 'support':
                # Para soporte, el precio debe haber subido
                price_movement = latest_candle['close'] - reaction_candle['low']
                price_movement_percent = (price_movement / level) * 100

                # Confirmar si el precio se movió al menos 0.15% hacia arriba (OPCIÓN 2: más flexible)
                if price_movement_percent >= 0.15:
                    # Strength basado en qué tan fuerte fue el movimiento
                    strength = min(1.0, price_movement_percent / 1.0)  # Max 1.0 at 1%
                    return {'confirmed': True, 'strength': strength}

            elif level_type == 'resistance':
                # Para resistencia, el precio debe haber bajado
                price_movement = reaction_candle['high'] - latest_candle['close']
                price_movement_percent = (price_movement / level) * 100

                # Confirmar si el precio se movió al menos 0.15% hacia abajo (OPCIÓN 2: más flexible)
                if price_movement_percent >= 0.15:
                    # Strength basado en qué tan fuerte fue el movimiento
                    strength = min(1.0, price_movement_percent / 1.0)  # Max 1.0 at 1%
                    return {'confirmed': True, 'strength': strength}

            return {'confirmed': False, 'strength': 0.0}

        except Exception as e:
            logger.error(f"Error checking reaction confirmation: {e}")
            return {'confirmed': False, 'strength': 0.0}
