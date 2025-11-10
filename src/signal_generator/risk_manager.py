"""
Risk Manager
Calcula los parámetros de riesgo para las señales de trading.
"""

import pandas as pd
from typing import Dict, List, Optional
from loguru import logger

from config import config
from ai_engine.market_analyzer import MarketAnalysis

class RiskManager:
    """Gestiona los parámetros de riesgo para las señales de trading."""

    def __init__(self):
        """Inicializa el Risk Manager."""
        if config.enable_dynamic_risk:
            logger.info("Risk Manager inicializado con gestión de riesgo DINÁMICA basada en ATR.")
            logger.info(f"Multiplicadores: SL={config.stop_loss_atr_multiplier}*ATR, TP1={config.take_profit_1_atr_multiplier}*ATR, TP2={config.take_profit_2_atr_multiplier}*ATR")
        else:
            logger.info("Risk Manager inicializado con gestión de riesgo FIJA.")
            logger.info(f"SL Fijo={config.fixed_stop_loss_points} puntos, TP1 Fijo={config.fixed_take_profit_1_points} puntos, TP2 Fijo={config.fixed_take_profit_2_points} puntos")

        if config.enable_dynamic_lot_size:
            logger.info(f"Lotaje DINÁMICO activado: Min={config.min_lot_size}, Max={config.max_lot_size}")
        else:
            logger.info(f"Lotaje FIJO activado: {config.mt5_lot_size} lotes")

    def calculate_dynamic_lot_size(self, confidence: float) -> float:
        """
        Calcula el tamaño del lote dinámicamente basado en la confianza de la señal.
        """
        if not config.enable_dynamic_lot_size:
            return config.mt5_lot_size

        min_conf = config.confidence_threshold
        max_conf = 1.0
        
        # Escalar la confianza de su rango original (min_conf a 1.0) a un rango de 0 a 1
        if confidence < min_conf:
            confidence = min_conf
        
        scaled_confidence = (confidence - min_conf) / (max_conf - min_conf)
        
        # Calcular el tamaño del lote
        lot_range = config.max_lot_size - config.min_lot_size
        dynamic_lot = config.min_lot_size + (scaled_confidence * lot_range)
        
        # Asegurarse de que el lote esté dentro de los límites y redondear a 2 decimales
        return round(max(config.min_lot_size, min(dynamic_lot, config.max_lot_size)), 2)

    def _calculate_fixed_risk_parameters(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        confidence: float
    ) -> Optional[Dict]:
        """
        Calcula el stop loss y take profit usando valores fijos en puntos.

        Args:
            symbol: Par de trading.
            signal_type: BUY o SELL.
            entry_price: Precio de entrada.
            confidence: Confianza de la señal (para lotaje dinámico si está habilitado).

        Returns:
            Diccionario con los parámetros de riesgo fijos.
        """
        try:
            # Para índices sintéticos: 1 punto = 1.0 en el precio
            point_value = 1.0

            sl_distance = config.fixed_stop_loss_points * point_value
            tp1_distance = config.fixed_take_profit_1_points * point_value
            tp2_distance = config.fixed_take_profit_2_points * point_value

            if signal_type == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit_1 = entry_price + tp1_distance
                take_profit_2 = entry_price + tp2_distance
            elif signal_type == 'SELL':
                stop_loss = entry_price + sl_distance
                take_profit_1 = entry_price - tp1_distance
                take_profit_2 = entry_price - tp2_distance
            else:
                logger.warning(f"Tipo de señal desconocido: {signal_type}")
                return None

            if stop_loss <= 0 or take_profit_1 <= 0:
                logger.warning(f"{symbol}: SL ({stop_loss}) o TP1 ({take_profit_1}) inválidos calculados.")
                return None

            risk_amount = abs(entry_price - stop_loss)
            reward_amount_1 = abs(take_profit_1 - entry_price)
            risk_reward_ratio_1 = reward_amount_1 / risk_amount if risk_amount > 0 else 0

            logger.info(f"Parámetros de riesgo FIJOS para {symbol} ({signal_type}):")
            lot_size = self.calculate_dynamic_lot_size(confidence)
            logger.info(f"Lotaje: {lot_size} {'(Dinámico)' if config.enable_dynamic_lot_size else '(Fijo)'}")
            logger.info(f"SL={stop_loss:.5f}, TP1={take_profit_1:.5f}, TP2={take_profit_2:.5f}, RR1={risk_reward_ratio_1:.2f}")

            return {
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit_levels': [float(take_profit_1), float(take_profit_2)],
                'risk_amount': float(risk_amount),
                'risk_reward_ratio': float(risk_reward_ratio_1),
                'lot_size': lot_size,
                'atr_at_signal': None  # No hay ATR en modo fijo
            }

        except Exception as e:
            logger.error(f"Error al calcular los parámetros de riesgo fijos para {symbol}: {e}")
            return None

    def calculate_risk_parameters(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        analysis: MarketAnalysis,
        confidence: float,
        full_market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Calcula la entrada, el stop loss y el take profit.
        Usa modo dinámico (ATR) o fijo según configuración.

        Args:
            symbol: Par de trading.
            signal_type: BUY o SELL.
            entry_price: Precio de entrada.
            analysis: Análisis de mercado que contiene el valor del ATR.
            confidence: Confianza de la señal.
            full_market_data: DataFrame completo con datos de mercado para buscar ATR válidos.

        Returns:
            Diccionario con los parámetros de riesgo.
        """
        # Si el modo dinámico está desactivado, usar valores fijos
        if not config.enable_dynamic_risk:
            return self._calculate_fixed_risk_parameters(symbol, signal_type, entry_price, confidence)
        try:
            # Intenta obtener el ATR más reciente. Si no es válido, busca el último valor válido.
            atr = analysis.indicators.get('atr')
            if atr is None or pd.isna(atr) or atr <= 0:
                logger.warning(f"ATR más reciente para {symbol} es inválido. Buscando el último valor válido...")
                # Busca hacia atrás en la columna 'atr' del DataFrame completo
                last_valid_atr = full_market_data['atr'].dropna().iloc[-1]
                if last_valid_atr and last_valid_atr > 0:
                    atr = last_valid_atr
                    logger.info(f"Usando el último ATR válido encontrado: {atr:.5f}")
                else:
                    logger.error(f"No se encontró ningún valor ATR válido para {symbol}. No se puede calcular el riesgo.")
                    return None

            sl_distance = atr * config.stop_loss_atr_multiplier
            tp1_distance = atr * config.take_profit_1_atr_multiplier
            tp2_distance = atr * config.take_profit_2_atr_multiplier

            if signal_type == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit_1 = entry_price + tp1_distance
                take_profit_2 = entry_price + tp2_distance
            elif signal_type == 'SELL':
                stop_loss = entry_price + sl_distance
                take_profit_1 = entry_price - tp1_distance
                take_profit_2 = entry_price - tp2_distance
            else:
                logger.warning(f"Tipo de señal desconocido: {signal_type}")
                return None

            if stop_loss <= 0 or take_profit_1 <= 0:
                logger.warning(f"{symbol}: SL ({stop_loss}) o TP1 ({take_profit_1}) inválidos calculados.")
                return None

            risk_amount = abs(entry_price - stop_loss)
            reward_amount_1 = abs(take_profit_1 - entry_price)
            risk_reward_ratio_1 = reward_amount_1 / risk_amount if risk_amount > 0 else 0

            logger.info(f"Parámetros de riesgo dinámico para {symbol} ({signal_type}) con ATR={atr:.5f}:")
            lot_size = self.calculate_dynamic_lot_size(confidence)
            logger.info(f"Lotaje Dinámico Calculado: {lot_size} (Confianza: {confidence:.2f})")

            logger.info(f"SL={stop_loss:.5f}, TP1={take_profit_1:.5f}, TP2={take_profit_2:.5f}, RR1={risk_reward_ratio_1:.2f}")

            return {
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit_levels': [float(take_profit_1), float(take_profit_2)],
                'risk_amount': float(risk_amount),
                'risk_reward_ratio': float(risk_reward_ratio_1),
                'lot_size': lot_size,
                'atr_at_signal': atr # Guardamos el ATR para la gestión dinámica posterior
            }

        except Exception as e:
            logger.error(f"Error al calcular los parámetros de riesgo para {symbol}: {e}")
            return None
