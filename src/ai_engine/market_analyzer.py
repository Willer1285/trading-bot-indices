"""
Market Analyzer
Motor principal de IA para el análisis y la predicción del mercado.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime

from .feature_engineering import FeatureEngineer
from .technical_indicators import TechnicalIndicators
from .ai_models import EnsembleModel

# Mapeo de timeframes MT5 a nombres de modelos entrenados
TIMEFRAME_MAPPING = {
    '1m': 'M1',
    '5m': 'M5',
    '15m': 'M15',
    '30m': 'M30',
    '1h': 'H1',
    '4h': 'H4',
    '1d': 'D1',
    '1w': 'W1',
    '1M': 'MN1',
    # También soportar nombres en mayúsculas
    '1M': 'M1',
    '5M': 'M5',
    '15M': 'M15',
    '30M': 'M30',
    '1H': 'H1',
    '4H': 'H4',
    '1D': 'D1',
    '1W': 'W1'
}


class MarketAnalysis:
    """Contenedor para los resultados del análisis de mercado."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        prediction: int,
        confidence: float,
        probabilities: Dict[str, float],
        indicators: Dict,
        patterns: Dict[str, bool],
        support_resistance: Dict[str, float],
        timestamp: datetime
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.prediction = prediction  # 0=SELL, 1=HOLD, 2=BUY
        self.confidence = confidence
        self.probabilities = probabilities
        self.indicators = indicators
        self.patterns = patterns
        self.support_resistance = support_resistance
        self.timestamp = timestamp

    @property
    def signal(self) -> str:
        """Obtiene la señal como una cadena de texto."""
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return signal_map.get(self.prediction, 'HOLD')

    @property
    def is_actionable(self) -> bool:
        """Verifica si la señal es accionable (BUY o SELL)."""
        return self.prediction in [0, 2]

    def to_dict(self) -> Dict:
        """Convierte el objeto a un diccionario."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal': self.signal,
            'confidence': round(self.confidence, 4),
            'probabilities': {
                'sell': round(self.probabilities['sell'], 4),
                'hold': round(self.probabilities['hold'], 4),
                'buy': round(self.probabilities['buy'], 4)
            },
            'indicators': self.indicators,
            'patterns': self.patterns,
            'support_resistance': self.support_resistance,
            'timestamp': self.timestamp.isoformat()
        }


class MarketAnalyzer:
    """Analizador de mercado avanzado impulsado por IA que gestiona modelos especializados."""

    def __init__(self, enable_training: bool = False):
        self.feature_engineer = FeatureEngineer()
        self.technical_indicators = TechnicalIndicators()
        # Almacenará los modelos en un diccionario anidado: {symbol: {timeframe: model}}
        self.models: Dict[str, Dict[str, EnsembleModel]] = {}
        self.enable_training = enable_training
        self.is_trained = False
        logger.info("Market Analyzer inicializado para gestionar modelos especializados por símbolo/timeframe.")

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[MarketAnalysis]:
        """Realiza un análisis de mercado completo utilizando el modelo especializado correcto."""
        try:
            # Convertir timeframe MT5 al formato del modelo
            model_timeframe = TIMEFRAME_MAPPING.get(timeframe.lower(), timeframe.upper())

            # Construir la clave del modelo: "Symbol Timeframe" (ej: "GainX 1200 M1")
            model_key = f"{symbol}_{model_timeframe}"

            # Seleccionar el modelo correcto para el símbolo y timeframe
            model = self.models.get(symbol, {}).get(model_key)
            if not model or not model.is_fitted:
                logger.warning(f"No se encontró o no está entrenado un modelo para {symbol} [{timeframe} → {model_key}]. Saltando análisis.")
                logger.debug(f"Modelos disponibles para {symbol}: {list(self.models.get(symbol, {}).keys())}")
                return None

            if df.empty or len(df) < 100:
                logger.warning(f"Datos insuficientes para el análisis de {symbol} {timeframe}: {len(df)} velas")
                return None

            df_features = self.feature_engineer.extract_features(df)
            df_prepared = self.feature_engineer.prepare_for_model(df_features)

            # Seleccionar solo las columnas de características en las que el modelo fue entrenado
            feature_cols = self.feature_engineer.get_feature_importance_names()
            # Asegurarse de que todas las columnas de características existan en el dataframe
            feature_cols_exist = [col for col in feature_cols if col in df_prepared.columns]
            df_clean = df_prepared[feature_cols_exist]

            if df_clean.empty:
                logger.warning(f"No hay características válidas después de la preparación para {symbol} {timeframe}")
                return None

            # --- Lógica de Predicción Híbrida ---
            # El modelo primario ahora solo necesita la última fila para su lógica de patrones.
            latest_features_for_pattern = df_clean.tail(1)
            primary_prediction = model.base_models['pattern_based'].predict(latest_features_for_pattern)[0]

            if primary_prediction == 1:  # HOLD
                prediction = 1
                confidence = 0.6
                probabilities = np.array([0.2, 0.6, 0.2])
            else:
                # El meta-modelo (y el LSTM dentro de él) necesita el historial completo.
                # Pasamos `df_clean` que contiene el historial de 200 velas.
                # La predicción del meta-modelo es un array, tomamos la última que corresponde a la vela actual.
                all_probas = model.predict_proba(df_clean)
                latest_proba = all_probas[-1]
                
                meta_confidence = latest_proba[1]
                prediction = primary_prediction
                confidence = meta_confidence
                if prediction == 2: # BUY
                    probabilities = np.array([(1 - confidence) / 2, (1 - confidence) / 2, confidence])
                else: # SELL
                    probabilities = np.array([confidence, (1 - confidence) / 2, (1 - confidence) / 2])

            indicators = self._extract_indicator_summary(df)
            patterns = self.technical_indicators.detect_patterns(df)
            support_resistance = self.technical_indicators.calculate_support_resistance(df)

            temp_analysis = MarketAnalysis(
                symbol=symbol, timeframe=timeframe, prediction=int(prediction),
                confidence=float(confidence), probabilities={'sell': float(probabilities[0]), 'hold': float(probabilities[1]), 'buy': float(probabilities[2])},
                indicators=indicators, patterns=patterns, support_resistance=support_resistance,
                timestamp=datetime.utcnow()
            )

            logger.debug(f"Análisis para {symbol} {timeframe}: {temp_analysis.signal} (Confianza: {temp_analysis.confidence:.2%})")
            return temp_analysis

        except Exception as e:
            logger.error(f"Error analizando {symbol} {timeframe}: {e}")
            return None

    def analyze_multi_timeframe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, Optional[MarketAnalysis]]:
        """Analiza múltiples timeframes para un símbolo, usando el modelo correcto para cada uno."""
        results = {}
        for timeframe, df in data_dict.items():
            analysis = self.analyze(df, symbol, timeframe)
            results[timeframe] = analysis
        return results

    def _extract_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Extrae los valores clave de los indicadores."""
        latest = df.iloc[-1]
        summary = {'price': float(latest['close'])}
        indicators_to_extract = [
            'rsi_14', 'macd', 'macd_signal', 'adx', 'sma_25', 'sma_50',
            'ema_21', 'ema_50', 'bb_high', 'bb_low', 'atr', 'obv'
        ]
        for indicator in indicators_to_extract:
            if indicator in df.columns:
                value = latest.get(indicator)
                if pd.notna(value):
                    summary[indicator] = float(value)
        return summary

    def load_models(self, base_directory: str = "models"):
        """Escanea, carga y organiza todos los modelos entrenados por símbolo y timeframe."""
        logger.info(f"Buscando y cargando modelos especializados desde '{base_directory}'...")
        self.models = {}
        model_count = 0

        for symbol_dir in os.listdir(base_directory):
            symbol_path = os.path.join(base_directory, symbol_dir)
            if os.path.isdir(symbol_path):
                # Convertir nombre de directorio: "GainX_1200" → "GainX 1200"
                symbol = symbol_dir.replace("_", " ")
                self.models[symbol] = {}

                for timeframe_dir in os.listdir(symbol_path):
                    timeframe_path = os.path.join(symbol_path, timeframe_dir)
                    if os.path.isdir(timeframe_path):
                        try:
                            model = EnsembleModel()
                            model.load_all(timeframe_path)

                            # Extraer el timeframe del nombre del directorio
                            # Formato esperado: "GainX 1200_M1" → extraer "M1"
                            # Separar por espacio y luego por guión bajo
                            parts = timeframe_dir.split('_')
                            if len(parts) >= 2:
                                timeframe_code = parts[-1]  # Último elemento: "M1", "H1", etc.
                            else:
                                timeframe_code = timeframe_dir  # Fallback al nombre completo

                            # Guardar con la clave: "Symbol_Timeframe" (ej: "GainX 1200_M1")
                            model_key = f"{symbol}_{timeframe_code}"
                            self.models[symbol][model_key] = model

                            logger.success(f"Modelo para {symbol} [{timeframe_code}] cargado exitosamente con clave '{model_key}'.")
                            model_count += 1
                        except Exception as e:
                            logger.error(f"Fallo al cargar el modelo para {symbol} [{timeframe_dir}]: {e}")

        if model_count > 0:
            self.is_trained = True
            logger.info(f"Carga completa. Se cargaron un total de {model_count} modelos especializados.")

            # Log de las claves de modelos cargados para debugging
            logger.debug("Resumen de modelos cargados:")
            for symbol, models in self.models.items():
                logger.debug(f"  {symbol}: {list(models.keys())}")
        else:
            logger.error("No se encontraron modelos entrenados. El bot no puede funcionar.")
            raise FileNotFoundError("No se encontraron modelos de IA entrenados.")
