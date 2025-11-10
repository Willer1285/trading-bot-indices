"""
AI Models
Ensemble machine learning models for market prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from loguru import logger
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall


class BaseModel:
    """Base class for all ML models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _align_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aligns the input DataFrame to match the features the model was trained on."""
        if not self.is_fitted or not hasattr(self.scaler, 'n_features_in_'):
            return X

        if hasattr(self.scaler, 'feature_names_in_'):
            expected_features = self.scaler.feature_names_in_
            if list(X.columns) == list(expected_features):
                return X

            logger.warning(f"Feature mismatch for model {self.name}. Realigning columns.")
            X_realigned = X.reindex(columns=expected_features, fill_value=0)
            return X_realigned
        else:
            expected_n_features = self.scaler.n_features_in_
            if X.shape[1] != expected_n_features:
                raise ValueError(f"Feature mismatch for model {self.name}: expects {expected_n_features}, got {X.shape[1]}. Retrain models.")
            return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        raise NotImplementedError

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model {self.name} saved to {path}")

    def load(self, path: str):
        """Load model from disk and validate it"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data.get('is_fitted', False)

        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted.")
        
        if hasattr(self.model, 'classes_') and len(self.model.classes_) < 2:
            raise ValueError(f"Model {self.name} is invalid (only knows 1 class).")

        logger.info(f"Model {self.name} loaded and validated from {path}")


class RandomForestModel(BaseModel):
    """Random Forest Classifier"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest"""
        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.error(f"Cannot train {self.name}: only {unique_classes} unique class(es) in data.")
            self.is_fitted = False
            return

        self.scaler.feature_names_in_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            return np.full(len(X), 1)

        X_aligned = self._align_dataframe(X)
        X_scaled = self.scaler.transform(X_aligned)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        n_samples = len(X)
        n_classes = 3

        if not self.is_fitted or len(self.model.classes_) < n_classes:
            return np.full((n_samples, n_classes), 1/n_classes)

        X_aligned = self._align_dataframe(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        probas = self.model.predict_proba(X_scaled)
        
        if probas.shape[1] < n_classes:
            full_probas = np.full((n_samples, n_classes), 0.0)
            for i, class_label in enumerate(self.model.classes_):
                full_probas[:, class_label] = probas[:, i]
            return full_probas
            
        return probas


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Classifier"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__("GradientBoosting")
        base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train Gradient Boosting"""
        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.error(f"Cannot train {self.name}: only {unique_classes} unique class(es) in data.")
            self.is_fitted = False
            return

        self.scaler.feature_names_in_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            return np.full(len(X), 1)

        X_aligned = self._align_dataframe(X)
        X_scaled = self.scaler.transform(X_aligned)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        n_samples = len(X)
        n_classes = 3

        if not self.is_fitted or len(self.model.classes_) < n_classes:
            return np.full((n_samples, n_classes), 1/n_classes)

        X_aligned = self._align_dataframe(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        probas = self.model.predict_proba(X_scaled)
        
        if probas.shape[1] < n_classes:
            full_probas = np.full((n_samples, n_classes), 0.0)
            for i, class_label in enumerate(self.model.classes_):
                full_probas[:, class_label] = probas[:, i]
            return full_probas
            
        return probas


class SimplePatternModel(BaseModel):
    """
    Optimized rule-based model on technical indicators.

    Generates more signals by using multiple technical indicators with
    weighted scoring system. Less conservative than original version.
    """

    def __init__(self, signal_threshold: float = 0.3):
        super().__init__("PatternBased")
        self.is_fitted = True
        self.signal_threshold = signal_threshold  # Lower threshold = more signals

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals based on technical indicators.

        Scoring system:
        - RSI: Strong weight (current value, not historical)
        - MACD: Medium weight (trend confirmation)
        - Moving Averages: Medium weight (trend direction)
        - Stochastic: Light weight (momentum confirmation)
        - Bollinger Bands: Light weight (volatility signals)

        Lower threshold means more signals will be generated.
        """
        predictions = []

        for _, row in X.iterrows():
            score = 0.0

            # === RSI Signals (Weight: 2.0) ===
            if 'rsi_14' in row and not pd.isna(row['rsi_14']):
                rsi = row['rsi_14']
                if rsi < 30:  # Oversold
                    score += 2.0
                elif rsi < 40:  # Approaching oversold
                    score += 1.0
                elif rsi > 70:  # Overbought
                    score -= 2.0
                elif rsi > 60:  # Approaching overbought
                    score -= 1.0

            # === MACD Signals (Weight: 1.5) ===
            if 'macd_diff' in row and not pd.isna(row['macd_diff']):
                macd_diff = row['macd_diff']
                if macd_diff > 0:
                    score += 1.5
                else:
                    score -= 1.5

            # === Moving Average Trend (Weight: 1.0) ===
            # Check if price is above/below moving averages
            if all(k in row for k in ['close', 'sma_50']) and not any(pd.isna(row[k]) for k in ['close', 'sma_50']):
                if row['close'] > row['sma_50']:
                    score += 1.0
                else:
                    score -= 1.0

            # === EMA Crossover (Weight: 1.0) ===
            if all(k in row for k in ['ema_9', 'ema_21']) and not any(pd.isna(row[k]) for k in ['ema_9', 'ema_21']):
                if row['ema_9'] > row['ema_21']:  # Bullish
                    score += 1.0
                else:  # Bearish
                    score -= 1.0

            # === Stochastic Oscillator (Weight: 0.8) ===
            if 'stoch_k' in row and not pd.isna(row['stoch_k']):
                stoch = row['stoch_k']
                if stoch < 20:  # Oversold
                    score += 0.8
                elif stoch > 80:  # Overbought
                    score -= 0.8

            # === Bollinger Bands (Weight: 0.7) ===
            if all(k in row for k in ['close', 'bb_low', 'bb_high']) and not any(pd.isna(row[k]) for k in ['close', 'bb_low', 'bb_high']):
                if row['close'] <= row['bb_low']:  # Price at lower band
                    score += 0.7
                elif row['close'] >= row['bb_high']:  # Price at upper band
                    score -= 0.7

            # === ADX Trend Strength (Weight: 0.5) ===
            # Only consider signals if trend is strong enough
            if 'adx' in row and not pd.isna(row['adx']):
                if row['adx'] < 25:  # Weak trend, reduce signal strength
                    score *= 0.7

            # === Momentum Indicators (Weight: 0.5) ===
            if 'momentum_5' in row and not pd.isna(row['momentum_5']):
                if row['momentum_5'] > 0:
                    score += 0.5
                else:
                    score -= 0.5

            # === Generate Prediction ===
            # Use lower threshold to generate more signals
            # The meta-model will learn to filter out bad signals
            if score >= self.signal_threshold:
                predictions.append(2)  # BUY
            elif score <= -self.signal_threshold:
                predictions.append(0)  # SELL
            else:
                predictions.append(1)  # HOLD

        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        probas = np.zeros((len(predictions), 3))
        for i, pred in enumerate(predictions):
            if pred == 2: probas[i] = [0.1, 0.2, 0.7]
            elif pred == 0: probas[i] = [0.7, 0.2, 0.1]
            else: probas[i] = [0.2, 0.6, 0.2]
        return probas


class EnsembleModel:
    """Stacking ensemble of multiple models."""

    def __init__(self):
        self.base_models = {
            'random_forest': RandomForestModel(),
            'gradient_boosting': GradientBoostingModel(),
            'pattern_based': SimplePatternModel(),
            'lstm': None
        }
        self.meta_model = LogisticRegression()
        self.is_fitted = False
        logger.info("Initialized Stacking Ensemble.")

    def _get_base_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        base_predictions = {}
        from .feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()

        for name, model in self.base_models.items():
            if model is None:
                logger.warning(f"Model {name} is not available, skipping.")
                continue

            try:
                if name == 'lstm':
                    dummy_y = pd.Series(np.zeros(len(X)), index=X.index)
                    X_seq, _ = feature_engineer.create_sequences(X, dummy_y, default_sequence_length=model.sequence_length)
                    
                    if X_seq.shape[0] > 0:
                        probas = model.predict_proba(X_seq)
                        pred_index = X.index[-len(probas):]
                        for i in range(probas.shape[1]):
                            base_predictions[f"{name}_proba_{i}"] = pd.Series(probas[:, i], index=pred_index)
                    else:
                        logger.warning(f"Not enough data for LSTM sequence. Using neutral placeholders.")
                        base_predictions[f"{name}_proba_0"] = 0.5
                        base_predictions[f"{name}_proba_1"] = 0.5
                else:
                    probas = model.predict_proba(X)
                    for i in range(probas.shape[1]):
                        base_predictions[f"{name}_proba_{i}"] = pd.Series(probas[:, i], index=X.index)
            except ValueError as e:
                logger.error(f"Could not get prediction from model {name}: {e}")
                continue

        return pd.DataFrame(base_predictions, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_seq: np.ndarray, y_seq: np.ndarray):
        if self.base_models['lstm'] is None:
            logger.info("Creating LSTM model...")
            input_dim = X_seq.shape[2]
            self.base_models['lstm'] = LSTMModel(input_dim=input_dim)

        logger.info("Training base models...")
        X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)

        for name, model in self.base_models.items():
            try:
                logger.info(f"Training {name}...")
                if name == 'lstm':
                    model.fit(X_seq, y_seq)
                elif name != 'pattern_based':
                    model.fit(X_clean, y)
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        logger.info("Generating base model predictions for meta-model training...")
        meta_features = self._get_base_model_predictions(X)
        
        combined_data = meta_features.join(y.rename('target')).dropna()
        clean_meta_features = combined_data.drop('target', axis=1)
        clean_y = combined_data['target']

        if clean_meta_features.empty:
            logger.error("Meta-features are empty. Cannot train meta-model.")
            self.is_fitted = False
            return

        logger.info("Training meta-model...")
        self.meta_model.fit(clean_meta_features, clean_y)
        self.is_fitted = True
        logger.success("Stacking Ensemble trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ensemble model must be fitted.")
        
        meta_features = self._get_base_model_predictions(X)
        meta_features.ffill(inplace=True)
        meta_features.bfill(inplace=True)
        
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ensemble model must be fitted.")
            
        meta_features = self._get_base_model_predictions(X)
        meta_features.ffill(inplace=True)
        meta_features.bfill(inplace=True)

        return self.meta_model.predict_proba(meta_features)

    def save_all(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.base_models.items():
            if name != 'pattern_based' and model is not None:
                model.save(str(path / f"{name}.pkl"))

        # Guardar lista de features esperadas para validación posterior
        feature_info = {}
        for name, model in self.base_models.items():
            if name != 'pattern_based' and model is not None and hasattr(model, 'scaler'):
                if hasattr(model.scaler, 'feature_names_in_'):
                    feature_info[name] = list(model.scaler.feature_names_in_)
                elif hasattr(model.scaler, 'n_features_in_'):
                    feature_info[name] = {'n_features': model.scaler.n_features_in_}

        if feature_info:
            with open(path / "feature_info.pkl", 'wb') as f:
                pickle.dump(feature_info, f)
            logger.info(f"Feature information saved to {path / 'feature_info.pkl'}")

        if self.is_fitted:
            with open(path / "meta_model.pkl", 'wb') as f:
                pickle.dump(self.meta_model, f)
            logger.info(f"Meta-model saved to {path / 'meta_model.pkl'}")

    def load_all(self, directory: str):
        path = Path(directory)

        # Cargar información de features esperadas
        feature_info_path = path / "feature_info.pkl"
        expected_features = {}
        if feature_info_path.exists():
            try:
                with open(feature_info_path, 'rb') as f:
                    expected_features = pickle.load(f)
                logger.info(f"Loaded feature information from {feature_info_path}")
            except Exception as e:
                logger.warning(f"Failed to load feature info: {e}")

        if self.base_models['lstm'] is None:
            self.base_models['lstm'] = LSTMModel()

        for name, model in self.base_models.items():
            if name != 'pattern_based':
                model_path = path / f"{name}.pkl"
                if model_path.exists():
                    try:
                        model.load(str(model_path))

                        # Validar features si están disponibles
                        if name in expected_features:
                            expected = expected_features[name]
                            if isinstance(expected, list):
                                logger.info(f"Model {name} expects {len(expected)} features")
                            elif isinstance(expected, dict) and 'n_features' in expected:
                                logger.info(f"Model {name} expects {expected['n_features']} features")

                    except Exception as e:
                        logger.error(f"Failed to load {name}: {e}")
                elif name == 'lstm':
                    self.base_models['lstm'] = None

        meta_model_path = path / "meta_model.pkl"
        if meta_model_path.exists():
            try:
                with open(meta_model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                self.is_fitted = True
                logger.info("Ensemble model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load meta-model: {e}")
                self.is_fitted = False
        else:
            logger.error("Meta-model not found.")
            self.is_fitted = False

# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss para clasificación multi-clase con labels enteros.

    Focal Loss reduce la contribución de ejemplos fáciles y enfoca el
    entrenamiento en ejemplos difíciles y clases minoritarias.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Factor de balanceo para clases (default 0.25)
        gamma: Factor de enfoque (default 2.0). Valores más altos reducen
               más la contribución de ejemplos fáciles.
        from_logits: Si True, espera logits sin softmax (default False)
    """

    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Calcula focal loss.

        Args:
            y_true: Labels enteros (batch_size,) con valores [0, 1, 2]
            y_pred: Predicciones (batch_size, n_classes) con probabilidades

        Returns:
            Focal loss promediado sobre el batch
        """
        # Aplicar softmax si recibimos logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip para estabilidad numérica
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Convertir y_true a one-hot si es necesario
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        # Calcular p_t (probabilidad de la clase correcta)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Calcular focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        focal_weight = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        """Para serialización del modelo."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(BaseModel):
    """LSTM model for multi-class sequence classification with Focal Loss."""

    def __init__(self, sequence_length: int = 20, input_dim: int = 50):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.model = None
        self.n_classes = 3  # SELL=0, HOLD=1, BUY=2

    def _build_model(self):
        """
        Build the Keras LSTM model for multi-class classification.

        Arquitectura simplificada:
        - LSTM con 32 unidades (vs 50 anterior) para reducir parámetros
        - Dropout 0.3 para mayor regularización
        - Dense final con 3 salidas (SELL, HOLD, BUY)
        - Softmax para clasificación multi-clase
        - Focal Loss para manejar desbalance de clases
        """
        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(self.sequence_length, self.input_dim)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')  # Multi-clase: 3 salidas
        ])

        # Usar Focal Loss para penalizar más errores en clases minoritarias
        model.compile(
            optimizer='adam',
            loss=SparseCategoricalFocalLoss(alpha=0.25, gamma=2.0),
            metrics=['accuracy',
                    tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                    Precision(name='precision'),
                    Recall(name='recall')]
        )
        self.model = model
        logger.info("LSTM model built with Focal Loss (alpha=0.25, gamma=2.0)")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the LSTM model with intelligent undersampling for class balancing.

        Estrategia de balanceo:
        - Mantener todas las muestras BUY y SELL (clases minoritarias)
        - Reducir HOLD del ~95% a ~70% mediante undersampling aleatorio
        - Aplicar class_weights adicionales para compensar desbalance residual
        """
        if self.model is None:
            self._build_model()

        # --- PASO 1: Analizar distribución original ---
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        total_samples = len(y)

        logger.info(f"Original class distribution: {class_distribution}")
        logger.info(f"Original total samples: {total_samples}")

        # --- PASO 2: Undersampling inteligente de HOLD ---
        # Calcular cuántas muestras HOLD queremos mantener
        buy_count = class_distribution.get(2, 0)  # BUY
        sell_count = class_distribution.get(0, 0)  # SELL
        hold_count = class_distribution.get(1, 0)  # HOLD

        # Mantener todas BUY/SELL, reducir HOLD a ~70% del total
        minority_total = buy_count + sell_count
        target_hold_count = int(minority_total * 2.3)  # HOLD será ~70% del nuevo dataset

        if hold_count > target_hold_count and target_hold_count > 0:
            # Undersampling: mantener todas BUY/SELL, samplear HOLD
            indices_buy = np.where(y == 2)[0]
            indices_sell = np.where(y == 0)[0]
            indices_hold = np.where(y == 1)[0]

            # Samplear aleatoriamente HOLD
            np.random.seed(42)
            indices_hold_sampled = np.random.choice(indices_hold, size=target_hold_count, replace=False)

            # Combinar índices
            indices_balanced = np.concatenate([indices_sell, indices_hold_sampled, indices_buy])
            np.random.shuffle(indices_balanced)

            # Aplicar balanceo
            X = X[indices_balanced]
            y = y[indices_balanced]

            logger.info(f"Applied undersampling: HOLD reduced from {hold_count} to {target_hold_count}")
        else:
            logger.info("Skipping undersampling (insufficient HOLD samples or already balanced)")

        # --- PASO 3: Recalcular class weights después del balanceo ---
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples_balanced = len(y)
        class_weights = {int(cls): total_samples_balanced / (len(unique_classes) * count)
                        for cls, count in zip(unique_classes, class_counts)}

        logger.info(f"Balanced class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Balanced total samples: {total_samples_balanced}")
        logger.info(f"Class weights: {class_weights}")

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        self.scaler.fit(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        history = self.model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,  # Aumentado de 0.1 a 0.2 para métricas más confiables
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )

        # Log de métricas finales
        final_epoch = len(history.history['loss'])
        logger.info(f"Training stopped at epoch {final_epoch}")
        logger.info(f"Final metrics - Loss: {history.history['loss'][-1]:.4f}, "
                   f"Accuracy: {history.history['accuracy'][-1]:.4f}, "
                   f"AUC: {history.history['auc'][-1]:.4f}")
        logger.info(f"Validation metrics - Loss: {history.history['val_loss'][-1]:.4f}, "
                   f"Accuracy: {history.history['val_accuracy'][-1]:.4f}, "
                   f"AUC: {history.history['val_auc'][-1]:.4f}")

        self.is_fitted = True
        logger.info(f"{self.name} trained successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for multi-class classification.

        Returns:
            np.ndarray: Predicted classes (0=SELL, 1=HOLD, 2=BUY)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        # Para multi-clase, usar argmax en vez de threshold
        probas = self.model.predict(X_scaled, verbose=0)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for multi-class classification.

        Returns:
            np.ndarray: Probability matrix (n_samples, 3) donde cada fila suma 1.0
                       Columnas: [P(SELL), P(HOLD), P(BUY)]
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        # El modelo ya devuelve probabilidades para las 3 clases (softmax)
        return self.model.predict(X_scaled, verbose=0)

    def save(self, path: str):
        """Save Keras model and scaler separately."""
        if self.model is None:
            logger.warning("Attempted to save an un-built LSTM model.")
            return
            
        model_path = path.replace('.pkl', '.keras')
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        
        self.model.save(model_path)
        
        scaler_data = {'scaler': self.scaler, 'is_fitted': self.is_fitted}
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)

        Path(path).touch()
            
        logger.info(f"LSTM model saved to {model_path} and {scaler_path}")

    def load(self, path: str):
        """Load Keras model and scaler with custom Focal Loss."""
        model_path = path.replace('.pkl', '.keras')
        scaler_path = path.replace('.pkl', '_scaler.pkl')

        if not Path(model_path).exists() or not Path(scaler_path).exists():
            raise FileNotFoundError(f"Model files not found: {model_path} or {scaler_path}")

        # Cargar modelo con custom objects para Focal Loss
        self.model = load_model(
            model_path,
            custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss}
        )
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.is_fitted = scaler_data['is_fitted']

        logger.info(f"LSTM model loaded from {model_path} and {scaler_path}")


# ============================================================================
# META-LABELING FUNCTIONS
# ============================================================================

def create_meta_labels(df: pd.DataFrame, primary_predictions: pd.Series,
                      lookforward_periods: int = 20,
                      profit_target_atr_mult: float = 1.5,
                      loss_limit_atr_mult: float = 1.0) -> pd.Series:
    """
    Create meta-labels for filtering primary model signals.

    Meta-labeling: Instead of predicting direction (BUY/SELL), we predict
    whether a signal from the primary model will be profitable or not.

    This allows the meta-model to learn which signals to take and which to skip.

    Args:
        df: DataFrame with OHLC data and ATR indicator
        primary_predictions: Series with primary model predictions (0=SELL, 1=HOLD, 2=BUY)
        lookforward_periods: How many periods to look ahead to evaluate profitability
        profit_target_atr_mult: Profit target as multiple of ATR
        loss_limit_atr_mult: Maximum acceptable loss as multiple of ATR

    Returns:
        Series with meta-labels:
        - 1: Signal was profitable (good signal)
        - 0: Signal was not profitable (bad signal)
        - NaN: No signal (HOLD), will be dropped later
    """
    meta_labels = pd.Series(index=primary_predictions.index, dtype=float)

    # Ensure we have ATR in the dataframe
    if 'atr' not in df.columns:
        # Calculate ATR if not present
        from ta.volatility import average_true_range
        df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=14)

    for i, (idx, pred) in enumerate(primary_predictions.items()):
        # Only create labels for actual signals (not HOLD)
        if pred == 1:  # HOLD - skip
            meta_labels[idx] = np.nan
            continue

        # Get current price and ATR
        try:
            current_price = df.loc[idx, 'close']
            atr = df.loc[idx, 'atr']

            # Handle missing ATR
            if pd.isna(atr) or atr == 0:
                meta_labels[idx] = np.nan
                continue

        except (KeyError, IndexError):
            meta_labels[idx] = np.nan
            continue

        # Calculate profit target and stop loss
        profit_target = atr * profit_target_atr_mult
        loss_limit = atr * loss_limit_atr_mult

        # Get future prices (lookforward window)
        future_start_idx = i + 1
        future_end_idx = min(i + 1 + lookforward_periods, len(df))

        if future_start_idx >= len(df):
            # Not enough future data
            meta_labels[idx] = np.nan
            continue

        future_prices = df['close'].iloc[future_start_idx:future_end_idx]
        future_highs = df['high'].iloc[future_start_idx:future_end_idx]
        future_lows = df['low'].iloc[future_start_idx:future_end_idx]

        if len(future_prices) == 0:
            meta_labels[idx] = np.nan
            continue

        # Determine if signal was profitable
        is_profitable = False

        if pred == 2:  # BUY signal
            # Check if price reached profit target before hitting stop loss
            max_profit = (future_highs - current_price).max()
            max_loss = (current_price - future_lows).min()

            # Signal is profitable if:
            # 1. Profit target was reached, OR
            # 2. Max profit > loss and final price is positive
            if max_profit >= profit_target:
                is_profitable = True
            elif max_profit > max_loss and future_prices.iloc[-1] > current_price:
                is_profitable = True
            elif max_loss > loss_limit:
                is_profitable = False
            else:
                # Check final outcome
                final_profit = future_prices.iloc[-1] - current_price
                is_profitable = final_profit > 0

        elif pred == 0:  # SELL signal
            # Check if price reached profit target before hitting stop loss
            max_profit = (current_price - future_lows).max()
            max_loss = (future_highs - current_price).min()

            # Signal is profitable if:
            # 1. Profit target was reached, OR
            # 2. Max profit > loss and final price is negative
            if max_profit >= profit_target:
                is_profitable = True
            elif max_profit > max_loss and future_prices.iloc[-1] < current_price:
                is_profitable = True
            elif max_loss > loss_limit:
                is_profitable = False
            else:
                # Check final outcome
                final_profit = current_price - future_prices.iloc[-1]
                is_profitable = final_profit > 0

        meta_labels[idx] = 1 if is_profitable else 0

    # Log statistics
    total_signals = (~meta_labels.isna()).sum()
    profitable_signals = (meta_labels == 1).sum()
    unprofitable_signals = (meta_labels == 0).sum()

    if total_signals > 0:
        win_rate = (profitable_signals / total_signals) * 100
        logger.info(f"Meta-labeling stats: {total_signals} signals total, "
                   f"{profitable_signals} profitable ({win_rate:.1f}% win rate), "
                   f"{unprofitable_signals} unprofitable")
    else:
        logger.warning("No signals generated by primary model for meta-labeling")

    return meta_labels
