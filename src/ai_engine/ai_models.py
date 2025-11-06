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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
    """Rule-based model on technical indicators."""

    def __init__(self):
        super().__init__("PatternBased")
        self.is_fitted = True

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_with_memory = X.copy()
        
        window = 20
        if 'rsi_14' in X.columns:
            X_with_memory['rsi_recently_oversold'] = X['rsi_14'].rolling(window=window).apply(lambda x: (x < 30).any(), raw=True).fillna(0).astype(bool)
            X_with_memory['rsi_recently_overbought'] = X['rsi_14'].rolling(window=window).apply(lambda x: (x > 70).any(), raw=True).fillna(0).astype(bool)
        else:
            X_with_memory['rsi_recently_oversold'] = False
            X_with_memory['rsi_recently_overbought'] = False

        predictions = []
        for _, row in X_with_memory.iterrows():
            score = 0
            if row.get('rsi_recently_oversold', False): score += 2
            if row.get('rsi_recently_overbought', False): score -= 2
            if 'macd_diff' in row:
                if row['macd_diff'] > 0: score += 1
                else: score -= 1
            if 'trend_20' in row: score += row['trend_20']

            if score >= 1: predictions.append(2)
            elif score <= -1: predictions.append(0)
            else: predictions.append(1)

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
                model.save(path / f"{name}.pkl")
        
        if self.is_fitted:
            with open(path / "meta_model.pkl", 'wb') as f:
                pickle.dump(self.meta_model, f)
            logger.info(f"Meta-model saved to {path / 'meta_model.pkl'}")

    def load_all(self, directory: str):
        path = Path(directory)
        
        if self.base_models['lstm'] is None:
            self.base_models['lstm'] = LSTMModel()

        for name, model in self.base_models.items():
            if name != 'pattern_based':
                model_path = path / f"{name}.pkl"
                if model_path.exists():
                    try:
                        model.load(str(model_path))
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
            except Exception as e:
                logger.error(f"Failed to load meta-model: {e}")
                self.is_fitted = False
        else:
            logger.error("Meta-model not found.")
            self.is_fitted = False

# ... (rest of the file is the same)
class LSTMModel(BaseModel):
    """LSTM model for sequence classification."""

    def __init__(self, sequence_length: int = 50, input_dim: int = 50):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.model = None

    def _build_model(self):
        """Build the Keras LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.input_dim)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the LSTM model."""
        if self.model is None:
            self._build_model()
            
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        self.scaler.fit(X_reshaped)
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        self.model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        self.is_fitted = True
        logger.info(f"{self.name} trained successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))

        return (self.model.predict(X_scaled) > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape((n_samples, n_timesteps, n_features))
        
        proba_positive = self.model.predict(X_scaled)
        proba_negative = 1 - proba_positive
        return np.hstack([proba_negative, proba_positive])

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
        """Load Keras model and scaler."""
        model_path = path.replace('.pkl', '.keras')
        scaler_path = path.replace('.pkl', '_scaler.pkl')

        if not Path(model_path).exists() or not Path(scaler_path).exists():
            raise FileNotFoundError(f"Model files not found: {model_path} or {scaler_path}")

        self.model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.is_fitted = scaler_data['is_fitted']
        
        logger.info(f"LSTM model loaded from {model_path} and {scaler_path}")
