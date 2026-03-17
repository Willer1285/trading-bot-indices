"""
Temporal Fusion Transformer (TFT) Model
Advanced time series forecasting using attention mechanisms
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import pickle

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    TFT_AVAILABLE = True
except ImportError:
    logger.warning("pytorch-forecasting not installed. TFT model will not be available.")
    TFT_AVAILABLE = False


class TFTModel:
    """
    Temporal Fusion Transformer for multi-horizon time series forecasting.

    TFT combines:
    - Variable selection networks
    - Static covariate encoders
    - Sequence-to-sequence layer with LSTM
    - Multi-head attention mechanisms
    - Quantile forecasts for uncertainty estimation

    Perfect for financial time series with multiple features.
    """

    def __init__(
        self,
        max_encoder_length: int = 60,
        max_prediction_length: int = 10,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001
    ):
        """
        Initialize TFT model.

        Args:
            max_encoder_length: Historical sequence length to encode
            max_prediction_length: Number of future steps to predict
            hidden_size: Size of hidden layers
            lstm_layers: Number of LSTM layers
            attention_head_size: Number of attention heads
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for training
        """
        if not TFT_AVAILABLE:
            raise ImportError("pytorch-forecasting is required for TFT model")

        self.name = "TFT"
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.is_fitted = False

        # Feature names will be set during training
        self.time_varying_known_reals = []
        self.time_varying_unknown_reals = []
        self.static_categoricals = []

        logger.info(f"Initialized TFT model with encoder={max_encoder_length}, "
                   f"prediction={max_prediction_length}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        time_idx_col: str = 'time_idx',
        group_ids: List[str] = ['symbol']
    ) -> TimeSeriesDataSet:
        """
        Prepare data for TFT training.

        Args:
            df: DataFrame with time series data
            target_col: Name of target column to predict
            time_idx_col: Name of time index column
            group_ids: List of columns identifying different time series

        Returns:
            TimeSeriesDataSet ready for training
        """
        # Ensure required columns exist
        if time_idx_col not in df.columns:
            df[time_idx_col] = range(len(df))

        if group_ids[0] not in df.columns:
            df[group_ids[0]] = 'default'

        # Identify feature types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and time_idx from features
        feature_cols = [col for col in numeric_cols
                       if col not in [target_col, time_idx_col] and col not in group_ids]

        # Split into known and unknown reals
        # Known reals: features we know in advance (e.g., time features)
        # Unknown reals: features we don't know in advance (most technical indicators)
        self.time_varying_known_reals = []  # Empty for now, could add time features
        self.time_varying_unknown_reals = feature_cols

        logger.info(f"Preparing TFT dataset with {len(feature_cols)} features")
        logger.info(f"Target: {target_col}, Time varying unknown: {len(self.time_varying_unknown_reals)}")

        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            df,
            time_idx=time_idx_col,
            target=target_col,
            group_ids=group_ids,
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=group_ids,
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        return training

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10
    ):
        """
        Train the TFT model.

        Args:
            df: DataFrame with time series data
            target_col: Column to predict
            max_epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
        """
        logger.info("Starting TFT training...")

        # Prepare training dataset
        self.training_dataset = self.prepare_data(df, target_col=target_col)

        # Create validation dataset (last 20% of data)
        validation = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )

        # Create dataloaders
        train_dataloader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0
        )

        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=batch_size * 2,
            num_workers=0
        )

        # Configure trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=patience,
            verbose=False,
            mode="min"
        )

        lr_logger = LearningRateMonitor()

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback],
            enable_progress_bar=True,
            enable_model_summary=True
        )

        # Create TFT model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size // 2,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            lstm_layers=self.lstm_layers,
            reduce_on_plateau_patience=4
        )

        # Train
        logger.info(f"Training TFT for {max_epochs} epochs...")
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        self.is_fitted = True
        logger.success("TFT training completed!")

    def predict(
        self,
        df: pd.DataFrame,
        return_quantiles: bool = False
    ) -> np.ndarray:
        """
        Make predictions using trained TFT model.

        Args:
            df: DataFrame with input features
            return_quantiles: If True, return all quantile predictions

        Returns:
            Predictions array (median prediction by default)
        """
        if not self.is_fitted or self.model is None:
            logger.warning("TFT model not fitted. Returning neutral predictions.")
            return np.ones(len(df))

        # Create prediction dataset
        predict_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )

        predict_dataloader = predict_dataset.to_dataloader(
            train=False,
            batch_size=64,
            num_workers=0
        )

        # Make predictions
        predictions = self.model.predict(
            predict_dataloader,
            mode="quantiles" if return_quantiles else "prediction",
            return_x=False
        )

        # Extract predictions
        if return_quantiles:
            return predictions.cpu().numpy()
        else:
            # Return median (50th percentile)
            if len(predictions.shape) == 3:
                # Shape: (batch, time, quantiles)
                median_idx = predictions.shape[2] // 2
                return predictions[:, -1, median_idx].cpu().numpy()
            else:
                return predictions[:, -1].cpu().numpy()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for trading signals.

        Converts price predictions to BUY/HOLD/SELL probabilities.

        Args:
            df: Input features DataFrame

        Returns:
            Probability array with shape (n_samples, 3) for [SELL, HOLD, BUY]
        """
        if not self.is_fitted:
            n_samples = len(df)
            return np.full((n_samples, 3), 1/3)

        # Get quantile predictions (lower, median, upper)
        predictions = self.predict(df, return_quantiles=True)

        # Convert to probabilities based on predicted direction
        probas = []

        if len(predictions.shape) == 3:
            # Get last prediction for each sample
            last_predictions = predictions[:, -1, :]

            for pred_quantiles in last_predictions:
                lower = pred_quantiles[0]  # 10th percentile
                median = pred_quantiles[3]  # 50th percentile
                upper = pred_quantiles[6]  # 90th percentile

                # Calculate trend strength
                upward_strength = (median - lower) / (upper - lower + 1e-8)

                # Convert to probabilities
                if upward_strength > 0.6:
                    # Strong upward trend -> BUY
                    probas.append([0.1, 0.2, 0.7])
                elif upward_strength < 0.4:
                    # Strong downward trend -> SELL
                    probas.append([0.7, 0.2, 0.1])
                else:
                    # Uncertain -> HOLD
                    probas.append([0.2, 0.6, 0.2])
        else:
            # Fallback to neutral probabilities
            probas = [[1/3, 1/3, 1/3]] * len(df)

        return np.array(probas)

    def get_attention_weights(self, df: pd.DataFrame) -> Dict:
        """
        Extract attention weights to understand which features are important.

        Args:
            df: Input data

        Returns:
            Dictionary with attention weights
        """
        if not self.is_fitted or self.model is None:
            return {}

        predict_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )

        predict_dataloader = predict_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0
        )

        # Get interpretation
        interpretation = self.model.interpret_output(
            next(iter(predict_dataloader))
        )

        return {
            'attention': interpretation.get('attention', None),
            'static_variables': interpretation.get('static_variables', None),
            'encoder_variables': interpretation.get('encoder_variables', None)
        }

    def save(self, path: str):
        """Save TFT model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch Lightning checkpoint
        checkpoint_path = str(model_path).replace('.pkl', '.ckpt')
        self.trainer.save_checkpoint(checkpoint_path)

        # Save metadata
        metadata = {
            'is_fitted': self.is_fitted,
            'max_encoder_length': self.max_encoder_length,
            'max_prediction_length': self.max_prediction_length,
            'hidden_size': self.hidden_size,
            'lstm_layers': self.lstm_layers,
            'attention_head_size': self.attention_head_size,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'time_varying_known_reals': self.time_varying_known_reals,
            'time_varying_unknown_reals': self.time_varying_unknown_reals,
            'static_categoricals': self.static_categoricals
        }

        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"TFT model saved to {checkpoint_path} and {path}")

    def load(self, path: str):
        """Load TFT model from disk."""
        # Load metadata
        with open(path, 'rb') as f:
            metadata = pickle.load(f)

        self.is_fitted = metadata['is_fitted']
        self.max_encoder_length = metadata['max_encoder_length']
        self.max_prediction_length = metadata['max_prediction_length']
        self.hidden_size = metadata['hidden_size']
        self.lstm_layers = metadata['lstm_layers']
        self.attention_head_size = metadata['attention_head_size']
        self.dropout = metadata['dropout']
        self.learning_rate = metadata['learning_rate']
        self.time_varying_known_reals = metadata['time_varying_known_reals']
        self.time_varying_unknown_reals = metadata['time_varying_unknown_reals']
        self.static_categoricals = metadata['static_categoricals']

        # Load PyTorch Lightning checkpoint
        checkpoint_path = str(path).replace('.pkl', '.ckpt')
        self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

        logger.info(f"TFT model loaded from {checkpoint_path}")
