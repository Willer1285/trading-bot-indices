"""
Training Script for Hybrid AI System (TFT + RL + LLM)
Optimizes trading bot profitability using advanced AI techniques
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config
from src.data_collector.mt5_connector import MT5Connector
from src.data_collector.mt5_market_data_manager import MT5MarketDataManager
from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import EnsembleModel


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/hybrid_training.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG"
    )


def collect_training_data(symbol: str, timeframe: str, bars: int = 10000) -> pd.DataFrame:
    """
    Collect historical data for training.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (e.g., "1h")
        bars: Number of historical bars

    Returns:
        DataFrame with OHLC data
    """
    logger.info(f"Collecting {bars} bars of {timeframe} data for {symbol}...")

    mt5 = MT5Connector()

    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")

    try:
        data_manager = MT5MarketDataManager(mt5)
        df = data_manager.get_historical_data(symbol, timeframe, bars)

        if df is None or len(df) == 0:
            raise ValueError(f"No data collected for {symbol}")

        logger.info(f"Collected {len(df)} bars for {symbol}")
        return df

    finally:
        mt5.shutdown()


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for training.

    Args:
        df: Raw OHLC data

    Returns:
        Tuple of (features, labels)
    """
    logger.info("Engineering features...")

    feature_engineer = FeatureEngineer()

    # Add technical indicators
    df = feature_engineer.add_technical_indicators(df)

    # Create labels (1 if price goes up in next 10 bars, 0 otherwise)
    df['future_return'] = df['close'].shift(-10) / df['close'] - 1
    df['label'] = 0  # HOLD
    df.loc[df['future_return'] > 0.01, 'label'] = 2  # BUY (>1% gain)
    df.loc[df['future_return'] < -0.01, 'label'] = 0  # SELL (<-1% loss)

    # Drop rows with NaN
    df = df.dropna()

    # Split features and labels
    label_cols = ['label', 'future_return']
    feature_cols = [col for col in df.columns if col not in label_cols]

    X = df[feature_cols]
    y = df['label']

    logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    return X, y


def train_hybrid_model(
    symbol: str,
    timeframe: str,
    use_tft: bool = True,
    use_rl: bool = True,
    use_llm: bool = True,
    save_path: str = "models/hybrid"
):
    """
    Train hybrid AI model.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        use_tft: Enable TFT model
        use_rl: Enable RL agent
        use_llm: Enable LLM sentiment
        save_path: Path to save models
    """
    logger.info("=" * 80)
    logger.info("HYBRID AI TRAINING - TFT + RL + LLM")
    logger.info("=" * 80)

    # Collect data
    df = collect_training_data(symbol, timeframe, bars=10000)

    # Prepare features
    X, y = prepare_features(df)

    # Create sequences for LSTM and TFT
    feature_engineer = FeatureEngineer()
    X_seq, y_seq = feature_engineer.create_sequences(X, y, default_sequence_length=50)

    logger.info(f"Created {len(X_seq)} sequences for sequence models")

    # Create ensemble model with hybrid enabled
    logger.info("Initializing Hybrid Ensemble Model...")
    ensemble = EnsembleModel(use_hybrid=True)

    # Train the ensemble
    logger.info("Starting training...")
    try:
        ensemble.fit(X, y, X_seq, y_seq)
        logger.success("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save models
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving models to {save_dir}...")
    ensemble.save_all(str(save_dir))

    logger.success(f"Models saved to {save_dir}")

    # Evaluate on training data (just for sanity check)
    logger.info("Evaluating models...")
    try:
        predictions = ensemble.predict(X)
        accuracy = (predictions == y).mean()
        logger.info(f"Training accuracy: {accuracy:.2%}")

        # Show prediction distribution
        pred_counts = pd.Series(predictions).value_counts()
        logger.info(f"Prediction distribution: {pred_counts.to_dict()}")

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")

    logger.info("=" * 80)
    logger.success("HYBRID AI TRAINING COMPLETED!")
    logger.info("=" * 80)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid AI Model (TFT + RL + LLM) for Trading Bot"
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Trading symbol (default: SPY)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Timeframe (default: 1h)'
    )

    parser.add_argument(
        '--no-tft',
        action='store_true',
        help='Disable TFT model'
    )

    parser.add_argument(
        '--no-rl',
        action='store_true',
        help='Disable RL agent'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM sentiment'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='models/hybrid',
        help='Path to save models (default: models/hybrid)'
    )

    args = parser.parse_args()

    setup_logging()

    logger.info("Starting Hybrid AI Training...")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"TFT enabled: {not args.no_tft}")
    logger.info(f"RL enabled: {not args.no_rl}")
    logger.info(f"LLM enabled: {not args.no_llm}")

    try:
        train_hybrid_model(
            symbol=args.symbol,
            timeframe=args.timeframe,
            use_tft=not args.no_tft,
            use_rl=not args.no_rl,
            use_llm=not args.no_llm,
            save_path=args.save_path
        )

        logger.success("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
