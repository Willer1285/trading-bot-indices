#!/usr/bin/env python3
"""
Script de prueba rápida para entrenar un solo símbolo/timeframe
Útil para verificar configuraciones sin esperar 50 modelos
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_models import (
    initialize_mt5, download_data_from_mt5, prepare_training_data,
    save_training_metadata, logger
)
from src.ai_engine.ai_models import StackingEnsemble
from src.config import config
import MetaTrader5 as mt5

def test_train_single_model():
    """Train a single model for testing"""

    # CONFIGURACIÓN DE PRUEBA - Modifica estos valores
    TEST_SYMBOL = "PainX 400"  # Símbolo a probar
    TEST_TIMEFRAME = "15m"     # Timeframe a probar (1m, 5m, 15m, 1h, 4h, 1d)
    TEST_CANDLES = 20000       # Cantidad de velas a descargar

    logger.info("=" * 90)
    logger.info(f"🧪 TEST MODE: Training single model - {TEST_SYMBOL} [{TEST_TIMEFRAME}]")
    logger.info("=" * 90)

    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MT5")
        return False

    try:
        # Convert timeframe string to MT5 constant
        timeframe_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1
        }

        mt5_timeframe = timeframe_map.get(TEST_TIMEFRAME)
        if not mt5_timeframe:
            logger.error(f"Invalid timeframe: {TEST_TIMEFRAME}")
            return False

        logger.info(f"\n--- Training {TEST_SYMBOL} [{TEST_TIMEFRAME}] ---")

        # 1. Download data
        df = download_data_from_mt5(TEST_SYMBOL, mt5_timeframe, TEST_CANDLES)
        if df is None or len(df) < 500:
            logger.error(f"Insufficient data for {TEST_SYMBOL} [{TEST_TIMEFRAME}]")
            return False

        logger.info(f"✓ Downloaded {len(df)} candles")

        # 2. Save historical data
        historical_dir = Path("historical_data") / TEST_SYMBOL.replace(" ", "_")
        historical_dir.mkdir(parents=True, exist_ok=True)

        tf_file_map = {
            "1m": "M1", "5m": "M5", "15m": "M15",
            "1h": "H1", "4h": "H4", "1d": "D1"
        }

        file_timeframe = tf_file_map.get(TEST_TIMEFRAME, TEST_TIMEFRAME.upper())
        historical_file = historical_dir / f"{TEST_SYMBOL.replace(' ', '_')}_{file_timeframe}.csv"
        df.to_csv(historical_file, index=False)
        logger.success(f"Saved historical data to {historical_file}")

        # 3. Prepare training data
        logger.info("Preparing training data with Meta-Labeling...")
        X_train, y_train, X_train_seq, y_train_seq = prepare_training_data(df)

        if X_train is None or len(X_train) == 0:
            logger.error("Data preparation failed")
            return False

        logger.info(f"✓ Training data prepared: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"✓ Sequences: X_seq={X_train_seq.shape}, y_seq={y_train_seq.shape}")

        # 4. Train model
        logger.info("Training Ensemble model...")
        model = StackingEnsemble()
        model.fit(X_train, y_train, X_train_seq, y_train_seq)
        logger.success("Model trained successfully!")

        # 5. Save model
        model_dir = Path("models") / TEST_SYMBOL.replace(" ", "_") / f"{TEST_SYMBOL.replace(' ', '_')}_{file_timeframe}"
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to '{model_dir}'...")
        model.save_all(str(model_dir))

        # Save metadata
        save_training_metadata(
            model_dir=model_dir,
            symbol=TEST_SYMBOL,
            timeframe=TEST_TIMEFRAME,
            samples=len(X_train),
            features=X_train.shape[1] if len(X_train.shape) > 1 else 0
        )

        logger.success(f"✅ Test training completed successfully for {TEST_SYMBOL} [{TEST_TIMEFRAME}]")
        logger.info(f"📁 Model saved to: {model_dir}")

        return True

    except Exception as e:
        logger.error(f"Error during test training: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

if __name__ == "__main__":
    logger.info("Starting single model test training...")
    success = test_train_single_model()

    if success:
        logger.success("✅ TEST PASSED: Single model training completed successfully")
        logger.info("\n💡 If results look good, run full training with:")
        logger.info("   python train_models.py --source mt5")
    else:
        logger.error("❌ TEST FAILED: Check logs above for errors")
        sys.exit(1)
