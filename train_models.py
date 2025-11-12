import asyncio
import sys
import os
import io
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import joblib
import MetaTrader5 as mt5
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
# Load .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import config
from src.ai_engine.feature_engineering import FeatureEngineer
from src.ai_engine.ai_models import EnsembleModel, SimplePatternModel, create_meta_labels

# --- Configuration ---
MODEL_OUTPUT_DIR = config.models_directory
HISTORICAL_DATA_DIR = "historical_data"

# Mapeo de timeframes
TIMEFRAME_MAP = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1,
    '1w': mt5.TIMEFRAME_W1,
    '1M': mt5.TIMEFRAME_MN1
}

# --- MT5 Connection Functions ---

def initialize_mt5() -> bool:
    """Initialize MT5 connection."""
    try:
        # Initialize MT5
        if config.mt5_path:
            if not mt5.initialize(config.mt5_path):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

        logger.info("MT5 initialized successfully")

        # Login if credentials provided
        if config.mt5_login and config.mt5_password and config.mt5_server:
            if not mt5.login(config.mt5_login, config.mt5_password, config.mt5_server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

            logger.info(f"Logged in to MT5 account {config.mt5_login}")

        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Could not get account info, but MT5 is initialized")
        else:
            logger.info(f"Connected to MT5 - Account: {account_info.login}, "
                       f"Balance: {account_info.balance}, "
                       f"Server: {account_info.server}")

        return True

    except Exception as e:
        logger.error(f"Error initializing MT5: {e}")
        return False


def download_data_from_mt5(symbol: str, timeframe: str, num_candles: int = 5000) -> pd.DataFrame:
    """
    Download historical data from MT5.

    Args:
        symbol: Trading symbol (e.g., "GainX 600")
        timeframe: Timeframe string (e.g., "1m", "1h")
        num_candles: Number of candles to download

    Returns:
        DataFrame with OHLCV data
    """
    try:
        mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        # Download data
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to download data for {symbol} [{timeframe}]: {mt5.last_error()}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)

        # Rename columns
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'VOL',
            'spread': 'SPREAD'
        }, inplace=True)

        # Ensure required columns exist
        if 'VOL' not in df.columns:
            df['VOL'] = df.get('volume', 0)
        if 'SPREAD' not in df.columns:
            df['SPREAD'] = 0

        # Select only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']]

        logger.success(f"Downloaded {len(df)} candles for {symbol} [{timeframe}] from MT5")
        return df

    except Exception as e:
        logger.error(f"Error downloading data from MT5 for {symbol} [{timeframe}]: {e}")
        return pd.DataFrame()


def shutdown_mt5():
    """Shutdown MT5 connection."""
    try:
        mt5.shutdown()
        logger.info("MT5 connection closed")
    except Exception as e:
        logger.warning(f"Error shutting down MT5: {e}")


# --- Main Training Logic ---

def prepare_training_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Prepares the feature set (X) and the target variable (y) for meta-labeling."""
    logger.info("Preparing data for training with Meta-Labeling...")
    feature_engineer = FeatureEngineer()

    # 1. Engineer features
    df_features = feature_engineer.extract_features(df)

    # 2. Get primary model (rules) predictions
    logger.info("Generating signals from primary model (SimplePatternModel)...")
    primary_model = SimplePatternModel()

    # Prepare a temporary feature set for the primary model
    temp_X = feature_engineer.prepare_for_model(df_features)

    # Align indices before making predictions
    df_features_aligned = df_features.loc[temp_X.index]

    primary_predictions = pd.Series(primary_model.predict(temp_X), index=temp_X.index)

    # 3. Create meta-labels based on primary model signals
    # IMPORTANTE: Usar parámetros más permisivos durante entrenamiento para que el modelo
    # aprenda a identificar señales ganadoras de perdedoras con datos suficientes
    # Los filtros estrictos (ADX, Market Regime) se aplicarán en producción
    meta_labels = create_meta_labels(
        df.loc[temp_X.index],
        primary_predictions,
        lookforward_periods=20,          # Valores originales que funcionaron bien
        profit_target_atr_mult=2.0,     # R:R 1.33:1 para entrenamiento
        loss_limit_atr_mult=1.5          # Más permisivo durante entrenamiento
    )
    meta_labels.name = 'meta_label'

    # 4. Combine features and labels
    df_combined = df_features_aligned.join(meta_labels)

    # 5. Clean data and select final features
    # First, drop rows where we couldn't generate a meta-label (i.e., primary model said HOLD)
    df_combined.dropna(subset=['meta_label'], inplace=True)

    df_clean = feature_engineer.prepare_for_model(df_combined, target_column='meta_label')

    if df_clean.empty:
        logger.warning("No signals from primary model to create meta-labels. Training will be skipped.")
        return pd.DataFrame(), pd.Series()

    X = df_clean[feature_engineer.get_feature_importance_names()]
    y = df_clean['meta_label']

    # Ensure y is a 1D array
    if y.ndim > 1 and 'meta_label' in df_clean.columns:
        y = df_clean['meta_label']
    elif y.ndim > 1:
        y = y.iloc[:, 0]

    logger.info(f"Data preparation complete. Shape of X: {X.shape}, Shape of y: {y.shape}")

    return X, y


def get_historical_data_files() -> list:
    """Scans the historical data directory and returns a list of all individual file paths."""
    logger.info(f"Scanning for historical data files in '{HISTORICAL_DATA_DIR}'...")
    file_paths = []

    if not os.path.exists(HISTORICAL_DATA_DIR):
        logger.warning(f"Historical data directory '{HISTORICAL_DATA_DIR}' does not exist.")
        return file_paths

    for symbol_dir in os.listdir(HISTORICAL_DATA_DIR):
        symbol_path = os.path.join(HISTORICAL_DATA_DIR, symbol_dir)
        if os.path.isdir(symbol_path):
            for timeframe_file in os.listdir(symbol_path):
                if timeframe_file.endswith(('.csv', '.txt')):
                    file_paths.append(os.path.join(symbol_path, timeframe_file))
    logger.info(f"Found {len(file_paths)} historical data files to process.")
    return file_paths


def load_single_historical_data(file_path: str) -> pd.DataFrame:
    """Loads and processes a single historical data file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('"', '')

        df = pd.read_csv(io.StringIO(content), sep='\t')

        df.columns = df.columns.str.replace(r'[<>]', '', regex=True).str.strip()

        if 'TIME' in df.columns and 'DATE' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
        elif 'DATE' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d', errors='coerce')
        else:
            logger.warning(f"Skipping {file_path} due to missing 'DATE' or 'TIME' columns.")
            return pd.DataFrame()

        # Renombrar columnas principales
        df.rename(columns={
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'TICKVOL': 'volume'
        }, inplace=True)

        # Asegurar que VOL y SPREAD existan
        if 'VOL' not in df.columns:
            df['VOL'] = 0
        if 'SPREAD' not in df.columns:
            df['SPREAD'] = 0

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        return df
    except Exception as e:
        logger.error(f"Failed to load or process {file_path}: {e}")
        return pd.DataFrame()


def save_training_metadata(model_dir: str, symbol: str, timeframe: str, num_records: int, source: str):
    """Save metadata about the training session."""
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'trained_at': datetime.now().isoformat(),
        'num_records': num_records,
        'source': source,  # 'mt5' or 'local_file'
        'candles_used': config.retrain_candles if source == 'mt5' else num_records
    }

    metadata_path = os.path.join(model_dir, 'training_metadata.json')
    os.makedirs(model_dir, exist_ok=True)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training metadata saved to {metadata_path}")


def get_model_age(model_dir: str) -> int:
    """Get the age of the model in days."""
    metadata_path = os.path.join(model_dir, 'training_metadata.json')

    if not os.path.exists(metadata_path):
        return 9999  # Very old if no metadata

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        trained_at = datetime.fromisoformat(metadata['trained_at'])
        age_days = (datetime.now() - trained_at).days

        return age_days
    except Exception as e:
        logger.warning(f"Error reading model metadata: {e}")
        return 9999


async def train_from_mt5():
    """
    Train models using data downloaded from MT5.

    Validación de datos mínimos por timeframe:
    - Timeframes bajos (15m, 1h): Requieren más datos para captar patrones intraday
    - Timeframes altos (4h, 1d): Requieren menos datos absolutos pero suficientes para LSTM

    IMPORTANTE: Activos nuevos (<2 años) en timeframes altos (1d) tendrán menos datos disponibles.
    El bot opera principalmente en timeframes bajos y usa timeframes altos como confirmación.
    """
    logger.info("=" * 30 + " Training Models from MT5 Data " + "=" * 30)

    # Definir datos mínimos recomendados por timeframe
    # IMPORTANTE: Mínimo 2000 velas para garantizar calidad del LSTM
    # Con menos de 2000 velas, el modelo tiende a overfitting o predicciones aleatorias
    MINIMUM_CANDLES_BY_TIMEFRAME = {
        '15m': 2000,  # ~20 días - timeframe principal de operación
        '1h': 2000,   # ~83 días - confirmación importante
        '4h': 2000,   # ~333 días (~11 meses) - confirmación de tendencia
        '1d': 2000    # ~5.5 años - confirmación de largo plazo
    }

    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MT5. Cannot download data.")
        return

    try:
        symbols = config.trading_symbols
        timeframes = config.timeframes

        if not symbols or not timeframes:
            logger.error("No symbols or timeframes configured in .env file.")
            return

        logger.info(f"Will train models for {len(symbols)} symbols × {len(timeframes)} timeframes = {len(symbols) * len(timeframes)} models")
        logger.info(f"Requesting {config.retrain_candles} candles per symbol/timeframe")
        logger.info(f"Minimum data thresholds: {MINIMUM_CANDLES_BY_TIMEFRAME}")

        trained_count = 0
        failed_count = 0
        skipped_count = 0
        models_since_reconnect = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logger.info(f"\n--- Training {symbol} [{timeframe}] ---")

                    # Reconectar MT5 cada 5 modelos para evitar IPC errors
                    if models_since_reconnect >= 5:
                        logger.info("Reconnecting to MT5 to prevent IPC errors...")
                        shutdown_mt5()
                        import time
                        time.sleep(3)
                        if not initialize_mt5():
                            logger.error("Failed to reconnect to MT5. Aborting.")
                            return
                        models_since_reconnect = 0

                    # 1. Download data from MT5 with retry logic
                    df = None
                    for retry in range(3):  # Intentar hasta 3 veces
                        df = download_data_from_mt5(symbol, timeframe, config.retrain_candles)
                        if not df.empty:
                            break

                        if retry < 2:  # No esperar en el último intento
                            logger.warning(f"Download failed, retrying in 5 seconds... (attempt {retry + 1}/3)")
                            import time
                            time.sleep(5)
                            # Reconectar antes de reintentar
                            shutdown_mt5()
                            time.sleep(2)
                            if not initialize_mt5():
                                logger.error("Failed to reconnect to MT5 for retry.")
                                break

                    if df is None or df.empty:
                        logger.error(f"No data downloaded for {symbol} [{timeframe}] after 3 attempts. Skipping.")
                        failed_count += 1
                        continue

                    # Validar datos mínimos por timeframe
                    actual_candles = len(df)
                    minimum_required = MINIMUM_CANDLES_BY_TIMEFRAME.get(timeframe, 500)

                    if actual_candles < minimum_required:
                        logger.warning(
                            f"Insufficient data for {symbol} [{timeframe}]: "
                            f"Downloaded {actual_candles} candles, minimum required {minimum_required}. "
                            f"SKIPPING - Activo puede ser nuevo o timeframe muy alto para histórico disponible."
                        )
                        skipped_count += 1
                        continue
                    else:
                        logger.info(f"✓ Data validation passed: {actual_candles} candles (minimum: {minimum_required})")

                    # Delay después de descarga exitosa para no sobrecargar MT5
                    import time
                    time.sleep(2)

                    # 1.5. Save/update historical data to local CSV files
                    # This keeps local files synchronized with latest MT5 data
                    try:
                        # Create directory structure: historical_data/Symbol/
                        csv_dir = os.path.join(HISTORICAL_DATA_DIR, symbol)
                        os.makedirs(csv_dir, exist_ok=True)

                        # Generate filename: Symbol_TF.csv (e.g., "PainX 400_M1.csv")
                        tf_map = {
                            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                            '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'W1', '1M': 'MN1'
                        }
                        tf_code = tf_map.get(timeframe, timeframe.upper())
                        csv_filename = f"{symbol}_{tf_code}.csv"
                        csv_path = os.path.join(csv_dir, csv_filename)

                        # Save DataFrame to CSV (overwrite existing file)
                        df.to_csv(csv_path)
                        logger.success(f"Saved historical data to {csv_path} ({len(df)} records)")

                    except Exception as e:
                        logger.warning(f"Failed to save CSV file for {symbol} [{timeframe}]: {e}")
                        # Continue training even if CSV save fails

                    # 2. Prepare training data
                    X, y = prepare_training_data(df)
                    if X.empty or y.empty:
                        logger.error(f"Data preparation failed for {symbol} [{timeframe}]. Skipping.")
                        failed_count += 1
                        continue

                    # 3. Create sequences for LSTM
                    feature_engineer = FeatureEngineer()
                    X_seq, y_seq = feature_engineer.create_sequences(X, y)
                    if X_seq.shape[0] == 0:
                        logger.error(f"Not enough data to create sequences for {symbol} [{timeframe}]. Skipping.")
                        failed_count += 1
                        continue

                    # 4. Train Ensemble Model
                    logger.info("Training Ensemble model...")
                    ensemble_model = EnsembleModel()
                    ensemble_model.fit(X, y, X_seq, y_seq)
                    logger.success("Ensemble model trained successfully")

                    # 5. Convert timeframe for directory name
                    tf_map = {
                        '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                        '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'W1', '1M': 'MN1'
                    }
                    tf_code = tf_map.get(timeframe, timeframe.upper())

                    # 6. Save models with metadata
                    model_dir = os.path.join(MODEL_OUTPUT_DIR, symbol.replace(" ", "_"), f"{symbol.replace(' ', '_')}_{tf_code}")
                    logger.info(f"Saving models to '{model_dir}'...")
                    ensemble_model.save_all(model_dir)

                    # 7. Save training metadata
                    save_training_metadata(model_dir, symbol, timeframe, len(df), 'mt5')

                    trained_count += 1
                    models_since_reconnect += 1
                    logger.success(f"✅ Finished training {symbol} [{timeframe}] ({trained_count}/{len(symbols) * len(timeframes)})")

                except Exception as e:
                    logger.error(f"Error training {symbol} [{timeframe}]: {e}")
                    failed_count += 1
                    continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training complete:")
        logger.info(f"  ✅ {trained_count} models trained successfully")
        logger.info(f"  ⚠️  {skipped_count} models skipped (insufficient historical data)")
        logger.info(f"  ❌ {failed_count} models failed")
        logger.info(f"  📊 Total processed: {trained_count + skipped_count + failed_count}/{len(symbols) * len(timeframes)}")
        if skipped_count > 0:
            logger.warning(f"\nNOTA: {skipped_count} modelos fueron omitidos por datos insuficientes.")
            logger.warning(f"Esto es NORMAL para activos nuevos en timeframes altos (4h, 1d).")
            logger.warning(f"El bot operará en timeframes bajos (15m, 1h) donde hay más datos disponibles.")
        logger.info(f"{'=' * 80}\n")

    finally:
        shutdown_mt5()


async def train_from_local_files():
    """Train models using local historical data files (original behavior)."""
    logger.info("=" * 30 + " Training Models from Local Files " + "=" * 30)

    historical_files = get_historical_data_files()
    if not historical_files:
        logger.error("No historical data files found. Aborting.")
        return

    trained_count = 0
    failed_count = 0

    for file_path in historical_files:
        try:
            path_parts = Path(file_path).parts
            symbol = path_parts[-2]
            timeframe = Path(path_parts[-1]).stem
            logger.info(f"\n--- Training {symbol} from file: {timeframe} ---")

            # 1. Load data
            historical_data = load_single_historical_data(file_path)
            if historical_data.empty:
                logger.warning(f"No data loaded for {file_path}. Skipping.")
                failed_count += 1
                continue

            logger.success(f"Loaded {len(historical_data)} records.")

            # 2. Prepare data
            X, y = prepare_training_data(historical_data)
            if X.empty or y.empty:
                logger.error(f"Data preparation resulted in empty dataframes. Skipping.")
                failed_count += 1
                continue

            # 3. Create sequences for LSTM
            feature_engineer = FeatureEngineer()
            X_seq, y_seq = feature_engineer.create_sequences(X, y)
            if X_seq.shape[0] == 0:
                logger.error(f"Not enough data to create sequences for LSTM. Skipping.")
                failed_count += 1
                continue

            # 4. Train Ensemble Model
            logger.info(f"Training Ensemble model...")
            ensemble_model = EnsembleModel()
            ensemble_model.fit(X, y, X_seq, y_seq)
            logger.success(f"Ensemble model trained.")

            # 5. Save the models to a structured directory
            model_dir = os.path.join(MODEL_OUTPUT_DIR, symbol.replace(" ", "_"), timeframe)
            logger.info(f"Saving models to '{model_dir}'...")
            ensemble_model.save_all(model_dir)

            # 6. Save training metadata
            save_training_metadata(model_dir, symbol, timeframe, len(historical_data), 'local_file')

            trained_count += 1
            logger.success(f"✅ Finished training {symbol} from {timeframe}")

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {file_path}: {e}")
            failed_count += 1
            continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training complete: {trained_count} models trained, {failed_count} failed")
    logger.info(f"{'=' * 80}\n")


async def main(use_mt5: bool = False):
    """
    Main function to run the training process.

    Args:
        use_mt5: If True, download data from MT5. If False, use local files.
    """
    logger.add("logs/training.log", rotation="10 MB", level="INFO")

    if use_mt5:
        await train_from_mt5()
    else:
        await train_from_local_files()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train trading bot AI models')
    parser.add_argument(
        '--source',
        choices=['mt5', 'local'],
        default='local',
        help='Data source: mt5 (download from MT5) or local (use historical_data files)'
    )

    args = parser.parse_args()

    use_mt5 = (args.source == 'mt5')

    asyncio.run(main(use_mt5=use_mt5))
