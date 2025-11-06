import asyncio
import sys
import os
import io
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import joblib
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
MODEL_OUTPUT_DIR = "models"
HISTORICAL_DATA_DIR = "historical_data"

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
    meta_labels = create_meta_labels(df.loc[temp_X.index], primary_predictions)
    
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

        df.rename(columns={
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'VOL': 'volume'
        }, inplace=True)
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Failed to load or process {file_path}: {e}")
        return pd.DataFrame()

async def main():
    """Main function to run the training process for each individual data file."""
    logger.add("logs/training.log", rotation="10 MB", level="INFO")
    logger.info("=" * 30 + " Starting AI Model Training (Per-File) " + "=" * 30)

    historical_files = get_historical_data_files()
    if not historical_files:
        logger.error("No historical data files found. Aborting.")
        return

    for file_path in historical_files:
        try:
            path_parts = Path(file_path).parts
            symbol = path_parts[-2]
            timeframe = Path(path_parts[-1]).stem
            logger.info(f"--- Starting training for Symbol: {symbol}, Timeframe: {timeframe} ---")

            # 1. Load data for the current file
            historical_data = load_single_historical_data(file_path)
            if historical_data.empty:
                logger.warning(f"No data loaded for {file_path}. Skipping.")
                continue
            
            logger.success(f"Loaded {len(historical_data)} records.")

            # 2. Prepare data
            X, y = prepare_training_data(historical_data)
            if X.empty or y.empty:
                logger.error(f"Data preparation resulted in empty dataframes. Skipping.")
                continue

            # 3. Create sequences for LSTM
            feature_engineer = FeatureEngineer()
            X_seq, y_seq = feature_engineer.create_sequences(X, y)
            if X_seq.shape[0] == 0:
                logger.error(f"Not enough data to create sequences for LSTM. Skipping.")
                continue

            # 4. Train Ensemble Model
            logger.info(f"Initializing and training Ensemble model...")
            ensemble_model = EnsembleModel()
            ensemble_model.fit(X, y, X_seq, y_seq)
            logger.success(f"Final Ensemble model trained.")

            # 5. Save the models to a structured directory
            model_dir = os.path.join(MODEL_OUTPUT_DIR, symbol.replace(" ", "_"), timeframe)
            logger.info(f"Saving models to '{model_dir}'...")
            ensemble_model.save_all(model_dir)
            
            logger.success(f"--- Finished training for Symbol: {symbol}, Timeframe: {timeframe} ---")

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {file_path}: {e}")
            continue

    logger.info("=" * 30 + " AI Model Training Finished for All Files " + "=" * 30)


if __name__ == "__main__":
    asyncio.run(main())
