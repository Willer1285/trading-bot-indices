"""
MT5 Market Data Manager
Manages real-time market data collection from MT5
"""

import asyncio
import io
import os
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from .mt5_connector import MT5Connector
from .timeframe_aggregator import TimeframeAggregator


class MT5MarketDataManager:
    """Manages real-time collection of market data from MT5"""

    def __init__(
        self,
        connector: MT5Connector,
        symbols: List[str],
        timeframes: List[str],
        update_interval: int = 60
    ):
        """
        Initialize MT5 Market Data Manager

        Args:
            connector: MT5Connector instance
            symbols: List of symbols to monitor (e.g., ['EURUSD', 'GBPUSD', 'XAUUSD'])
            timeframes: List of timeframes to track
            update_interval: Update interval in seconds
        """
        self.connector = connector
        self.symbols = symbols
        self.timeframes = timeframes
        self.update_interval = update_interval

        # Data storage
        self.market_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Timeframe aggregator
        self.aggregator = TimeframeAggregator()

        # Control flags
        self.is_running = False
        self._tasks: Set[asyncio.Task] = set()

        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize data storage structure"""
        for symbol in self.symbols:
            self.market_data[symbol] = {}
            for timeframe in self.timeframes:
                self.market_data[symbol][timeframe] = pd.DataFrame()

        logger.info(f"Initialized storage for {len(self.symbols)} symbols across {len(self.timeframes)} timeframes")

    async def start(self):
        """Start continuous data collection"""
        self.is_running = True
        logger.info("Starting MT5 Market Data Manager...")

        # Initial data fetch
        await self._fetch_initial_data()

        # Start update loop for each symbol
        for symbol in self.symbols:
            task = asyncio.create_task(self._update_loop(symbol))
            self._tasks.add(task)

        logger.info(f"MT5 Market Data Manager started for {len(self.symbols)} symbols")

    async def stop(self):
        """Stop data collection"""
        self.is_running = False
        logger.info("Stopping MT5 Market Data Manager...")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("MT5 Market Data Manager stopped")

    async def _fetch_initial_data(self):
        """Load initial historical data from local files (.csv or .txt)."""
        logger.info("Loading initial historical data from local files...")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Check for both .txt and .csv extensions
                txt_path = f"historical_data/{symbol}/{timeframe}.txt"
                csv_path = f"historical_data/{symbol}/{timeframe}.csv"
                file_path = None

                if os.path.exists(txt_path):
                    file_path = txt_path
                elif os.path.exists(csv_path):
                    file_path = csv_path

                if file_path:
                    try:
                        # Using the robust loading logic from train_models.py
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().replace('"', '')
                        
                        df = pd.read_csv(io.StringIO(content), sep='\t')
                        
                        df.columns = df.columns.str.replace(r'[<>]', '', regex=True).str.strip()
                        
                        if 'TIME' in df.columns and 'DATE' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
                        elif 'DATE' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d', errors='coerce')
                        else:
                            raise ValueError("Missing 'DATE' or 'TIME' columns.")

                        # Renombrar columnas principales
                        # Nota: Mantenemos VOL y SPREAD como features adicionales (esperadas por los modelos)
                        df.rename(columns={
                            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
                            'CLOSE': 'close', 'TICKVOL': 'volume'
                        }, inplace=True)

                        # Asegurar que VOL y SPREAD existan (son features requeridas por los modelos)
                        if 'VOL' not in df.columns:
                            df['VOL'] = 0  # Si no existe, rellenar con 0
                        if 'SPREAD' not in df.columns:
                            df['SPREAD'] = 0  # Si no existe, rellenar con 0

                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)

                        # Select necessary columns including VOL and SPREAD
                        required_cols = ['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']
                        df = df[required_cols]
                        df.dropna(inplace=True)
                        
                        self.market_data[symbol][timeframe] = df
                        logger.success(f"Loaded {len(df)} records for {symbol} [{timeframe}] from {file_path}")

                    except Exception as e:
                        logger.error(f"Error loading local data for {symbol} [{timeframe}] from {file_path}: {e}")
                        await self._fetch_and_store(symbol, timeframe, limit=500)
                else:
                    logger.warning(f"No local data file found for {symbol} [{timeframe}]. Will fetch from MT5.")
                    await self._fetch_and_store(symbol, timeframe, limit=500)

        logger.info("Initial data loading process completed.")

    async def _fetch_and_store(self, symbol: str, timeframe: str, limit: int = 100):
        """
        Fetch and store OHLCV data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of candles to fetch
        """
        try:
            df = await self.connector.fetch_ohlcv(symbol, timeframe, limit)

            if not df.empty:
                # Store or update data
                if self.market_data[symbol][timeframe].empty:
                    self.market_data[symbol][timeframe] = df
                else:
                    # Merge with existing data, avoiding duplicates
                    existing = self.market_data[symbol][timeframe]
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()

                    # Keep only recent data (e.g., last 1000 candles)
                    self.market_data[symbol][timeframe] = combined.tail(1000)

                logger.debug(f"Updated {symbol} {timeframe}: {len(self.market_data[symbol][timeframe])} candles")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")

    async def _update_loop(self, symbol: str):
        """
        Continuous update loop for a symbol

        Args:
            symbol: Symbol to update
        """
        while self.is_running:
            try:
                logger.debug(f"Updating market data for {symbol}...")
                # Fetch data for all timeframes
                for timeframe in self.timeframes:
                    await self._fetch_and_store(symbol, timeframe, limit=10)
                
                logger.debug(f"Data update for {symbol} complete.")
                # Wait before next update
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info(f"Update loop cancelled for {symbol}")
                break
            except Exception as e:
                logger.error(f"Error in update loop for {symbol}: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get latest market data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of recent candles (None for all)

        Returns:
            DataFrame with latest data
        """
        try:
            df = self.market_data.get(symbol, {}).get(timeframe, pd.DataFrame())

            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()

            if limit:
                return df.tail(limit)
            return df

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None
        """
        return self.connector.get_current_price(symbol)

    def get_multi_timeframe_data(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for all timeframes of a symbol

        Args:
            symbol: Trading symbol
            limit: Number of recent candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}
        for timeframe in self.timeframes:
            result[timeframe] = self.get_latest_data(symbol, timeframe, limit)

        return result

    def get_all_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all symbols

        Returns:
            Dictionary mapping symbol to current price
        """
        prices = {}
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price

        return prices

    def get_data_summary(self) -> Dict:
        """
        Get summary of collected data

        Returns:
            Summary statistics
        """
        summary = {
            'symbols': len(self.symbols),
            'timeframes': len(self.timeframes),
            'is_running': self.is_running,
            'data_points': {}
        }

        for symbol in self.symbols:
            summary['data_points'][symbol] = {}
            for timeframe in self.timeframes:
                df = self.market_data.get(symbol, {}).get(timeframe, pd.DataFrame())
                summary['data_points'][symbol][timeframe] = len(df)

        return summary
