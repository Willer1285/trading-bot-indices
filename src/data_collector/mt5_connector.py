"""
MT5 Connector
Connects to MetaTrader 5 and manages market data and order execution
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import asyncio


class MT5Connector:
    """Connector for MetaTrader 5 platform"""

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None
    ):
        """
        Initialize MT5 Connector

        Args:
            login: MT5 account login (optional if already logged in)
            password: MT5 account password (optional)
            server: MT5 server name (optional)
            path: Path to MT5 terminal (optional, auto-detected)
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.is_connected = False

        self._initialize_mt5()

    def _initialize_mt5(self):
        """Initialize connection to MT5"""
        try:
            # Initialize MT5
            if self.path:
                if not mt5.initialize(self.path):
                    logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                    return
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                    return

            logger.info("MT5 initialized successfully")

            # Login if credentials provided
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return

                logger.info(f"Logged in to MT5 account {self.login}")

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Could not get account info, but MT5 is initialized")
                self.is_connected = True
            else:
                logger.info(f"Connected to MT5 - Account: {account_info.login}, "
                           f"Balance: {account_info.balance}, "
                           f"Server: {account_info.server}")
                self.is_connected = True

        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            self.is_connected = False

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from MT5

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'GBPUSD', 'XAUUSD')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframes
            tf_map = {
                '1m': mt5.TIMEFRAME_M1,
                '5m': mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30,
                '1h': mt5.TIMEFRAME_H1,
                '4h': mt5.TIMEFRAME_H4,
                '1d': mt5.TIMEFRAME_D1,
            }

            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

            # Se ha modificado la obtención de datos para que sea más robusta.
            # En lugar de solicitar los últimos N registros, se solicita un rango de fechas.
            # Esto puede forzar a la terminal MT5 a descargar el historial si no está disponible localmente.
            date_to = datetime.now()
            
            # Se calcula un rango de fechas generoso para asegurar que se obtengan suficientes datos.
            days_to_request = 30  # Valor por defecto
            tf_lower = timeframe.lower()
            if 'd' in tf_lower:
                days_to_request = limit * 2  # Para timeframe diario, solicitar el doble de días como búfer.
            elif 'h' in tf_lower:
                hours = int(tf_lower.replace('h', ''))
                days_to_request = (limit * hours) // 24 * 3 + 15  # Búfer para fines de semana.
            elif 'm' in tf_lower:
                minutes = int(tf_lower.replace('m', ''))
                # Se asegura un mínimo de días para timeframes pequeños
                days_to_request = max((limit * minutes) // (24*60) * 3 + 10, 10)

            date_from = date_to - timedelta(days=days_to_request)

            # Agregar timeout de 30 segundos para evitar bloqueos
            rates = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: mt5.copy_rates_range(symbol, mt5_timeframe, date_from, date_to)
                ),
                timeout=30.0
            )

            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Rename columns
            df.rename(columns={
                'time': 'timestamp',
                'tick_volume': 'volume',
                'real_volume': 'VOL',
                'spread': 'SPREAD'
            }, inplace=True)

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

            # Asegurar que VOL y SPREAD existan (requeridas por los modelos)
            if 'VOL' not in df.columns:
                df['VOL'] = 0
            if 'SPREAD' not in df.columns:
                df['SPREAD'] = 0

            # Se seleccionan las columnas relevantes (incluyendo VOL y SPREAD)
            df = df[['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']]

            # Después de obtener un rango, nos aseguramos de usar solo los últimos `limit` registros
            # para mantener la consistencia con la lógica anterior.
            if len(df) > limit:
                df = df.tail(limit)

            # Add metadata
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")

            return df

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching OHLCV for {symbol} {timeframe} (>30s)")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        number_of_candles: int
    ) -> Optional[pd.DataFrame]:
        """Synchronously fetch historical data"""
        try:
            tf_map = {
                '1m': mt5.TIMEFRAME_M1, '5m': mt5.TIMEFRAME_M5, '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30, '1h': mt5.TIMEFRAME_H1, '4h': mt5.TIMEFRAME_H4,
                '1d': mt5.TIMEFRAME_D1,
            }
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, number_of_candles)

            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df.rename(columns={
                'time': 'timestamp',
                'tick_volume': 'volume',
                'real_volume': 'VOL',
                'spread': 'SPREAD'
            }, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

            # Asegurar que VOL y SPREAD existan (requeridas por los modelos)
            if 'VOL' not in df.columns:
                df['VOL'] = 0
            if 'SPREAD' not in df.columns:
                df['SPREAD'] = 0

            df = df[['open', 'high', 'low', 'close', 'volume', 'VOL', 'SPREAD']]
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {timeframe}: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol info or None
        """
        try:
            info = mt5.symbol_info(symbol)

            if info is None:
                logger.warning(f"Symbol {symbol} not found")
                return None

            return {
                'name': info.name,
                'description': info.description,
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'trade_contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'currency_base': info.currency_base,
                'currency_profit': info.currency_profit,
                'currency_margin': info.currency_margin,
            }

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price (bid) for symbol

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None
        """
        try:
            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                return None

            return tick.bid

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            account = mt5.account_info()

            if account is None:
                return None

            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit,
                'currency': account.currency,
                'server': account.server,
                'leverage': account.leverage,
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            symbols = mt5.symbols_get()

            if symbols is None:
                return []

            # Filter only visible and tradeable symbols
            return [
                s.name for s in symbols
                if s.visible and s.select
            ]

        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

    def shutdown(self):
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error shutting down MT5: {e}")

    def check_connection(self) -> bool:
        """Check if MT5 is connected"""
        try:
            terminal_info = mt5.terminal_info()
            return terminal_info is not None
        except:
            return False


class MT5OrderExecutor:
    """Executes trading orders on MT5"""

    def __init__(self, connector: MT5Connector, magic_number: int = 234000):
        """
        Initialize Order Executor

        Args:
            connector: MT5Connector instance
            magic_number: Magic number for identifying bot orders
        """
        self.connector = connector
        self.magic_number = magic_number

    def execute_market_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "AI Trading Bot"
    ) -> Optional[Dict]:
        """
        Execute market order

        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Lot size
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment

        Returns:
            Order result dict or None
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None

            # Enable symbol if not enabled
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to enable symbol {symbol}")
                    return None

            # Prepare request
            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if order_type == 'BUY' else mt5.symbol_info_tick(symbol).bid

            # Map order type
            order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL

            # Build request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            # Add SL/TP if provided
            if stop_loss is not None:
                request["sl"] = stop_loss

            if take_profit is not None:
                request["tp"] = take_profit

            # Send order
            result = mt5.order_send(request)

            if result is None:
                logger.error(f"Order send failed: {mt5.last_error()}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return None

            logger.info(
                f"Order executed: {order_type} {volume} lots {symbol} "
                f"at {price}, ticket: {result.order}"
            )

            return {
                'ticket': result.order,
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit,
                'time': datetime.now(),
                'comment': comment
            }

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None

    def modify_order(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify existing order SL/TP

        Args:
            ticket: Order ticket number
            stop_loss: New stop loss (optional)
            take_profit: New take profit (optional)

        Returns:
            True if successful
        """
        try:
            position = mt5.positions_get(ticket=ticket)

            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return False

            position = position[0]

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": position.symbol,
            }

            # Asegurarse de no borrar el TP o SL existente si no se proporciona uno nuevo
            request["sl"] = stop_loss if stop_loss is not None else position.sl
            request["tp"] = take_profit if take_profit is not None else position.tp

            # No hacer nada si los valores no cambian
            if request["sl"] == position.sl and request["tp"] == position.tp:
                return True # Considerado exitoso ya que no se necesita ningún cambio

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order {ticket} modified successfully")
                return True
            else:
                logger.error(f"Failed to modify order {ticket}: {result.comment}")
                return False

        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False

    def close_order(self, ticket: int) -> bool:
        """
        Close an open position

        Args:
            ticket: Position ticket

        Returns:
            True if closed successfully
        """
        try:
            position = mt5.positions_get(ticket=ticket)

            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return False

            position = position[0]

            # Determine close order type
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "Close by bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {result.comment}")
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open positions
        """
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_closed_positions_today(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get closed positions from today for tracking consecutive losses

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of closed positions with profit/loss info
        """
        try:
            # Get deals from today
            date_from = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            date_to = datetime.now()

            if symbol:
                deals = mt5.history_deals_get(date_from, date_to, group=symbol)
            else:
                deals = mt5.history_deals_get(date_from, date_to)

            if deals is None:
                return []

            closed_positions = []
            for deal in deals:
                # Only include OUT deals (exit positions) with our magic number
                if deal.entry == mt5.DEAL_ENTRY_OUT and deal.magic == self.magic_number:
                    # Determine if it was TP or SL based on profit
                    result = 'TP' if deal.profit > 0 else 'SL'
                    closed_positions.append({
                        'ticket': deal.ticket,
                        'symbol': deal.symbol,
                        'profit': deal.profit,
                        'result': result,
                        'time': datetime.fromtimestamp(deal.time)
                    })

            return closed_positions

        except Exception as e:
            logger.error(f"Error getting closed positions: {e}")
            return []

    def calculate_lot_size(
        self,
        symbol: str,
        risk_percent: float,
        stop_loss_pips: float,
        account_balance: float
    ) -> float:
        """
        Calculate lot size based on risk management

        Args:
            symbol: Trading symbol
            risk_percent: Risk percentage (e.g., 1.0 for 1%)
            stop_loss_pips: Stop loss in pips
            account_balance: Account balance

        Returns:
            Calculated lot size
        """
        try:
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                return 0.01  # Minimum lot

            # Calculate risk amount
            risk_amount = account_balance * (risk_percent / 100)

            # Get pip value
            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size

            # Calculate lot size
            pip_value = contract_size * point * 10  # 1 pip = 10 points for most pairs
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Round to volume step
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step

            # Ensure within limits
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))

            return lot_size

        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return 0.01
