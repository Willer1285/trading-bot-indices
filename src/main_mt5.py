"""
Main Application for MT5 Trading Bot
Trading Bot with AI-powered signal generation and automatic execution on MT5
"""

import asyncio
import os
from typing import Dict, List
from datetime import datetime
from loguru import logger
import signal as sys_signal

from config import config
from utils.logger import setup_logger
from utils.performance_tracker import PerformanceTracker

from data_collector.mt5_connector import MT5Connector, MT5OrderExecutor
from data_collector.mt5_market_data_manager import MT5MarketDataManager

from ai_engine.market_analyzer import MarketAnalyzer

from signal_generator.signal_generator import SignalGenerator
from signal_generator.signal_filter import SignalFilter
from signal_generator.risk_manager import RiskManager

from telegram_bot.telegram_bot import TelegramBot


class MT5TradingBot:
    """Main MT5 trading bot orchestrator with automatic execution"""

    def __init__(self):
        """Initialize MT5 Trading Bot"""
        # Setup logging
        setup_logger(
            log_level=config.log_level,
            log_file=config.log_file
        )

        logger.info("=" * 60)
        logger.info("AI MT5 Trading Bot Starting...")
        logger.info("=" * 60)

        # Performance tracking
        self.performance = PerformanceTracker()

        # Initialize components
        self.mt5_connector = None
        self.order_executor = None
        self.market_data_manager = None
        self.analyzer = None
        self.signal_generator = None
        self.telegram_bot = None

        # Control flags
        self.is_running = False
        self.analysis_interval = 60  # seconds
        self.auto_trading_enabled = config.mt5_auto_trading

        # Risk management
        self.lot_size = config.mt5_lot_size
        self.max_open_positions = config.mt5_max_open_positions
        self.break_even_activated = {} # Rastrea las operaciones con BE activado

        # Initialize all components - will be called in async initialize()

    async def _check_and_retrain_models(self):
        """Check model age and retrain automatically if needed."""
        import json
        from pathlib import Path
        from datetime import datetime
        import subprocess

        if not config.enable_auto_retrain:
            logger.info("Automatic retraining is disabled (ENABLE_AUTO_RETRAIN=false)")
            return

        logger.info("Checking model age for automatic retraining...")

        models_dir = Path(config.models_directory)
        needs_retrain = False

        if not models_dir.exists():
            logger.warning(f"Models directory '{models_dir}' does not exist. Will trigger retraining.")
            needs_retrain = True
        else:
            # Check age of all models
            oldest_model_age = 0

            for symbol in config.trading_symbols:
                symbol_safe = symbol.replace(" ", "_")
                symbol_dir = models_dir / symbol_safe

                if not symbol_dir.exists():
                    logger.warning(f"No models found for {symbol}. Will trigger retraining.")
                    needs_retrain = True
                    break

                # Check each timeframe
                for timeframe in config.timeframes:
                    tf_map = {
                        '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                        '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'W1', '1M': 'MN1'
                    }
                    tf_code = tf_map.get(timeframe, timeframe.upper())
                    model_dir = symbol_dir / f"{symbol_safe}_{tf_code}"

                    metadata_file = model_dir / 'training_metadata.json'

                    if not metadata_file.exists():
                        logger.warning(f"No metadata found for {symbol} [{timeframe}]. Will trigger retraining.")
                        needs_retrain = True
                        break

                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        trained_at = datetime.fromisoformat(metadata['trained_at'])
                        age_days = (datetime.now() - trained_at).days
                        oldest_model_age = max(oldest_model_age, age_days)

                        logger.info(f"Model for {symbol} [{timeframe}] is {age_days} days old")

                    except Exception as e:
                        logger.warning(f"Error reading metadata for {symbol} [{timeframe}]: {e}")
                        needs_retrain = True
                        break

                if needs_retrain:
                    break

            if not needs_retrain and oldest_model_age >= config.auto_retrain_days:
                logger.warning(f"Oldest model is {oldest_model_age} days old (threshold: {config.auto_retrain_days} days)")
                needs_retrain = True

        if needs_retrain:
            logger.info("=" * 80)
            logger.info("⚠️  AUTOMATIC RETRAINING TRIGGERED")
            logger.info(f"Will download {config.retrain_candles} candles from MT5 for each symbol/timeframe")
            logger.info(f"Symbols: {', '.join(config.trading_symbols)}")
            logger.info(f"Timeframes: {', '.join(config.timeframes)}")
            logger.info("This may take 15-30 minutes...")
            logger.info("=" * 80)

            try:
                # Call train_models.py with MT5 source
                result = subprocess.run(
                    ['python', 'train_models.py', '--source', 'mt5'],
                    cwd=Path(__file__).parent.parent,  # Root directory of project
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

                if result.returncode == 0:
                    logger.success("✅ Automatic retraining completed successfully!")
                    logger.info("Models have been updated with fresh data from MT5")
                else:
                    logger.error(f"Retraining failed with exit code {result.returncode}")
                    logger.error(f"STDERR: {result.stderr}")
                    raise Exception("Automatic retraining failed")

            except subprocess.TimeoutExpired:
                logger.error("Retraining timed out after 1 hour")
                raise Exception("Automatic retraining timed out")
            except Exception as e:
                logger.error(f"Error during automatic retraining: {e}")
                raise

        else:
            logger.success(f"✅ Models are up to date (oldest model: {oldest_model_age} days old)")

    async def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # MT5 connector
            logger.info("Initializing MT5 connector...")
            self.mt5_connector = MT5Connector(
                login=config.mt5_login,
                password=config.mt5_password,
                server=config.mt5_server,
                path=config.mt5_path
            )

            if not self.mt5_connector.is_connected:
                raise Exception("Failed to connect to MT5")

            # Order executor
            logger.info("Initializing order executor...")
            self.order_executor = MT5OrderExecutor(
                connector=self.mt5_connector,
                magic_number=config.mt5_magic_number
            )

            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if account_info:
                logger.info(f"MT5 Account: {account_info['login']}, "
                           f"Balance: {account_info['balance']} {account_info['currency']}, "
                           f"Server: {account_info['server']}")

            # Log all available symbols for debugging
            available_symbols = self.mt5_connector.get_available_symbols()
            logger.info(f"Available symbols from broker: {available_symbols}")

            # Market data manager
            logger.info("Initializing market data manager...")
            self.market_data_manager = MT5MarketDataManager(
                connector=self.mt5_connector,
                symbols=config.trading_symbols,
                timeframes=config.timeframes,
                update_interval=60
            )

            # AI Analyzer
            logger.info("Initializing AI analyzer...")
            self.analyzer = MarketAnalyzer(enable_training=False)

            # Check model age and retrain if necessary
            await self._check_and_retrain_models()

            # Try to load pre-trained models
            try:
                self.analyzer.load_models(config.models_directory)
                logger.info("Loaded pre-trained models")
            except Exception as e:
                logger.critical(f"Could not load models: {e}. The bot cannot function without trained models.")
                logger.critical("Please run train_models.py to train and save the models before starting the bot.")
                raise SystemExit("CRITICAL: AI Models not found. Exiting.")

            # Signal components
            logger.info("Initializing signal generator...")
            signal_filter = SignalFilter(
                max_signals_per_day=config.max_signals_per_day,
                max_signals_per_pair=config.max_signals_per_pair
            )

            risk_manager = RiskManager()

            self.signal_generator = SignalGenerator(
                analyzer=self.analyzer,
                signal_filter=signal_filter,
                risk_manager=risk_manager,
                min_confidence=config.confidence_threshold
            )

            # Telegram bot
            logger.info("Initializing Telegram bot...")
            self.telegram_bot = TelegramBot(
                bot_token=config.telegram_bot_token,
                channel_id=config.telegram_channel_id,
                enable_charts=config.telegram_include_charts
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    async def start(self):
        """Start the trading bot"""
        try:
            self.is_running = True

            logger.info("Starting MT5 Trading Bot...")

            # Initialize all components (async)
            await self._initialize_components()

            # Test Telegram connection
            if self.telegram_bot:
                telegram_ok = await self.telegram_bot.test_connection()
                if not telegram_ok:
                    logger.warning("Telegram connection test failed, but continuing...")

            # Get account info
            account_info = self.mt5_connector.get_account_info()

            # Send startup message
            await self.telegram_bot.send_message(
                f"🚀 **AI MT5 Trading Bot Started**\n\n"
                f"**Account:** {account_info['login']}\n"
                f"**Balance:** {account_info['balance']} {account_info['currency']}\n"
                f"**Server:** {account_info['server']}\n"
                f"**Leverage:** 1:{account_info['leverage']}\n\n"
                f"Monitoring {len(config.trading_symbols)} symbols: {', '.join(config.trading_symbols)}\n"
                f"Timeframes: {', '.join(config.timeframes)}\n\n"
                f"**Auto-Trading:** {'✅ ENABLED' if self.auto_trading_enabled else '❌ DISABLED'}\n"
                f"**Lot Size:** {self.lot_size} lots\n"
                f"**Max Positions:** {self.max_open_positions}\n"
                f"**Min Confidence:** {config.confidence_threshold:.0%}"
            )

            # Start market data collection
            logger.info("Starting market data collection...")
            await self.market_data_manager.start()

            # Wait for initial data
            logger.info("Waiting for initial data collection...")
            await asyncio.sleep(30)

            # Start main loop
            logger.info("Starting analysis loop...")
            await self._run_main_loop()

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping MT5 Trading Bot...")

        self.is_running = False

        # Stop market data collection
        if self.market_data_manager:
            await self.market_data_manager.stop()

        # Close MT5 connection
        if self.mt5_connector:
            self.mt5_connector.shutdown()

        # Send shutdown message
        if self.telegram_bot:
            account_info = self.mt5_connector.get_account_info()
            await self.telegram_bot.send_message(
                f"🛑 **AI MT5 Trading Bot Stopped**\n\n"
                f"**Final Balance:** {account_info['balance'] if account_info else 'N/A'}\n"
                f"**Total Signals:** {self.performance.signals_generated}\n"
                f"**Uptime:** {self.performance.get_statistics()['uptime_hours']:.2f} hours"
            )

        logger.info("MT5 Trading Bot stopped")

    async def _run_main_loop(self):
        """Main analysis and signal generation loop"""
        logger.info("Main loop started")

        while self.is_running:
            try:
                logger.info("=" * 30 + " Starting New Analysis Cycle " + "=" * 30)
                # Check MT5 connection
                if not self.mt5_connector.check_connection():
                    logger.error("MT5 connection lost!")
                    await self.telegram_bot.send_error_alert(
                        "MT5 connection lost!",
                        "Attempting to reconnect..."
                    )
                    await asyncio.sleep(10)
                    continue

                # Gestionar posiciones abiertas (BE y TS)
                await self._manage_open_positions()

                # Analizar todos los símbolos para nuevas señales
                for symbol in config.trading_symbols:
                    await self._analyze_and_execute(symbol)

                logger.info("=" * 30 + " Analysis Cycle Completed " + "=" * 32)
                logger.info(f"Waiting for {self.analysis_interval} seconds until next cycle...")
                # Wait before next iteration
                await asyncio.sleep(self.analysis_interval)

                # Periodic tasks (every hour)
                if datetime.utcnow().minute == 0:
                    await self._run_periodic_tasks()

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.performance.record_error("main_loop", str(e))
                await asyncio.sleep(10)

    async def _analyze_and_execute(self, symbol: str):
        """
        Analyze market and execute trade automatically

        Args:
            symbol: Trading symbol to analyze
        """
        try:
            # VERIFICACIÓN INICIAL: Si ya hay una posición abierta para este símbolo, no se analiza.
            # Esto previene señales duplicadas mientras una operación está activa.
            if self.order_executor.get_open_positions(symbol):
                logger.info(f"Skipping analysis for {symbol}: An open position already exists.")
                return

            start_time = datetime.utcnow()

            # Get multi-timeframe data
            mtf_data = self.market_data_manager.get_multi_timeframe_data(symbol, limit=200)

            if not mtf_data or all(df.empty for df in mtf_data.values()):
                logger.debug(f"{symbol}: No data available yet")
                return

            # Se enriquece cada DataFrame de timeframe con características técnicas.
            # Esto es crucial para que el RiskManager tenga acceso a la columna 'atr'.
            enriched_mtf_data = {
                tf: self.analyzer.feature_engineer.extract_features(df)
                for tf, df in mtf_data.items()
            }

            logger.info(f"Analyzing {symbol}...")
            # Analizar todos los timeframes usando los datos ya enriquecidos.
            analyses = self.analyzer.analyze_multi_timeframe(enriched_mtf_data, symbol)

            # Get current price
            current_price = self.market_data_manager.get_current_price(symbol)

            if not current_price:
                logger.warning(f"{symbol}: Could not get current price")
                return

            # Generate signal
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                multi_tf_analyses=analyses,
                current_price=current_price,
                market_data=enriched_mtf_data  # Se pasan los datos enriquecidos
            )

            # If signal generated
            if signal:
                # La decisión de ejecutar se toma ANTES de notificar.
                # Se comprueba si el auto-trading está activado y si la señal pasa los filtros de ejecución.
                if self.auto_trading_enabled and self.signal_generator.signal_filter.should_execute(symbol, signal.signal_type):
                    logger.success(f"Signal for {symbol} passed filters. Executing and notifying.")
                    
                    # 1. Ejecutar la operación en MT5.
                    execution_result = await self._execute_signal(signal)
                    
                    # 2. Notificar a Telegram SOLO si la ejecución fue exitosa.
                    if execution_result:
                        logger.info(f"Sending successful execution of {symbol} to Telegram.")
                        chart_data = {'data': mtf_data.get(signal.timeframe, None)}
                        await self.telegram_bot.send_signal(signal, chart_data)
                    else:
                        logger.warning(f"Execution for {symbol} failed. Signal will not be sent to Telegram.")

                else:
                    # Si el auto-trading está desactivado o los filtros no pasan, solo se registra internamente.
                    log_reason = "Auto-trading disabled" if not self.auto_trading_enabled else "Execution limits reached"
                    logger.info(f"Signal for {symbol} generated but not executed: {log_reason}")
                    # Se registra la señal para estadísticas, aunque no se ejecute.
                    self.performance.record_signal(symbol, signal.signal_type)
            else:
                logger.info(f"Analysis for {symbol} complete. No signal generated (HOLD).")

            # Record analysis time
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Analysis for {symbol} took {elapsed:.2f} seconds.")
            self.performance.record_analysis_time(elapsed)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            self.performance.record_error("analysis", str(e), symbol)

    async def _execute_signal(self, signal) -> bool:
        """
        Execute signal automatically on MT5

        Args:
            signal: TradingSignal object
        
        Returns:
            True if execution was successful, False otherwise.
        """
        try:
            logger.info(f"Executing signal: {signal.signal_type} {signal.symbol}")

            # Check position limits BEFORE executing on MT5
            open_positions = self.order_executor.get_open_positions(signal.symbol)
            total_positions = len(self.order_executor.get_open_positions())

            if len(open_positions) > 0:
                logger.warning(f"{signal.symbol}: Already has open position, NOT executing on MT5")
                return False

            if total_positions >= self.max_open_positions:
                logger.warning(f"Max positions ({self.max_open_positions}) reached, NOT executing {signal.symbol} on MT5")
                return False

            # Get account info
            account_info = self.mt5_connector.get_account_info()

            if not account_info:
                logger.error("Could not get account info")
                return

            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)

            if not symbol_info:
                logger.error(f"Could not get symbol info for {signal.symbol}")
                return

            # Usar el tamaño de lote dinámico de la señal
            lot_size = signal.lot_size
            logger.info(f"Using dynamic lot size: {lot_size}")

            # Execute order
            # Se ha acortado el comentario para asegurar que el ATR siempre se guarde correctamente.
            # Formato: AI|{atr_value}
            comment = f"AI|{signal.atr_at_signal:.5f}"

            # Seleccionar el Take Profit según configuración (TP1 o TP2)
            if signal.take_profit_levels:
                if config.use_take_profit == "TP1":
                    initial_tp = signal.take_profit_levels[0]  # TP1
                    logger.info(f"Using TP1: {initial_tp}")
                elif config.use_take_profit == "TP2":
                    initial_tp = signal.take_profit_levels[1] if len(signal.take_profit_levels) > 1 else signal.take_profit_levels[0]  # TP2 o TP1 si no hay TP2
                    logger.info(f"Using TP2: {initial_tp}")
                else:
                    logger.warning(f"Invalid USE_TAKE_PROFIT value: {config.use_take_profit}. Using TP1 as default.")
                    initial_tp = signal.take_profit_levels[0]
            else:
                initial_tp = None

            result = self.order_executor.execute_market_order(
                symbol=signal.symbol,
                order_type=signal.signal_type,
                volume=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=initial_tp,
                comment=comment
            )

            if result:
                # Log execution confirmation internally (NOT sent to Telegram)
                logger.info(
                    f"✅ ORDER EXECUTED - "
                    f"Ticket: {result['ticket']}, "
                    f"Symbol: {result['symbol']}, "
                    f"Type: {result['type']}, "
                    f"Volume: {result['volume']} lots, "
                    f"Entry: {result['price']}, "
                    f"SL: {result['sl']}, "
                    f"TP: {result['tp']}, "
                    f"Time: {result['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # Store execution info for internal tracking
                self.performance.record_signal(signal.symbol, signal.signal_type)
                return True

            else:
                # Only log error internally (NOT sent to Telegram)
                logger.error(f"Failed to execute order for {signal.symbol}")
                self.performance.record_error("order_execution", f"Failed to execute {signal.signal_type}", signal.symbol)
                return False

        except Exception as e:
            # Only log error internally (NOT sent to Telegram)
            logger.error(f"Error executing signal: {e}")
            self.performance.record_error("signal_execution", str(e), signal.symbol)
            return False

    async def _manage_open_positions(self):
        """Gestiona las posiciones abiertas para aplicar Break Even y Trailing Stop."""
        if not config.enable_break_even and not config.enable_trailing_stop:
            return

        try:
            open_positions = self.order_executor.get_open_positions()
            if not open_positions:
                return

            logger.info(f"Gestionando {len(open_positions)} posiciones abiertas...")

            for position in open_positions:
                ticket = position['ticket']
                symbol = position['symbol']
                order_type = position['type']
                open_price = position['price_open']
                current_price = position['price_current']
                current_sl = position['sl']

                symbol_info = self.mt5_connector.get_symbol_info(symbol)
                if not symbol_info:
                    continue
                
                point_size = symbol_info['point']
                profit_points = 0
                if order_type == 'BUY':
                    profit_points = (current_price - open_price) / point_size
                else: # SELL
                    profit_points = (open_price - current_price) / point_size

                # Determinar si usar modo dinámico o fijo
                if config.enable_dynamic_risk:
                    # --- MODO DINÁMICO: Extraer ATR del comentario ---
                    comment = position.get('comment', '')
                    atr_at_signal = 0.0
                    if 'AI|' in comment:
                        try:
                            atr_at_signal = float(comment.split('|')[1])
                        except (ValueError, IndexError):
                            logger.warning(f"No se pudo extraer el ATR del comentario: '{comment}'")

                    if atr_at_signal <= 0:
                        logger.warning(f"ATR inválido ({atr_at_signal}) para la operación #{ticket}. No se puede gestionar dinámicamente.")
                        continue

                    # --- Break Even Dinámico (basado en ATR) ---
                    if config.enable_break_even and ticket not in self.break_even_activated:
                        trigger_distance = atr_at_signal * config.break_even_trigger_atr_multiplier
                        profit_lock_distance = atr_at_signal * config.break_even_profit_lock_atr_multiplier

                        profit_in_currency = position.get('profit', 0.0)

                        if (order_type == 'BUY' and current_price >= open_price + trigger_distance) or \
                           (order_type == 'SELL' and current_price <= open_price - trigger_distance):

                            new_sl = open_price + profit_lock_distance if order_type == 'BUY' else open_price - profit_lock_distance

                            if (order_type == 'BUY' and new_sl > current_sl) or \
                               (order_type == 'SELL' and new_sl < current_sl):
                                logger.info(f"Activando Break Even DINÁMICO para #{ticket} en {symbol}. Nuevo SL: {new_sl:.5f} (ATR: {atr_at_signal:.5f})")
                                modified = self.order_executor.modify_order(ticket, stop_loss=new_sl)
                                if modified:
                                    self.break_even_activated[ticket] = True
                                    await self.telegram_bot.send_break_even_notification(position, new_sl)
                            else:
                                self.break_even_activated[ticket] = True

                    # --- Trailing Stop Dinámico (basado en ATR) ---
                    if config.enable_trailing_stop:
                        trigger_distance = atr_at_signal * config.trailing_stop_trigger_atr_multiplier
                        trailing_distance = atr_at_signal * config.trailing_stop_distance_atr_multiplier

                        if (order_type == 'BUY' and current_price >= open_price + trigger_distance) or \
                           (order_type == 'SELL' and current_price <= open_price - trigger_distance):

                            new_sl = current_price - trailing_distance if order_type == 'BUY' else current_price + trailing_distance

                            if (order_type == 'BUY' and new_sl > current_sl) or \
                               (order_type == 'SELL' and new_sl < current_sl):
                                logger.info(f"Actualizando Trailing Stop DINÁMICO para #{ticket} en {symbol} a {new_sl:.5f} (ATR: {atr_at_signal:.5f})")
                                modified = self.order_executor.modify_order(ticket, stop_loss=new_sl)
                                if modified:
                                    await self.telegram_bot.send_trailing_stop_notification(position, new_sl)

                else:
                    # --- MODO FIJO: Usar valores en PUNTOS ---
                    point_value = 1.0  # Para índices sintéticos: 1 punto = 1.0 en el precio

                    # --- Break Even Fijo (valores en puntos) ---
                    if config.enable_break_even and ticket not in self.break_even_activated:
                        trigger_distance = config.fixed_break_even_trigger_points * point_value
                        profit_lock_distance = config.fixed_break_even_profit_lock_points * point_value

                        if (order_type == 'BUY' and current_price >= open_price + trigger_distance) or \
                           (order_type == 'SELL' and current_price <= open_price - trigger_distance):

                            new_sl = open_price + profit_lock_distance if order_type == 'BUY' else open_price - profit_lock_distance

                            if (order_type == 'BUY' and new_sl > current_sl) or \
                               (order_type == 'SELL' and new_sl < current_sl):
                                logger.info(f"Activando Break Even FIJO para #{ticket} en {symbol}. Nuevo SL: {new_sl:.5f} (Trigger: {config.fixed_break_even_trigger_points} pts)")
                                modified = self.order_executor.modify_order(ticket, stop_loss=new_sl)
                                if modified:
                                    self.break_even_activated[ticket] = True
                                    await self.telegram_bot.send_break_even_notification(position, new_sl)
                            else:
                                self.break_even_activated[ticket] = True

                    # --- Trailing Stop Fijo (valores en puntos) ---
                    if config.enable_trailing_stop:
                        trigger_distance = config.fixed_trailing_stop_trigger_points * point_value
                        trailing_distance = config.fixed_trailing_stop_distance_points * point_value

                        if (order_type == 'BUY' and current_price >= open_price + trigger_distance) or \
                           (order_type == 'SELL' and current_price <= open_price - trigger_distance):

                            new_sl = current_price - trailing_distance if order_type == 'BUY' else current_price + trailing_distance

                            if (order_type == 'BUY' and new_sl > current_sl) or \
                               (order_type == 'SELL' and new_sl < current_sl):
                                logger.info(f"Actualizando Trailing Stop FIJO para #{ticket} en {symbol} a {new_sl:.5f} (Distance: {config.fixed_trailing_stop_distance_points} pts)")
                                modified = self.order_executor.modify_order(ticket, stop_loss=new_sl)
                                if modified:
                                    await self.telegram_bot.send_trailing_stop_notification(position, new_sl)

        except Exception as e:
            logger.error(f"Error al gestionar las posiciones abiertas: {e}")


    async def _run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()

            # Get open positions
            positions = self.order_executor.get_open_positions()

            # Send status update
            health = self.performance.get_health_status()
            logger.info(f"Health check: {health['status']}")

            # Hourly update (internal logging only, optional Telegram)
            # Uncomment below if you want hourly updates in Telegram
            # await self.telegram_bot.send_message(
            #     f"📊 **Hourly Status Update**\n\n"
            #     f"**Balance:** {account_info['balance']} {account_info['currency']}\n"
            #     f"**Equity:** {account_info['equity']}\n"
            #     f"**Profit:** {account_info['profit']}\n"
            #     f"**Open Positions:** {len(positions)}\n"
            #     f"**Signals Today:** {self.signal_generator.get_signal_statistics()['total_signals']}\n"
            #     f"**Health:** {health['status']}"
            # )

            # Internal logging
            logger.info(
                f"Hourly status - Balance: {account_info['balance']}, "
                f"Equity: {account_info['equity']}, "
                f"Profit: {account_info['profit']}, "
                f"Open Positions: {len(positions)}, "
                f"Signals: {self.signal_generator.get_signal_statistics()['total_signals']}"
            )

            # Daily summary at midnight UTC
            if datetime.utcnow().hour == 0:
                summary = self.performance.get_daily_summary()
                await self.telegram_bot.send_daily_summary(summary)
                self.performance.reset_daily_metrics()

        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")


async def main():
    """Main entry point"""
    bot = MT5TradingBot()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    sys_signal.signal(sys_signal.SIGINT, signal_handler)
    sys_signal.signal(sys_signal.SIGTERM, signal_handler)

    # Start bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await bot.stop()
        raise


if __name__ == "__main__":
    # Run the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        raise
