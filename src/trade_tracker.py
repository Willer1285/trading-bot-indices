"""
Trade Tracker
Sistema de monitoreo y tracking de operaciones en tiempo real
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import MetaTrader5 as mt5
from loguru import logger

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_interface.database import get_database, Trade, TradeEvent


class TradeTracker:
    """
    Rastrea y monitorea operaciones en tiempo real
    Detecta eventos: TP1, TP2, SL, Break Even, Trailing Stop, In Profit
    """

    def __init__(self, telegram_bot=None):
        """
        Inicializa el trade tracker

        Args:
            telegram_bot: Instancia del bot de Telegram para enviar notificaciones
        """
        self.db = get_database()
        self.telegram_bot = telegram_bot
        self.active_trades: Dict[str, Trade] = {}  # signal_id -> Trade object
        self.notified_events: Dict[str, set] = {}  # signal_id -> {event_types notificados}

        logger.info("Trade Tracker initialized")

    def register_trade_opened(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        entry_price: float,
        sl: float,
        tp1: float,
        tp2: float,
        lot_size: float,
        confidence: float,
        timeframe: str,
        mt5_ticket: Optional[int] = None
    ) -> Trade:
        """
        Registra una nueva operación abierta

        Args:
            signal_id: ID único de la señal
            symbol: Símbolo del activo
            signal_type: BUY o SELL
            entry_price: Precio de entrada
            sl: Stop Loss
            tp1: Take Profit 1
            tp2: Take Profit 2
            lot_size: Tamaño del lote
            confidence: Confianza del modelo
            timeframe: Timeframe de la señal
            mt5_ticket: Ticket de la orden en MT5

        Returns:
            Trade: Objeto del trade registrado
        """
        session = self.db.get_session()

        try:
            # Crear trade
            trade = Trade(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry_price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                lot_size=lot_size,
                confidence=confidence,
                timeframe=timeframe,
                status='OPEN',
                mt5_ticket=mt5_ticket
            )

            session.add(trade)

            # Crear evento de apertura
            event = TradeEvent(
                trade=trade,
                event_type='OPENED',
                price=entry_price,
                message=f'Trade opened at {entry_price}'
            )

            session.add(event)
            session.commit()

            # Agregar a trades activos
            self.active_trades[signal_id] = trade
            self.notified_events[signal_id] = {'OPENED'}

            logger.success(f"Trade registered: {signal_id} - {symbol} {signal_type} @ {entry_price}")

            # Enviar notificación de apertura
            self._send_notification('TRADE_OPENED', trade, entry_price)

            return trade

        except Exception as e:
            session.rollback()
            logger.error(f"Error registering trade: {e}")
            raise
        finally:
            session.close()

    def update_trade_monitoring(self):
        """
        Monitorea todas las operaciones activas y detecta eventos
        Debe ser llamado periódicamente (cada tick o cada segundo)
        """
        if not self.active_trades:
            return

        session = self.db.get_session()

        try:
            for signal_id in list(self.active_trades.keys()):
                # Obtener el trade fresco de la base de datos en esta sesión
                trade = session.query(Trade).filter_by(signal_id=signal_id).first()

                if not trade or trade.status != 'OPEN':
                    # Si el trade no existe o ya está cerrado, eliminarlo de activos
                    if signal_id in self.active_trades:
                        del self.active_trades[signal_id]
                    continue

                # Actualizar referencia en memoria
                self.active_trades[signal_id] = trade

                # Obtener precio actual desde MT5
                if trade.mt5_ticket:
                    current_price = self._get_current_price_from_mt5(trade.symbol, trade.mt5_ticket)
                else:
                    # Si no hay ticket, obtener último tick
                    tick = mt5.symbol_info_tick(trade.symbol)
                    if tick:
                        current_price = tick.bid if trade.signal_type == 'SELL' else tick.ask
                    else:
                        continue

                if not current_price:
                    continue

                # Detectar eventos
                self._check_trade_events(trade, current_price, session)

        except Exception as e:
            logger.error(f"Error in trade monitoring: {e}")
        finally:
            session.close()

    def _check_trade_events(self, trade: Trade, current_price: float, session):
        """
        Verifica y detecta eventos para un trade específico

        Args:
            trade: Trade a verificar
            current_price: Precio actual del activo
            session: Sesión de base de datos
        """
        signal_id = trade.signal_id
        notified = self.notified_events.get(signal_id, set())

        # Determinar si está en profit
        is_long = trade.signal_type == 'BUY'
        in_profit = (current_price > trade.entry_price) if is_long else (current_price < trade.entry_price)

        # 1. Detectar si entró en ganancia (primera vez)
        if in_profit and 'IN_PROFIT' not in notified:
            profit_pips = abs(current_price - trade.entry_price)
            profit_pct = (profit_pips / trade.entry_price) * 100

            event = TradeEvent(
                trade=trade,
                event_type='IN_PROFIT',
                price=current_price,
                message=f'Trade entered profit zone at {current_price}'
            )
            session.add(event)
            session.commit()

            self.notified_events[signal_id].add('IN_PROFIT')
            logger.info(f"Trade {signal_id} entered profit: {profit_pct:.2f}%")

            self._send_notification('TRADE_IN_PROFIT', trade, current_price, profit_pips, profit_pct)

        # 2. Detectar TP1 alcanzado
        if 'TP1_HIT' not in notified:
            tp1_hit = (current_price >= trade.tp1) if is_long else (current_price <= trade.tp1)

            if tp1_hit:
                profit = abs(trade.tp1 - trade.entry_price) * trade.lot_size * 100000  # Aproximación

                event = TradeEvent(
                    trade=trade,
                    event_type='TP1_HIT',
                    price=current_price,
                    message=f'TP1 reached at {current_price}'
                )
                session.add(event)
                session.commit()

                self.notified_events[signal_id].add('TP1_HIT')
                logger.success(f"Trade {signal_id} hit TP1 at {current_price}")

                self._send_notification('TP1_HIT', trade, current_price, profit)

        # 3. Detectar TP2 alcanzado
        if 'TP2_HIT' not in notified:
            tp2_hit = (current_price >= trade.tp2) if is_long else (current_price <= trade.tp2)

            if tp2_hit:
                profit = abs(trade.tp2 - trade.entry_price) * trade.lot_size * 100000  # Aproximación

                event = TradeEvent(
                    trade=trade,
                    event_type='TP2_HIT',
                    price=current_price,
                    message=f'TP2 reached at {current_price}'
                )
                session.add(event)
                session.commit()

                self.notified_events[signal_id].add('TP2_HIT')
                logger.success(f"Trade {signal_id} hit TP2 at {current_price}")

                self._send_notification('TP2_HIT', trade, current_price, profit)

                # Si alcanzó TP2, marcar como cerrado
                trade.status = 'CLOSED_TP2'
                trade.closed_at = datetime.utcnow()
                trade.profit = profit
                session.commit()

                # Remover de trades activos
                if signal_id in self.active_trades:
                    del self.active_trades[signal_id]

        # 4. Detectar SL alcanzado
        if 'SL_HIT' not in notified and trade.status == 'OPEN':
            sl_hit = (current_price <= trade.sl) if is_long else (current_price >= trade.sl)

            if sl_hit:
                loss = abs(trade.sl - trade.entry_price) * trade.lot_size * 100000  # Aproximación

                event = TradeEvent(
                    trade=trade,
                    event_type='SL_HIT',
                    price=current_price,
                    message=f'Stop Loss hit at {current_price}'
                )
                session.add(event)

                trade.status = 'CLOSED_SL'
                trade.closed_at = datetime.utcnow()
                trade.profit = -loss
                session.commit()

                self.notified_events[signal_id].add('SL_HIT')
                logger.warning(f"Trade {signal_id} hit SL at {current_price}")

                self._send_notification('SL_HIT', trade, current_price, loss)

                # Remover de trades activos
                if signal_id in self.active_trades:
                    del self.active_trades[signal_id]

    def register_break_even(self, signal_id: str, new_sl: float):
        """
        Registra cuando se activa Break Even

        Args:
            signal_id: ID de la señal
            new_sl: Nuevo Stop Loss (en break even)
        """
        if signal_id not in self.active_trades:
            return

        trade = self.active_trades[signal_id]
        session = self.db.get_session()

        try:
            event = TradeEvent(
                trade_id=trade.id,
                event_type='BE_ACTIVATED',
                price=new_sl,
                message=f'Break Even activated, SL moved to {new_sl}'
            )
            session.add(event)
            session.commit()

            self.notified_events[signal_id].add('BE_ACTIVATED')
            logger.info(f"Break Even activated for {signal_id}")

            self._send_notification('BREAK_EVEN_ACTIVATED', trade, new_sl)

        except Exception as e:
            session.rollback()
            logger.error(f"Error registering break even: {e}")
        finally:
            session.close()

    def register_trailing_stop(self, signal_id: str, new_sl: float):
        """
        Registra cuando se actualiza el Trailing Stop

        Args:
            signal_id: ID de la señal
            new_sl: Nuevo Stop Loss (trailing)
        """
        if signal_id not in self.active_trades:
            return

        trade = self.active_trades[signal_id]
        session = self.db.get_session()

        try:
            # Solo notificar la primera vez que se activa trailing stop
            if 'TS_ACTIVATED' not in self.notified_events.get(signal_id, set()):
                event = TradeEvent(
                    trade_id=trade.id,
                    event_type='TS_ACTIVATED',
                    price=new_sl,
                    message=f'Trailing Stop activated, SL at {new_sl}'
                )
                session.add(event)
                session.commit()

                self.notified_events[signal_id].add('TS_ACTIVATED')
                logger.info(f"Trailing Stop activated for {signal_id}")

                self._send_notification('TRAILING_STOP_ACTIVATED', trade, new_sl)

        except Exception as e:
            session.rollback()
            logger.error(f"Error registering trailing stop: {e}")
        finally:
            session.close()

    def close_trade(self, signal_id: str, close_price: float, profit: float, reason: str = 'MANUAL'):
        """
        Cierra un trade manualmente

        Args:
            signal_id: ID de la señal
            close_price: Precio de cierre
            profit: Ganancia/pérdida
            reason: Razón del cierre
        """
        if signal_id not in self.active_trades:
            logger.warning(f"Trade {signal_id} not found in active trades")
            return

        trade = self.active_trades[signal_id]
        session = self.db.get_session()

        try:
            trade.status = f'CLOSED_{reason}'
            trade.closed_at = datetime.utcnow()
            trade.profit = profit
            session.commit()

            logger.info(f"Trade {signal_id} closed manually at {close_price}")

            # Remover de trades activos
            del self.active_trades[signal_id]
            if signal_id in self.notified_events:
                del self.notified_events[signal_id]

        except Exception as e:
            session.rollback()
            logger.error(f"Error closing trade: {e}")
        finally:
            session.close()

    def _get_current_price_from_mt5(self, symbol: str, ticket: int) -> Optional[float]:
        """
        Obtiene el precio actual de una posición desde MT5

        Args:
            symbol: Símbolo del activo
            ticket: Ticket de la orden

        Returns:
            float: Precio actual o None si no se encuentra
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if positions and len(positions) > 0:
                position = positions[0]
                return position.price_current
        except Exception as e:
            logger.error(f"Error getting price from MT5: {e}")

        return None

    def _send_notification(self, message_type: str, trade: Trade, *args):
        """
        Envía una notificación via Telegram usando las plantillas configuradas

        Args:
            message_type: Tipo de mensaje
            trade: Trade relacionado
            *args: Argumentos adicionales específicos del mensaje
        """
        if not self.telegram_bot:
            return

        session = self.db.get_session()

        try:
            from web_interface.database import MessageTemplate

            # Obtener plantilla
            template_obj = session.query(MessageTemplate).filter_by(message_type=message_type).first()

            if not template_obj or not template_obj.enabled:
                return

            # Preparar variables según el tipo de mensaje
            variables = self._prepare_message_variables(message_type, trade, *args)

            # Formatear mensaje
            message = template_obj.template.format(**variables)

            # Enviar via Telegram
            import asyncio
            if self.telegram_bot:
                try:
                    asyncio.create_task(self.telegram_bot.send_message(message))
                except RuntimeError:
                    # Si no hay event loop activo, crear uno temporal
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.telegram_bot.send_message(message))
                    loop.close()

            logger.info(f"Notification sent: {message_type} for {trade.signal_id}")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
        finally:
            session.close()

    def _prepare_message_variables(self, message_type: str, trade: Trade, *args) -> dict:
        """
        Prepara las variables para formatear el mensaje

        Args:
            message_type: Tipo de mensaje
            trade: Trade relacionado
            *args: Argumentos adicionales

        Returns:
            dict: Variables para el template
        """
        base_vars = {
            'symbol': trade.symbol,
            'signal_type': trade.signal_type,
            'ticket': trade.mt5_ticket or 'N/A',
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

        if message_type == 'TRADE_OPENED':
            base_vars.update({
                'price': f'{args[0]:.5f}' if args else f'{trade.entry_price:.5f}',
                'lot_size': trade.lot_size
            })

        elif message_type == 'TRADE_IN_PROFIT':
            current_price, profit_pips, profit_pct = args if len(args) >= 3 else (0, 0, 0)
            base_vars.update({
                'current_price': f'{current_price:.5f}',
                'profit': f'{profit_pips:.5f}',
                'profit_pct': f'{profit_pct:.2f}'
            })

        elif message_type in ['TP1_HIT', 'TP2_HIT']:
            current_price, profit = args if len(args) >= 2 else (0, 0)
            base_vars.update({
                'price': f'{current_price:.5f}',
                'profit': f'{profit:.2f}'
            })

        elif message_type == 'SL_HIT':
            current_price, loss = args if len(args) >= 2 else (0, 0)
            base_vars.update({
                'price': f'{current_price:.5f}',
                'loss': f'{loss:.2f}'
            })

        elif message_type in ['BREAK_EVEN_ACTIVATED', 'TRAILING_STOP_ACTIVATED']:
            new_sl = args[0] if args else 0
            base_vars.update({
                'new_sl': f'{new_sl:.5f}'
            })

        return base_vars

    def get_active_trades_count(self) -> int:
        """Retorna el número de trades activos"""
        return len(self.active_trades)

    def load_active_trades(self):
        """Carga los trades activos desde la base de datos al iniciar"""
        session = self.db.get_session()

        try:
            active_trades = session.query(Trade).filter_by(status='OPEN').all()
            for trade in active_trades:
                self.active_trades[trade.signal_id] = trade
                # Cargar eventos ya notificados
                events = session.query(TradeEvent).filter_by(trade_id=trade.id).all()
                self.notified_events[trade.signal_id] = {event.event_type for event in events}

            logger.info(f"Loaded {len(active_trades)} active trades from database")

        except Exception as e:
            logger.error(f"Error loading active trades: {e}")
        finally:
            session.close()
