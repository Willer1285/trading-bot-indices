"""
Database models and initialization
Modelos de base de datos para tracking de trades y configuración
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from loguru import logger

Base = declarative_base()


class Trade(Base):
    """Modelo para trades ejecutados"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp1 = Column(Float, nullable=False)
    tp2 = Column(Float, nullable=False)
    lot_size = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED_TP1, CLOSED_TP2, CLOSED_SL, CLOSED_BE, CLOSED_MANUAL
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    profit = Column(Float, default=0.0)
    mt5_ticket = Column(Integer, nullable=True)  # MT5 order ticket

    # Relación con eventos
    events = relationship("TradeEvent", back_populates="trade", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Trade {self.signal_id} {self.symbol} {self.signal_type} {self.status}>"


class TradeEvent(Base):
    """Modelo para eventos de un trade (TP1, TP2, BE, etc)"""
    __tablename__ = 'trade_events'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False)
    event_type = Column(String(30), nullable=False)  # OPENED, IN_PROFIT, TP1_HIT, TP2_HIT, SL_HIT, BE_ACTIVATED, TS_ACTIVATED
    price = Column(Float, nullable=True)
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relación con trade
    trade = relationship("Trade", back_populates="events")

    def __repr__(self):
        return f"<TradeEvent {self.event_type} for Trade {self.trade_id}>"


class MessageTemplate(Base):
    """Modelo para plantillas de mensajes editables"""
    __tablename__ = 'message_templates'

    id = Column(Integer, primary_key=True)
    message_type = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)  # Nombre descriptivo
    template = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True)
    variables = Column(Text, nullable=True)  # JSON con variables disponibles
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MessageTemplate {self.message_type} enabled={self.enabled}>"


class BotStatus(Base):
    """Modelo para el estado del bot"""
    __tablename__ = 'bot_status'

    id = Column(Integer, primary_key=True)
    status = Column(String(20), default='STOPPED')  # RUNNING, PAUSED, STOPPED
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    uptime_seconds = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<BotStatus {self.status}>"


class ManualNotification(Base):
    """Modelo para notificaciones enviadas manualmente"""
    __tablename__ = 'manual_notifications'

    id = Column(Integer, primary_key=True)
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<ManualNotification {self.id} sent={self.success}>"


class Database:
    """Clase principal para gestionar la base de datos"""

    def __init__(self, db_path: str = "trading_bot.db"):
        """
        Inicializa la base de datos

        Args:
            db_path: Ruta del archivo de base de datos SQLite
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = scoped_session(sessionmaker(bind=self.engine))

        # Crear todas las tablas
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {db_path}")

        # Inicializar datos por defecto
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Inicializa plantillas de mensajes y estado del bot por defecto"""
        session = self.SessionLocal()

        try:
            # Verificar si ya hay plantillas
            if session.query(MessageTemplate).count() == 0:
                default_templates = self._get_default_templates()
                for template_data in default_templates:
                    template = MessageTemplate(**template_data)
                    session.add(template)
                logger.info(f"Initialized {len(default_templates)} default message templates")

            # Verificar si ya hay estado del bot
            if session.query(BotStatus).count() == 0:
                bot_status = BotStatus(status='STOPPED')
                session.add(bot_status)
                logger.info("Initialized bot status")

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing defaults: {e}")
        finally:
            session.close()

    def _get_default_templates(self) -> List[dict]:
        """Retorna las plantillas de mensajes por defecto"""
        return [
            {
                'message_type': 'SIGNAL_GENERATED',
                'name': 'Señal Generada',
                'template': '''🎯 **{signal_type} SIGNAL** 🎯

**Símbolo:** {symbol}
**Timeframe:** {timeframe}
**Confianza:** {confidence}%

📊 **Precio de Entrada:** ${entry_price}
🛑 **Stop Loss:** ${sl}
🎯 **Take Profit:**
   TP1: ${tp1}
   TP2: ${tp2}

📈 **Risk/Reward:** {risk_reward}
💰 **Lote:** {lot_size}

💡 **Razón:** {reason}

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "timeframe", "confidence", "entry_price", "sl", "tp1", "tp2", "risk_reward", "lot_size", "reason", "timestamp"]'
            },
            {
                'message_type': 'TRADE_OPENED',
                'name': 'Operación Abierta',
                'template': '''✅ **OPERACIÓN ABIERTA** ✅

**{signal_type}** en **{symbol}**
**Ticket MT5:** {ticket}
**Precio:** ${price}
**Lote:** {lot_size}

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["signal_type", "symbol", "ticket", "price", "lot_size", "timestamp"]'
            },
            {
                'message_type': 'TRADE_IN_PROFIT',
                'name': 'Operación en Ganancia',
                'template': '''💚 **OPERACIÓN EN GANANCIA** 💚

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Ganancia:** ${profit} ({profit_pct}%)
**Precio Actual:** ${current_price}

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "profit", "profit_pct", "current_price", "timestamp"]'
            },
            {
                'message_type': 'TP1_HIT',
                'name': 'TP1 Alcanzado',
                'template': '''🎯 **TP1 ALCANZADO** 🎯

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Precio:** ${price}
**Ganancia:** ${profit}

¡Primera meta alcanzada! 🎉

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "price", "profit", "timestamp"]'
            },
            {
                'message_type': 'TP2_HIT',
                'name': 'TP2 Alcanzado',
                'template': '''🎯🎯 **TP2 ALCANZADO** 🎯🎯

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Precio:** ${price}
**Ganancia:** ${profit}

¡Meta final alcanzada! 🚀🎉

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "price", "profit", "timestamp"]'
            },
            {
                'message_type': 'SL_HIT',
                'name': 'Stop Loss Alcanzado',
                'template': '''🛑 **STOP LOSS ALCANZADO** 🛑

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Precio:** ${price}
**Pérdida:** ${loss}

Operación cerrada en pérdida.

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "price", "loss", "timestamp"]'
            },
            {
                'message_type': 'BREAK_EVEN_ACTIVATED',
                'name': 'Break Even Activado',
                'template': '''⚖️ **BREAK EVEN ACTIVADO** ⚖️

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Nuevo SL:** ${new_sl} (Break Even)

Operación protegida sin riesgo.

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "new_sl", "timestamp"]'
            },
            {
                'message_type': 'TRAILING_STOP_ACTIVATED',
                'name': 'Trailing Stop Activado',
                'template': '''📈 **TRAILING STOP ACTIVADO** 📈

**{symbol}** {signal_type}
**Ticket:** {ticket}
**Nuevo SL:** ${new_sl}

SL siguiendo el precio automáticamente.

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["symbol", "signal_type", "ticket", "new_sl", "timestamp"]'
            },
            {
                'message_type': 'ERROR_ALERT',
                'name': 'Alerta de Error',
                'template': '''⚠️ **ALERTA DEL BOT** ⚠️

**Error:** {error_message}
**Componente:** {component}

⏰ {timestamp}''',
                'enabled': True,
                'variables': '["error_message", "component", "timestamp"]'
            }
        ]

    def get_session(self):
        """Retorna una nueva sesión de base de datos"""
        return self.SessionLocal()

    def close(self):
        """Cierra la conexión a la base de datos"""
        self.SessionLocal.remove()
        logger.info("Database connection closed")


# Instancia global de la base de datos
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """
    Obtiene la instancia global de la base de datos (Singleton)

    Returns:
        Database: Instancia de la base de datos
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def init_database(db_path: str = "trading_bot.db") -> Database:
    """
    Inicializa la base de datos con una ruta específica

    Args:
        db_path: Ruta del archivo de base de datos

    Returns:
        Database: Instancia de la base de datos
    """
    global _db_instance
    _db_instance = Database(db_path)
    return _db_instance
