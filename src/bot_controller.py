"""
Bot Controller
Controla el estado del bot (Running/Paused/Stopped) y gestiona el ciclo de vida
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
import threading
import time

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_interface.database import get_database, BotStatus


class BotController:
    """
    Controla el estado y ejecución del bot de trading
    """

    def __init__(self):
        """Inicializa el controlador del bot"""
        self.db = get_database()
        self._status = 'STOPPED'
        self._started_at: Optional[datetime] = None
        self._paused_at: Optional[datetime] = None
        self._pause_total_seconds = 0  # Total de segundos en pausa
        self._lock = threading.Lock()

        # Cargar estado desde base de datos
        self._load_status()

        logger.info(f"Bot Controller initialized - Status: {self._status}")

    def _load_status(self):
        """Carga el estado del bot desde la base de datos"""
        session = self.db.get_session()

        try:
            bot_status = session.query(BotStatus).first()
            if bot_status:
                self._status = bot_status.status
                self._started_at = bot_status.started_at
                self._paused_at = bot_status.paused_at

                # Si estaba en pausa, calcular tiempo total de pausa
                if bot_status.status == 'PAUSED' and bot_status.paused_at:
                    self._pause_total_seconds = (datetime.utcnow() - bot_status.paused_at).total_seconds()

        except Exception as e:
            logger.error(f"Error loading bot status: {e}")
        finally:
            session.close()

    def _save_status(self):
        """Guarda el estado actual en la base de datos"""
        session = self.db.get_session()

        try:
            bot_status = session.query(BotStatus).first()

            if not bot_status:
                bot_status = BotStatus()
                session.add(bot_status)

            bot_status.status = self._status
            bot_status.started_at = self._started_at
            bot_status.paused_at = self._paused_at
            bot_status.stopped_at = datetime.utcnow() if self._status == 'STOPPED' else None
            bot_status.uptime_seconds = self.get_uptime_seconds()

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving bot status: {e}")
        finally:
            session.close()

    def start(self) -> bool:
        """
        Inicia el bot

        Returns:
            bool: True si se inició correctamente
        """
        with self._lock:
            if self._status == 'RUNNING':
                logger.warning("Bot is already running")
                return False

            self._status = 'RUNNING'
            self._started_at = datetime.utcnow()
            self._paused_at = None
            self._pause_total_seconds = 0

            self._save_status()
            logger.success("🚀 Bot STARTED")

            return True

    def pause(self) -> bool:
        """
        Pausa el bot (detiene generación de nuevas señales pero mantiene el monitoreo)

        Returns:
            bool: True si se pausó correctamente
        """
        with self._lock:
            if self._status != 'RUNNING':
                logger.warning(f"Cannot pause bot, current status: {self._status}")
                return False

            self._status = 'PAUSED'
            self._paused_at = datetime.utcnow()

            self._save_status()
            logger.warning("⏸️  Bot PAUSED")

            return True

    def resume(self) -> bool:
        """
        Reanuda el bot desde pausa

        Returns:
            bool: True si se reanudó correctamente
        """
        with self._lock:
            if self._status != 'PAUSED':
                logger.warning(f"Cannot resume bot, current status: {self._status}")
                return False

            # Calcular tiempo total en pausa
            if self._paused_at:
                pause_duration = (datetime.utcnow() - self._paused_at).total_seconds()
                self._pause_total_seconds += pause_duration

            self._status = 'RUNNING'
            self._paused_at = None

            self._save_status()
            logger.success("▶️  Bot RESUMED")

            return True

    def stop(self) -> bool:
        """
        Detiene el bot completamente

        Returns:
            bool: True si se detuvo correctamente
        """
        with self._lock:
            if self._status == 'STOPPED':
                logger.warning("Bot is already stopped")
                return False

            self._status = 'STOPPED'

            self._save_status()
            logger.success("⏹️  Bot STOPPED")

            # Resetear contadores
            self._started_at = None
            self._paused_at = None
            self._pause_total_seconds = 0

            return True

    def is_running(self) -> bool:
        """Verifica si el bot está en ejecución"""
        return self._status == 'RUNNING'

    def is_paused(self) -> bool:
        """Verifica si el bot está pausado"""
        return self._status == 'PAUSED'

    def is_stopped(self) -> bool:
        """Verifica si el bot está detenido"""
        return self._status == 'STOPPED'

    def get_status(self) -> str:
        """Retorna el estado actual del bot"""
        return self._status

    def get_uptime_seconds(self) -> int:
        """
        Calcula el tiempo de funcionamiento en segundos (excluyendo pausas)

        Returns:
            int: Segundos de uptime
        """
        if not self._started_at:
            return 0

        if self._status == 'STOPPED':
            return 0

        # Tiempo total desde el inicio
        total_elapsed = (datetime.utcnow() - self._started_at).total_seconds()

        # Restar pausas
        pause_time = self._pause_total_seconds

        # Si está actualmente en pausa, agregar el tiempo de la pausa actual
        if self._status == 'PAUSED' and self._paused_at:
            current_pause = (datetime.utcnow() - self._paused_at).total_seconds()
            pause_time += current_pause

        uptime = int(total_elapsed - pause_time)
        return max(0, uptime)  # Nunca negativo

    def get_uptime_formatted(self) -> str:
        """
        Retorna el uptime formateado como string legible

        Returns:
            str: Uptime formateado (ej: "2d 5h 30m 15s")
        """
        seconds = self.get_uptime_seconds()

        if seconds == 0:
            return "0s"

        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def get_status_info(self) -> dict:
        """
        Retorna información completa del estado del bot

        Returns:
            dict: Información del estado
        """
        return {
            'status': self._status,
            'uptime_seconds': self.get_uptime_seconds(),
            'uptime_formatted': self.get_uptime_formatted(),
            'started_at': self._started_at.isoformat() if self._started_at else None,
            'paused_at': self._paused_at.isoformat() if self._paused_at else None,
            'is_running': self.is_running(),
            'is_paused': self.is_paused(),
            'is_stopped': self.is_stopped()
        }

    def can_generate_signals(self) -> bool:
        """
        Verifica si el bot puede generar nuevas señales

        Returns:
            bool: True si puede generar señales (solo si está RUNNING)
        """
        return self._status == 'RUNNING'

    def can_monitor_trades(self) -> bool:
        """
        Verifica si el bot puede monitorear trades existentes

        Returns:
            bool: True si puede monitorear (RUNNING o PAUSED)
        """
        return self._status in ['RUNNING', 'PAUSED']


# Instancia global del controlador
_controller_instance: Optional[BotController] = None


def get_bot_controller() -> BotController:
    """
    Obtiene la instancia global del controlador del bot (Singleton)

    Returns:
        BotController: Instancia del controlador
    """
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = BotController()
    return _controller_instance
