"""
Sistema de Watchdog para detectar y recuperar bloqueos del bot
Previene parálisis en producción monitoreando el heartbeat del sistema
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable
from loguru import logger


class BotWatchdog:
    """
    Watchdog que monitorea la salud del bot y detecta bloqueos.

    Funcionalidad:
    - Monitorea heartbeat del main loop
    - Detecta cuando el bot deja de responder
    - Puede ejecutar acciones de recuperación automáticas
    - Envía alertas cuando detecta problemas
    """

    def __init__(
        self,
        timeout_seconds: int = 180,  # 3 minutos sin heartbeat = alerta
        check_interval: int = 30,     # Verificar cada 30 segundos
        alert_callback: Optional[Callable] = None,
        recovery_callback: Optional[Callable] = None
    ):
        """
        Inicializa el watchdog.

        Args:
            timeout_seconds: Tiempo máximo sin heartbeat antes de alertar
            check_interval: Intervalo de verificación en segundos
            alert_callback: Función a llamar cuando se detecta un problema
            recovery_callback: Función a llamar para intentar recuperación
        """
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.recovery_callback = recovery_callback

        self.last_heartbeat: Optional[datetime] = None
        self.is_running = False
        self.watchdog_thread: Optional[threading.Thread] = None

        # Estadísticas
        self.total_checks = 0
        self.alerts_sent = 0
        self.recoveries_attempted = 0
        self.start_time: Optional[datetime] = None

        # Estado
        self.is_frozen = False
        self.freeze_detected_at: Optional[datetime] = None

        logger.info(f"Watchdog initialized: timeout={timeout_seconds}s, check_interval={check_interval}s")

    def start(self):
        """Inicia el watchdog en un thread separado."""
        if self.is_running:
            logger.warning("Watchdog already running")
            return

        self.is_running = True
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()

        self.watchdog_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="BotWatchdog"
        )
        self.watchdog_thread.start()

        logger.success("✅ Watchdog started and monitoring")

    def stop(self):
        """Detiene el watchdog."""
        if not self.is_running:
            return

        self.is_running = False

        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5)

        logger.info("Watchdog stopped")

    def heartbeat(self):
        """
        Registra un heartbeat del main loop.
        DEBE ser llamado regularmente por el main loop.
        """
        self.last_heartbeat = datetime.now()

        # Si estaba congelado y volvió a la vida, registrar recuperación
        if self.is_frozen:
            self.is_frozen = False
            freeze_duration = (datetime.now() - self.freeze_detected_at).total_seconds()
            logger.success(f"🔄 Bot recovered from freeze! Duration: {freeze_duration:.1f}s")
            self.freeze_detected_at = None

    def _monitor_loop(self):
        """Loop principal del watchdog que monitorea el heartbeat."""
        while self.is_running:
            try:
                self.total_checks += 1

                # Calcular tiempo desde el último heartbeat
                if self.last_heartbeat:
                    time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()

                    # Verificar si se excedió el timeout
                    if time_since_heartbeat > self.timeout_seconds:
                        if not self.is_frozen:
                            # Primera detección de congelamiento
                            self.is_frozen = True
                            self.freeze_detected_at = datetime.now()
                            self.alerts_sent += 1

                            logger.critical(
                                f"🚨 BOT FREEZE DETECTED! No heartbeat for {time_since_heartbeat:.1f}s "
                                f"(timeout: {self.timeout_seconds}s)"
                            )

                            # Ejecutar callback de alerta
                            if self.alert_callback:
                                try:
                                    self.alert_callback(time_since_heartbeat)
                                except Exception as e:
                                    logger.error(f"Error in alert callback: {e}")

                            # Intentar recuperación
                            if self.recovery_callback:
                                try:
                                    self.recoveries_attempted += 1
                                    logger.warning("Attempting automatic recovery...")
                                    self.recovery_callback()
                                except Exception as e:
                                    logger.error(f"Error in recovery callback: {e}")
                        else:
                            # Continúa congelado
                            freeze_duration = (datetime.now() - self.freeze_detected_at).total_seconds()
                            logger.critical(
                                f"🚨 Bot still frozen! Duration: {freeze_duration:.1f}s "
                                f"(last heartbeat: {time_since_heartbeat:.1f}s ago)"
                            )
                    else:
                        # Heartbeat normal
                        logger.debug(f"Watchdog check OK (last heartbeat: {time_since_heartbeat:.1f}s ago)")

                # Esperar antes del próximo chequeo
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in watchdog monitor loop: {e}")
                time.sleep(self.check_interval)

    def get_statistics(self) -> dict:
        """Retorna estadísticas del watchdog."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'total_checks': self.total_checks,
            'alerts_sent': self.alerts_sent,
            'recoveries_attempted': self.recoveries_attempted,
            'is_frozen': self.is_frozen,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'time_since_heartbeat': (datetime.now() - self.last_heartbeat).total_seconds() if self.last_heartbeat else None
        }

    def get_health_status(self) -> str:
        """Retorna el estado de salud del bot."""
        if not self.is_running:
            return "STOPPED"

        if self.is_frozen:
            return "FROZEN"

        if self.last_heartbeat:
            time_since = (datetime.now() - self.last_heartbeat).total_seconds()

            if time_since < self.timeout_seconds * 0.5:
                return "HEALTHY"
            elif time_since < self.timeout_seconds:
                return "WARNING"
            else:
                return "FROZEN"

        return "UNKNOWN"


class HeartbeatLogger:
    """
    Logger que registra heartbeats periódicos para verificar que el bot está vivo.
    Útil para debugging y monitoreo de logs.
    """

    def __init__(self, log_interval: int = 60):
        """
        Args:
            log_interval: Intervalo de logging en segundos
        """
        self.log_interval = log_interval
        self.last_log = datetime.now()
        self.heartbeat_count = 0

    def maybe_log_heartbeat(self, context: str = ""):
        """
        Loguea un heartbeat si ha pasado suficiente tiempo.

        Args:
            context: Contexto adicional para el log
        """
        self.heartbeat_count += 1

        time_since_last = (datetime.now() - self.last_log).total_seconds()

        if time_since_last >= self.log_interval:
            logger.info(
                f"💓 Heartbeat #{self.heartbeat_count} - Bot is alive "
                f"(last: {time_since_last:.0f}s ago){' - ' + context if context else ''}"
            )
            self.last_log = datetime.now()
