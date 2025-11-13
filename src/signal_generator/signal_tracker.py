"""
Signal Tracker
Tracks sent signals to avoid duplicates and ensure optimal timing
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackedSignal:
    """Represents a tracked signal"""
    symbol: str
    signal_type: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    signal_id: str
    status: str = "active"  # active, closed_tp, closed_sl, expired

    def is_still_valid(self, current_price: float, max_distance_percent: float = 0.5) -> bool:
        """
        Check if signal is still valid based on current price

        Args:
            current_price: Current market price
            max_distance_percent: Maximum distance from entry price (%)

        Returns:
            True if signal is still valid
        """
        price_distance = abs((current_price - self.entry_price) / self.entry_price) * 100

        # If price moved more than max_distance_percent from entry, signal is invalid
        if price_distance > max_distance_percent:
            return False

        # Check if SL or TP was hit
        if self.signal_type == "BUY":
            if current_price <= self.stop_loss:
                self.status = "closed_sl"
                return False
            if current_price >= self.take_profit:
                self.status = "closed_tp"
                return False
        else:  # SELL
            if current_price >= self.stop_loss:
                self.status = "closed_sl"
                return False
            if current_price <= self.take_profit:
                self.status = "closed_tp"
                return False

        return True


class SignalTracker:
    """Tracks sent signals to avoid duplicates"""

    def __init__(
        self,
        min_time_between_signals: int = 30,  # minutes
        max_price_distance_percent: float = 0.5,  # 0.5% from entry
        max_signal_age: int = 120  # minutes
    ):
        """
        Initialize Signal Tracker

        Args:
            min_time_between_signals: Minimum time between same signals (minutes)
            max_price_distance_percent: Max price distance to consider duplicate (%)
            max_signal_age: Maximum age of signal before considering expired (minutes)
        """
        self.min_time_between_signals = min_time_between_signals
        self.max_price_distance_percent = max_price_distance_percent
        self.max_signal_age = max_signal_age

        # Store active signals: {symbol: [TrackedSignal]}
        self.active_signals: Dict[str, List[TrackedSignal]] = {}

        logger.info(
            f"Signal Tracker initialized - "
            f"Min time: {min_time_between_signals}m, "
            f"Max distance: {max_price_distance_percent}%, "
            f"Max age: {max_signal_age}m"
        )

    def should_send_signal(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        current_price: float,
        mt5_connector=None  # Optional MT5 connector to verify real positions
    ) -> bool:
        """
        Check if signal should be sent (not a duplicate)

        Args:
            symbol: Trading symbol
            signal_type: BUY or SELL
            entry_price: Proposed entry price
            current_price: Current market price
            mt5_connector: Optional MT5Connector to verify real open positions

        Returns:
            True if signal should be sent
        """
        # Clean up old signals first
        self._cleanup_old_signals()

        # VERIFICACIÓN CRÍTICA: Si tenemos acceso a MT5, verificar posiciones reales
        if mt5_connector is not None:
            try:
                # Verificar que el método existe antes de llamarlo
                if not hasattr(mt5_connector, 'get_open_positions'):
                    logger.warning(f"{symbol}: MT5 connector does not have get_open_positions method, using cache only")
                else:
                    open_positions = mt5_connector.get_open_positions()
                    has_open_position = any(
                        pos['symbol'] == symbol and pos['type'].upper() == signal_type
                        for pos in open_positions
                    )

                    if not has_open_position:
                        # No hay posición abierta en MT5, limpiar señales antiguas de este símbolo
                        if symbol in self.active_signals:
                            # Limpiar señales del mismo tipo que no tienen posición real
                            self.active_signals[symbol] = [
                                sig for sig in self.active_signals[symbol]
                                if sig.signal_type != signal_type
                            ]
                            if not self.active_signals[symbol]:
                                del self.active_signals[symbol]

                        logger.info(f"{symbol}: ✅ No open {signal_type} position in MT5, OK to send new signal")
                        return True
                    else:
                        logger.info(f"{symbol}: ℹ️ Open {signal_type} position exists in MT5, checking cache")
            except Exception as e:
                logger.warning(f"{symbol}: Could not verify MT5 positions: {e}, using cache only")

        # Get signals for this symbol
        if symbol not in self.active_signals:
            logger.info(f"{symbol}: No previous signals, OK to send")
            return True

        recent_signals = self.active_signals[symbol]
        now = datetime.utcnow()

        for tracked_signal in recent_signals:
            # Skip if different signal type
            if tracked_signal.signal_type != signal_type:
                continue

            # Check time since last signal
            time_diff = (now - tracked_signal.timestamp).total_seconds() / 60

            if time_diff < self.min_time_between_signals:
                # Check if signal is still valid (price hasn't moved too far)
                if tracked_signal.is_still_valid(current_price, self.max_price_distance_percent):
                    logger.warning(
                        f"{symbol}: ❌ Duplicate signal blocked - "
                        f"Same {signal_type} signal sent {time_diff:.1f}m ago "
                        f"(entry: {tracked_signal.entry_price:.2f}, current: {current_price:.2f})"
                    )
                    return False
                else:
                    # Previous signal is no longer valid (price moved or hit SL/TP)
                    logger.info(
                        f"{symbol}: Previous {signal_type} signal is no longer valid "
                        f"(status: {tracked_signal.status}), OK to send new signal"
                    )

        logger.info(f"{symbol}: No recent duplicate signals, OK to send")
        return True

    def track_signal(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_id: str
    ):
        """
        Track a sent signal

        Args:
            symbol: Trading symbol
            signal_type: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_id: Unique signal ID
        """
        tracked_signal = TrackedSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.utcnow(),
            signal_id=signal_id,
            status="active"
        )

        if symbol not in self.active_signals:
            self.active_signals[symbol] = []

        self.active_signals[symbol].append(tracked_signal)

        logger.info(
            f"{symbol}: Tracking {signal_type} signal "
            f"(ID: {signal_id}, Entry: {entry_price:.2f})"
        )

    def _cleanup_old_signals(self):
        """Remove expired signals"""
        now = datetime.utcnow()
        max_age_delta = timedelta(minutes=self.max_signal_age)

        for symbol in list(self.active_signals.keys()):
            # Filter out old signals
            self.active_signals[symbol] = [
                sig for sig in self.active_signals[symbol]
                if (now - sig.timestamp) < max_age_delta
            ]

            # Remove symbol key if no signals left
            if not self.active_signals[symbol]:
                del self.active_signals[symbol]

    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        total_signals = sum(len(signals) for signals in self.active_signals.values())

        return {
            'symbols_tracked': len(self.active_signals),
            'total_active_signals': total_signals,
            'signals_by_symbol': {
                symbol: len(signals)
                for symbol, signals in self.active_signals.items()
            }
        }
