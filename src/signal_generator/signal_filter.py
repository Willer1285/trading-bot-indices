"""
Signal Filter
Filters trading signals based on various criteria
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis


class SignalFilter:
    """Filters signals to ensure high quality"""

    def __init__(
        self,
        min_volume_24h: float = 1000000,
        max_spread_percent: float = 0.5,
        max_signals_per_day: int = 10,
        max_signals_per_pair: int = 3,
        liquidity_check: bool = True
    ):
        """
        Initialize Signal Filter

        Args:
            min_volume_24h: Minimum 24h volume in USD
            max_spread_percent: Maximum spread percentage
            max_signals_per_day: Maximum signals per day
            max_signals_per_pair: Maximum signals per pair per day
            liquidity_check: Enable liquidity checking
        """
        self.min_volume_24h = min_volume_24h
        self.max_spread_percent = max_spread_percent
        self.max_signals_per_day = max_signals_per_day
        self.max_signals_per_pair = max_signals_per_pair
        self.liquidity_check = liquidity_check

        # Signal tracking
        self.daily_signal_count = 0
        self.pair_signal_count: Dict[str, int] = {}
        self.last_reset = datetime.utcnow()
        self.recent_signals: List[Dict] = []

        # Tracking de operaciones cerradas para filtro de pérdidas consecutivas
        # Formato: {symbol: [{'result': 'SL'/'TP', 'timestamp': datetime}, ...]}
        self.closed_trades: Dict[str, List[Dict]] = {}

        # Parámetros para filtros avanzados
        self.max_consecutive_losses = 2  # Máximo 2 pérdidas consecutivas
        self.cooldown_hours = 2  # Horas de enfriamiento después de pérdidas consecutivas
        self.min_adx_for_trend = 25  # ADX mínimo para considerar tendencia fuerte
        self.max_momentum_percent = 2.0  # Máximo % de cambio de precio en últimas 5 velas
        self.sr_proximity_percent = 0.5  # % de proximidad requerida a S/R

        logger.info("Signal Filter initialized")

    def should_notify(
        self,
        symbol: str,
        signal_type: str,
        analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> bool:
        """
        Check if signal should be sent to Telegram (without execution limits)

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL
            analyses: Multi-timeframe analyses

        Returns:
            True if signal passes quality filters (for notification)
        """
        # Check timeframe confluence
        confluence_result, confluence_ratio = self._check_timeframe_confluence(analyses, signal_type)
        if not confluence_result:
            logger.warning(f"{symbol}: ❌ Insufficient timeframe confluence ({confluence_ratio:.1%} < 50%)")
            return False
        logger.info(f"{symbol}: ✅ Timeframe confluence passed ({confluence_ratio:.1%} >= 50%)")

        # Check trend alignment
        trend_result, trend_reason = self._check_trend_alignment(analyses, signal_type)
        if not trend_result:
            logger.warning(f"{symbol}: ❌ Trend not aligned - {trend_reason}")
            return False
        logger.info(f"{symbol}: ✅ Trend alignment passed")

        # Check volatility
        volatility_result, volatility_reason = self._check_volatility(analyses)
        if not volatility_result:
            logger.warning(f"{symbol}: ❌ Volatility too high - {volatility_reason}")
            return False
        logger.info(f"{symbol}: ✅ Volatility check passed")

        # ========== NUEVOS FILTROS AVANZADOS ==========

        # Filtro 1: Divergencias RSI/MACD
        divergence_result, divergence_reason = self._check_divergence(analyses, signal_type)
        if not divergence_result:
            logger.warning(f"{symbol}: ❌ Divergence filter - {divergence_reason}")
            return False
        logger.info(f"{symbol}: ✅ Divergence check passed - {divergence_reason}")

        # Filtro 2: Límite de pérdidas consecutivas
        consecutive_result, consecutive_reason = self._check_consecutive_losses(symbol)
        if not consecutive_result:
            logger.warning(f"{symbol}: ❌ Consecutive losses filter - {consecutive_reason}")
            return False
        logger.info(f"{symbol}: ✅ Consecutive losses check passed - {consecutive_reason}")

        # Filtro 5: Momentum/Velocidad del precio
        momentum_result, momentum_reason = self._check_momentum_filter(analyses, signal_type)
        if not momentum_result:
            logger.warning(f"{symbol}: ❌ Momentum filter - {momentum_reason}")
            return False
        logger.info(f"{symbol}: ✅ Momentum check passed - {momentum_reason}")

        # Filtro 6: Proximidad a Soporte/Resistencia
        sr_result, sr_reason = self._check_support_resistance_proximity(analyses, signal_type)
        if not sr_result:
            logger.warning(f"{symbol}: ❌ Support/Resistance filter - {sr_reason}")
            return False
        logger.info(f"{symbol}: ✅ S/R proximity check passed - {sr_reason}")

        # ==============================================

        # Check for conflicting signals
        if self._has_conflicting_recent_signal(symbol, signal_type):
            logger.warning(f"{symbol}: ❌ Conflicting recent signal")
            return False
        logger.info(f"{symbol}: ✅ No conflicting signals")

        # All quality checks passed (no execution limits checked here)
        logger.info(f"{symbol}: ✅✅✅ ALL QUALITY FILTERS PASSED (Including Advanced Filters) - Signal will be sent to Telegram!")
        return True

    def should_execute(
        self,
        symbol: str,
        signal_type: str
    ) -> bool:
        """
        Check if signal should be executed on MT5 (includes execution limits)

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL

        Returns:
            True if signal can be executed (within limits)
        """
        self._reset_daily_counts_if_needed()

        # Check daily limit
        if self.daily_signal_count >= self.max_signals_per_day:
            logger.warning(f"{symbol}: ❌ Daily execution limit reached: {self.daily_signal_count}/{self.max_signals_per_day} (Telegram notification sent)")
            return False

        # Check per-pair limit
        pair_count = self.pair_signal_count.get(symbol, 0)
        if pair_count >= self.max_signals_per_pair:
            logger.warning(f"{symbol}: ❌ Pair execution limit reached: {pair_count}/{self.max_signals_per_pair} (Telegram notification sent)")
            return False

        # All checks passed - can execute
        self._record_signal(symbol, signal_type)
        logger.info(f"{symbol}: ✅ EXECUTION APPROVED - Within limits!")
        return True

    def should_trade(
        self,
        symbol: str,
        signal_type: str,
        analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> bool:
        """
        Check if signal should be traded (legacy method - kept for compatibility)

        This method checks both quality filters AND execution limits.
        For new code, use should_notify() for Telegram and should_execute() for MT5.

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL
            analyses: Multi-timeframe analyses

        Returns:
            True if signal passes all filters
        """
        # Check quality first
        if not self.should_notify(symbol, signal_type, analyses):
            return False

        # Then check execution limits
        return self.should_execute(symbol, signal_type)

    def _check_timeframe_confluence(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, float]:
        """Check if multiple timeframes agree on signal"""
        valid_analyses = [a for a in analyses.values() if a is not None]

        if not valid_analyses:
            return False, 0.0

        # Count agreements
        agreements = sum(1 for a in valid_analyses if a.signal == signal_type)

        # Need at least 50% agreement (majority consensus)
        agreement_ratio = agreements / len(valid_analyses)

        return agreement_ratio >= 0.5, agreement_ratio

    def _check_trend_alignment(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Verifica alineación con tendencia superior usando:
        - ADX para fuerza de tendencia
        - Alineación de EMAs (9, 21, 50)
        - Precio vs SMA50
        """
        # Get higher timeframe analysis (4h, 1h, 1d)
        higher_tf_analysis = None
        selected_tf = None

        for tf in ['4h', '1h', '1d']:
            if tf in analyses and analyses[tf]:
                higher_tf_analysis = analyses[tf]
                selected_tf = tf
                break

        if not higher_tf_analysis:
            return True, "No higher TF available"  # Can't check, so allow

        indicators = higher_tf_analysis.indicators

        price = indicators.get('price', 0)
        sma_50 = indicators.get('sma_50', 0)
        ema_9 = indicators.get('ema_9', 0)
        ema_21 = indicators.get('ema_21', 0)
        ema_50 = indicators.get('ema_50', 0)
        adx = indicators.get('adx', 0)

        # Verificar ADX para fuerza de tendencia
        if adx > 0 and adx < self.min_adx_for_trend:
            return False, f"⚠️ Tendencia débil en {selected_tf}: ADX={adx:.1f} < {self.min_adx_for_trend} (mercado lateral, evitar)"

        # Para señales de COMPRA
        if signal_type == 'BUY':
            # Verificar alineación alcista de EMAs: EMA9 > EMA21 > EMA50
            ema_aligned = False
            if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
                ema_aligned = (ema_9 > ema_21 > ema_50)
                if not ema_aligned:
                    return False, f"⚠️ EMAs no alineadas para BUY en {selected_tf}: EMA9={ema_9:.5f}, EMA21={ema_21:.5f}, EMA50={ema_50:.5f}"

            # Verificar que el precio esté en tendencia alcista
            if sma_50 > 0:
                percent_from_sma = ((price - sma_50) / sma_50) * 100
                # Permitir BUY solo si precio está cerca o arriba de SMA50
                if price < sma_50 * 0.97:  # Más del 3% debajo de SMA50
                    return False, f"⚠️ Precio muy debajo de SMA50 en {selected_tf}: {percent_from_sma:+.2f}% (contra-tendencia bajista)"

                return True, f"✅ Tendencia alcista en {selected_tf}: ADX={adx:.1f}, EMAs alineadas, Precio {percent_from_sma:+.2f}% de SMA50"

        # Para señales de VENTA
        else:  # SELL
            # Verificar alineación bajista de EMAs: EMA9 < EMA21 < EMA50
            ema_aligned = False
            if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
                ema_aligned = (ema_9 < ema_21 < ema_50)
                if not ema_aligned:
                    return False, f"⚠️ EMAs no alineadas para SELL en {selected_tf}: EMA9={ema_9:.5f}, EMA21={ema_21:.5f}, EMA50={ema_50:.5f}"

            # Verificar que el precio esté en tendencia bajista
            if sma_50 > 0:
                percent_from_sma = ((price - sma_50) / sma_50) * 100
                # Permitir SELL solo si precio está cerca o debajo de SMA50
                if price > sma_50 * 1.03:  # Más del 3% arriba de SMA50
                    return False, f"⚠️ Precio muy arriba de SMA50 en {selected_tf}: {percent_from_sma:+.2f}% (contra-tendencia alcista)"

                return True, f"✅ Tendencia bajista en {selected_tf}: ADX={adx:.1f}, EMAs alineadas, Precio {percent_from_sma:+.2f}% de SMA50"

        return True, "Trend alignment check passed"

    def _check_volatility(self, analyses: Dict[str, Optional[MarketAnalysis]]) -> tuple[bool, str]:
        """Check if volatility is acceptable"""
        # Get 1h analysis
        hourly_analysis = analyses.get('1h')

        if not hourly_analysis:
            return True, "No 1h analysis available"  # Can't check, so allow

        indicators = hourly_analysis.indicators
        atr = indicators.get('atr', 0)
        price = indicators.get('price', 1)

        # ATR should be less than 5% of price
        atr_percent = (atr / price) * 100 if price > 0 else 0

        result = atr_percent < 5.0
        reason = f"ATR {atr_percent:.2f}% of price (need < 5%)"
        return result, reason

    def _has_conflicting_recent_signal(self, symbol: str, signal_type: str) -> bool:
        """Check for conflicting signals in recent history"""
        # Look back 1 hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        recent_for_symbol = [
            s for s in self.recent_signals
            if s['symbol'] == symbol and s['timestamp'] > cutoff_time
        ]

        for signal in recent_for_symbol:
            if signal['type'] != signal_type:
                return True  # Conflicting signal found

        return False

    def _record_signal(self, symbol: str, signal_type: str):
        """Record that a signal was generated"""
        self.daily_signal_count += 1
        self.pair_signal_count[symbol] = self.pair_signal_count.get(symbol, 0) + 1

        self.recent_signals.append({
            'symbol': symbol,
            'type': signal_type,
            'timestamp': datetime.utcnow()
        })

        # Keep only last 100 signals
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]

    def _reset_daily_counts_if_needed(self):
        """Reset daily counts at start of new day"""
        now = datetime.utcnow()

        if now.date() > self.last_reset.date():
            logger.info("Resetting daily signal counts")
            self.daily_signal_count = 0
            self.pair_signal_count = {}
            self.last_reset = now

    def _check_divergence(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Detecta divergencias entre RSI/MACD y precio

        Una divergencia ocurre cuando:
        - RSI indica sobrecompra (>70) pero precio continúa subiendo (divergencia alcista)
        - RSI indica sobreventa (<30) pero precio continúa bajando (divergencia bajista)

        Args:
            analyses: Multi-timeframe analyses
            signal_type: BUY or SELL

        Returns:
            (True si NO hay divergencia peligrosa, mensaje)
        """
        # Usar timeframe principal (15m o 1h)
        primary_analysis = None
        for tf in ['15m', '1h', '5m']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break

        if not primary_analysis:
            return True, "No hay datos para verificar divergencias"

        indicators = primary_analysis.indicators
        rsi_14 = indicators.get('rsi_14', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        # Para señales de VENTA
        if signal_type == 'SELL':
            # Verificar si RSI está en sobrecompra (>70)
            if rsi_14 > 70:
                # Verificar momentum del precio (usando ROC o momentum_5)
                momentum_5 = indicators.get('momentum_5', 0)

                # Si el precio tiene momentum ALCISTA fuerte (>1.5%) mientras RSI >70
                # es probable que haya divergencia alcista → NO VENDER aún
                if momentum_5 > 0.015:  # 1.5% de subida
                    return False, f"⚠️ Divergencia alcista detectada: RSI={rsi_14:.1f} (sobrecompra) pero precio sube {momentum_5*100:.2f}% - Probable continuación alcista"

        # Para señales de COMPRA
        elif signal_type == 'BUY':
            # Verificar si RSI está en sobreventa (<30)
            if rsi_14 < 30:
                # Verificar momentum del precio
                momentum_5 = indicators.get('momentum_5', 0)

                # Si el precio tiene momentum BAJISTA fuerte (<-1.5%) mientras RSI <30
                # es probable que haya divergencia bajista → NO COMPRAR aún
                if momentum_5 < -0.015:  # -1.5% de caída
                    return False, f"⚠️ Divergencia bajista detectada: RSI={rsi_14:.1f} (sobreventa) pero precio cae {momentum_5*100:.2f}% - Probable continuación bajista"

        return True, f"No hay divergencias peligrosas (RSI={rsi_14:.1f})"

    def _check_consecutive_losses(
        self,
        symbol: str
    ) -> tuple[bool, str]:
        """
        Verifica si el símbolo tiene demasiadas pérdidas consecutivas
        y aplica período de enfriamiento si es necesario

        Args:
            symbol: Trading symbol

        Returns:
            (True si se puede operar, mensaje)
        """
        if symbol not in self.closed_trades:
            return True, "Sin historial de operaciones cerradas"

        # Obtener operaciones cerradas del símbolo
        closed = self.closed_trades[symbol]

        if not closed:
            return True, "Sin operaciones cerradas recientes"

        # Contar pérdidas consecutivas (SL) desde la última operación
        consecutive_losses = 0
        last_loss_time = None

        for trade in reversed(closed):  # Revisar desde la más reciente
            if trade['result'] == 'SL':
                consecutive_losses += 1
                if last_loss_time is None:
                    last_loss_time = trade['timestamp']
            else:
                break  # Romper la racha de pérdidas

        # Si hay pérdidas consecutivas >= límite
        if consecutive_losses >= self.max_consecutive_losses:
            # Verificar período de enfriamiento
            if last_loss_time:
                time_since_loss = datetime.utcnow() - last_loss_time
                cooldown_delta = timedelta(hours=self.cooldown_hours)

                if time_since_loss < cooldown_delta:
                    remaining_minutes = int((cooldown_delta - time_since_loss).total_seconds() / 60)
                    return False, f"🔒 Período de enfriamiento activo: {consecutive_losses} SLs consecutivos. Esperar {remaining_minutes} min más"
                else:
                    # Período de enfriamiento completado, limpiar historial
                    logger.info(f"{symbol}: Período de enfriamiento completado. Reseteando contador de pérdidas.")
                    return True, f"Período de enfriamiento completado ({consecutive_losses} SLs previos)"

        return True, f"Pérdidas consecutivas: {consecutive_losses}/{self.max_consecutive_losses}"

    def _check_momentum_filter(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Filtra señales cuando hay momentum muy fuerte en dirección opuesta

        Por ejemplo:
        - Si queremos VENDER pero el precio está subiendo muy rápido (>2% en 5 velas)
        - Si queremos COMPRAR pero el precio está cayendo muy rápido (<-2% en 5 velas)

        Args:
            analyses: Multi-timeframe analyses
            signal_type: BUY or SELL

        Returns:
            (True si momentum no es peligroso, mensaje)
        """
        # Usar timeframe más corto para detectar momentum rápido
        primary_analysis = None
        for tf in ['5m', '15m', '1h']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break

        if not primary_analysis:
            return True, "No hay datos para verificar momentum"

        indicators = primary_analysis.indicators
        momentum_5 = indicators.get('momentum_5', 0)  # Cambio de precio en últimas 5 velas
        roc = indicators.get('roc', 0)  # Rate of Change

        # Para señales de VENTA
        if signal_type == 'SELL':
            # Si momentum es muy alcista (>2%), NO VENDER aún
            if momentum_5 > (self.max_momentum_percent / 100):
                return False, f"⚠️ Momentum alcista muy fuerte: +{momentum_5*100:.2f}% en últimas 5 velas - Esperar agotamiento"

        # Para señales de COMPRA
        elif signal_type == 'BUY':
            # Si momentum es muy bajista (<-2%), NO COMPRAR aún
            if momentum_5 < -(self.max_momentum_percent / 100):
                return False, f"⚠️ Momentum bajista muy fuerte: {momentum_5*100:.2f}% en últimas 5 velas - Esperar agotamiento"

        return True, f"Momentum aceptable: {momentum_5*100:.2f}%"

    def _check_support_resistance_proximity(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Verifica que las señales estén cerca de zonas de soporte/resistencia

        - SELL solo cerca de resistencias
        - BUY solo cerca de soportes
        - Evitar operar "en medio de la nada"

        Args:
            analyses: Multi-timeframe analyses
            signal_type: BUY or SELL

        Returns:
            (True si está cerca de nivel clave, mensaje)
        """
        # Usar timeframe mediano para S/R
        primary_analysis = None
        for tf in ['1h', '4h', '15m']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break

        if not primary_analysis:
            return True, "No hay datos para verificar S/R"

        # Obtener niveles de soporte y resistencia
        sr_levels = primary_analysis.support_resistance
        if not sr_levels:
            return True, "No hay niveles S/R disponibles"

        support = sr_levels.get('support', 0)
        resistance = sr_levels.get('resistance', 0)
        current_price = primary_analysis.indicators.get('price', 0)

        if support == 0 or resistance == 0 or current_price == 0:
            return True, "Niveles S/R no válidos"

        # Calcular distancias porcentuales
        distance_to_support = abs((current_price - support) / support) * 100
        distance_to_resistance = abs((current_price - resistance) / resistance) * 100

        # Para señales de VENTA
        if signal_type == 'SELL':
            # Debe estar cerca de resistencia (dentro del 0.5%)
            if distance_to_resistance <= self.sr_proximity_percent:
                return True, f"✅ Cerca de resistencia: {distance_to_resistance:.2f}% (R={resistance:.5f})"
            else:
                # Si está muy lejos de resistencia, rechazar
                if distance_to_resistance > 1.5:  # Más del 1.5% lejos
                    return False, f"⚠️ Muy lejos de resistencia: {distance_to_resistance:.2f}% (operar solo cerca de niveles clave)"

        # Para señales de COMPRA
        elif signal_type == 'BUY':
            # Debe estar cerca de soporte (dentro del 0.5%)
            if distance_to_support <= self.sr_proximity_percent:
                return True, f"✅ Cerca de soporte: {distance_to_support:.2f}% (S={support:.5f})"
            else:
                # Si está muy lejos de soporte, rechazar
                if distance_to_support > 1.5:  # Más del 1.5% lejos
                    return False, f"⚠️ Muy lejos de soporte: {distance_to_support:.2f}% (operar solo cerca de niveles clave)"

        return True, "Proximidad S/R aceptable"

    def record_closed_trade(self, symbol: str, result: str):
        """
        Registra una operación cerrada para tracking de pérdidas consecutivas

        Args:
            symbol: Trading symbol
            result: 'SL' (Stop Loss) o 'TP' (Take Profit)
        """
        if symbol not in self.closed_trades:
            self.closed_trades[symbol] = []

        self.closed_trades[symbol].append({
            'result': result,
            'timestamp': datetime.utcnow()
        })

        # Mantener solo las últimas 10 operaciones por símbolo
        if len(self.closed_trades[symbol]) > 10:
            self.closed_trades[symbol] = self.closed_trades[symbol][-10:]

        logger.info(f"Operación cerrada registrada: {symbol} -> {result}")

    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        return {
            'daily_signals': self.daily_signal_count,
            'max_daily': self.max_signals_per_day,
            'pair_counts': self.pair_signal_count,
            'recent_signals': len(self.recent_signals),
            'closed_trades_tracked': {k: len(v) for k, v in self.closed_trades.items()}
        }
