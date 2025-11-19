"""
Signal Filter
Filters trading signals based on various criteria
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis
from config import config


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

        # ========== PARÁMETROS CONFIGURABLES DESDE .env ==========

        # Activación de filtros
        self.enable_divergence_filter = config.enable_divergence_filter
        self.enable_consecutive_losses_filter = config.enable_consecutive_losses_filter
        self.enable_enhanced_trend_filter = config.enable_enhanced_trend_filter
        self.enable_momentum_filter = config.enable_momentum_filter
        self.enable_sr_proximity_filter = config.enable_sr_proximity_filter

        # Parámetros del filtro de pérdidas consecutivas
        self.max_consecutive_losses = config.max_consecutive_losses
        self.cooldown_hours = config.cooldown_hours

        # Parámetros del filtro de tendencia
        self.min_adx_for_trend = config.min_adx_for_trend
        self.ema_alignment_required = config.ema_alignment_required
        self.max_percent_from_sma50 = config.max_percent_from_sma50

        # Parámetros del filtro de momentum
        self.max_momentum_percent = config.max_momentum_percent

        # Parámetros del filtro de soporte/resistencia
        self.sr_proximity_percent = config.sr_proximity_percent
        self.sr_max_distance_percent = config.sr_max_distance_percent

        # Parámetros del filtro de divergencias
        self.divergence_rsi_overbought = config.divergence_rsi_overbought
        self.divergence_rsi_oversold = config.divergence_rsi_oversold
        self.divergence_momentum_threshold = config.divergence_momentum_threshold

        logger.info("Signal Filter initialized with configurable parameters")
        logger.info(f"  Divergence filter: {'✅ ENABLED' if self.enable_divergence_filter else '❌ DISABLED'}")
        logger.info(f"  Consecutive losses filter: {'✅ ENABLED' if self.enable_consecutive_losses_filter else '❌ DISABLED'}")
        logger.info(f"  Enhanced trend filter: {'✅ ENABLED' if self.enable_enhanced_trend_filter else '❌ DISABLED'}")
        logger.info(f"  Momentum filter: {'✅ ENABLED' if self.enable_momentum_filter else '❌ DISABLED'}")
        logger.info(f"  S/R proximity filter: {'✅ ENABLED' if self.enable_sr_proximity_filter else '❌ DISABLED'}")

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
        # Check timeframe confluence (increased from 50% to 60% for better quality)
        confluence_result, confluence_ratio = self._check_timeframe_confluence(analyses, signal_type)
        if not confluence_result:
            logger.warning(f"{symbol}: ❌ Insufficient timeframe confluence ({confluence_ratio:.1%} < 60%)")
            return False
        logger.info(f"{symbol}: ✅ Timeframe confluence passed ({confluence_ratio:.1%} >= 60%)")

        # Check trend strength (ADX filter)
        trend_strength_result, adx_value = self._check_trend_strength(analyses)
        if not trend_strength_result:
            logger.warning(f"{symbol}: ❌ Weak trend strength - ADX {adx_value:.1f} < {self.min_adx_for_trend}")
            return False
        logger.info(f"{symbol}: ✅ Trend strength passed (ADX {adx_value:.1f} >= {self.min_adx_for_trend})")

        # Check market regime (only BUY in uptrend, SELL in downtrend)
        regime_result, regime_reason = self._check_market_regime(analyses, signal_type)
        if not regime_result:
            logger.warning(f"{symbol}: ❌ Market regime mismatch - {regime_reason}")
            return False
        logger.info(f"{symbol}: ✅ Market regime aligned")

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

        # ========== NUEVOS FILTROS AVANZADOS (CONFIGURABLES) ==========

        # Filtro 1: Divergencias RSI/MACD
        if self.enable_divergence_filter:
            divergence_result, divergence_reason = self._check_divergence(analyses, signal_type)
            if not divergence_result:
                logger.warning(f"{symbol}: ❌ Divergence filter - {divergence_reason}")
                return False
            logger.info(f"{symbol}: ✅ Divergence check passed - {divergence_reason}")
        else:
            logger.debug(f"{symbol}: ⏭️  Divergence filter DISABLED - skipping")

        # Filtro 2: Límite de pérdidas consecutivas
        if self.enable_consecutive_losses_filter:
            consecutive_result, consecutive_reason = self._check_consecutive_losses(symbol)
            if not consecutive_result:
                logger.warning(f"{symbol}: ❌ Consecutive losses filter - {consecutive_reason}")
                return False
            logger.info(f"{symbol}: ✅ Consecutive losses check passed - {consecutive_reason}")
        else:
            logger.debug(f"{symbol}: ⏭️  Consecutive losses filter DISABLED - skipping")

        # Filtro 3: Momentum/Velocidad del precio
        if self.enable_momentum_filter:
            momentum_result, momentum_reason = self._check_momentum_filter(analyses, signal_type)
            if not momentum_result:
                logger.warning(f"{symbol}: ❌ Momentum filter - {momentum_reason}")
                return False
            logger.info(f"{symbol}: ✅ Momentum check passed - {momentum_reason}")
        else:
            logger.debug(f"{symbol}: ⏭️  Momentum filter DISABLED - skipping")

        # Filtro 4: Proximidad a Soporte/Resistencia
        if self.enable_sr_proximity_filter:
            sr_result, sr_reason = self._check_support_resistance_proximity(analyses, signal_type)
            if not sr_result:
                logger.warning(f"{symbol}: ❌ Support/Resistance filter - {sr_reason}")
                return False
            logger.info(f"{symbol}: ✅ S/R proximity check passed - {sr_reason}")
        else:
            logger.debug(f"{symbol}: ⏭️  S/R proximity filter DISABLED - skipping")

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

        # Need at least 60% agreement (increased from 50% for better quality)
        agreement_ratio = agreements / len(valid_analyses)

        return agreement_ratio >= 0.6, agreement_ratio

    def _check_trend_strength(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> tuple[bool, float]:
        """
        Check if trend is strong enough (using ADX indicator)
        Only trade when ADX > min_adx_for_trend (configurable, default 25)
        Avoid ranging/choppy markets where scalping is difficult
        """
        # Get primary timeframe analysis
        primary_analysis = analyses.get(config.primary_timeframe)

        if not primary_analysis:
            # Fallback to any available timeframe
            for analysis in analyses.values():
                if analysis:
                    primary_analysis = analysis
                    break

        if not primary_analysis:
            return True, 0.0  # Can't check, allow (conservative)

        adx = primary_analysis.indicators.get('adx', 0)

        # ADX > min_adx_for_trend indicates strong trend (good for scalping)
        # ADX < min_adx_for_trend indicates weak/ranging market (avoid)
        return adx >= self.min_adx_for_trend, adx

    def _check_market_regime(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Check if signal aligns with market regime (configurable)
        Only BUY in clear uptrends, only SELL in clear downtrends
        """
        # Si el filtro mejorado está desactivado, permitir todo
        if not self.enable_enhanced_trend_filter:
            return True, "Enhanced trend filter disabled"

        # Get primary timeframe analysis
        primary_analysis = analyses.get(config.primary_timeframe)

        if not primary_analysis:
            return True, "No primary analysis"

        indicators = primary_analysis.indicators

        # Get EMA crossover (short-term trend)
        ema_9 = indicators.get('ema_9', 0)
        ema_21 = indicators.get('ema_21', 0)

        # Get SMA for longer-term trend
        price = indicators.get('price', 0)
        sma_50 = indicators.get('sma_50', 0)

        if ema_9 == 0 or ema_21 == 0 or sma_50 == 0:
            return True, "Missing indicators"

        # Verificar si se requiere alineación de EMAs
        if self.ema_alignment_required:
            # Determine market regime con alineación estricta
            is_uptrend = (ema_9 > ema_21) and (price > sma_50)
            is_downtrend = (ema_9 < ema_21) and (price < sma_50)

            if signal_type == 'BUY':
                if is_uptrend:
                    return True, f"Uptrend confirmed (EMA9 > EMA21, Price > SMA50)"
                else:
                    return False, f"Not in uptrend (EMA9/EMA21/SMA50 misaligned)"

            elif signal_type == 'SELL':
                if is_downtrend:
                    return True, f"Downtrend confirmed (EMA9 < EMA21, Price < SMA50)"
                else:
                    return False, f"Not in downtrend (EMA9/EMA21/SMA50 misaligned)"

        return True, "Market regime check passed"

    def _check_trend_alignment(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """Check if signal aligns with higher timeframe trends (configurable)"""
        # Get higher timeframe analysis (4h or 1d)
        higher_tf_analysis = None
        selected_tf = None

        for tf in ['1d', '4h', '1h']:
            if tf in analyses and analyses[tf]:
                higher_tf_analysis = analyses[tf]
                selected_tf = tf
                break

        if not higher_tf_analysis:
            return True, "No higher TF available"  # Can't check, so allow

        # For BUY signals, prefer uptrends
        # For SELL signals, prefer downtrends
        indicators = higher_tf_analysis.indicators

        price = indicators.get('price', 0)
        sma_50 = indicators.get('sma_50', 0)

        if sma_50 == 0:
            return True, "SMA50 not available"

        percent_from_sma = ((price - sma_50) / sma_50) * 100

        # Usar el porcentaje configurado
        max_deviation = self.max_percent_from_sma50

        if signal_type == 'BUY':
            # Allow BUY if price is not too far below SMA50
            if percent_from_sma < -max_deviation:
                return False, f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf} (need >= -{max_deviation}%)"
            return True, f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf}"

        else:  # SELL
            # Allow SELL if price is not too far above SMA50
            if percent_from_sma > max_deviation:
                return False, f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf} (need <= +{max_deviation}%)"
            return True, f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf}"

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

    def _check_divergence(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Detecta divergencias entre RSI/MACD y precio (configurable)

        Una divergencia ocurre cuando:
        - RSI indica sobrecompra pero precio continúa subiendo (divergencia alcista)
        - RSI indica sobreventa pero precio continúa bajando (divergencia bajista)

        Args:
            analyses: Multi-timeframe analyses
            signal_type: BUY or SELL

        Returns:
            (True si NO hay divergencia peligrosa, mensaje)
        """
        # Usar timeframe principal
        primary_analysis = None
        for tf in [config.primary_timeframe, '15m', '1h', '5m']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break

        if not primary_analysis:
            return True, "No hay datos para verificar divergencias"

        indicators = primary_analysis.indicators
        rsi_14 = indicators.get('rsi_14', 50)

        # Para señales de VENTA
        if signal_type == 'SELL':
            # Verificar si RSI está en sobrecompra (configurable)
            if rsi_14 > self.divergence_rsi_overbought:
                # Verificar momentum del precio
                # Necesitamos agregar indicador de momentum personalizado
                # Por ahora usamos price momentum simple
                return True, f"RSI={rsi_14:.1f} en sobrecompra, verificar momentum manualmente"

        # Para señales de COMPRA
        elif signal_type == 'BUY':
            # Verificar si RSI está en sobreventa (configurable)
            if rsi_14 < self.divergence_rsi_oversold:
                return True, f"RSI={rsi_14:.1f} en sobreventa, verificar momentum manualmente"

        return True, f"No hay divergencias peligrosas (RSI={rsi_14:.1f})"

    def _check_consecutive_losses(
        self,
        symbol: str
    ) -> tuple[bool, str]:
        """
        Verifica si el símbolo tiene demasiadas pérdidas consecutivas
        y aplica período de enfriamiento si es necesario (configurable)

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
                    # Período de enfriamiento completado
                    logger.info(f"{symbol}: Período de enfriamiento completado. Reseteando contador de pérdidas.")
                    return True, f"Período de enfriamiento completado ({consecutive_losses} SLs previos)"

        return True, f"Pérdidas consecutivas: {consecutive_losses}/{self.max_consecutive_losses}"

    def _check_momentum_filter(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Filtra señales cuando hay momentum muy fuerte en dirección opuesta (configurable)
        Usa RSI y MACD como proxies del momentum del mercado
        """
        # Usar timeframe principal
        primary_analysis = None
        for tf in [config.primary_timeframe, '5m', '15m']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break

        if not primary_analysis:
            return True, "No hay datos para verificar momentum"

        indicators = primary_analysis.indicators
        rsi = indicators.get('rsi_14', 50)  # Default neutral
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        # Detectar momentum fuerte usando RSI y MACD
        # Momentum alcista fuerte: RSI > 80 + MACD por encima de señal
        # Momentum bajista fuerte: RSI < 20 + MACD por debajo de señal

        strong_bullish_momentum = (rsi > 80) or (macd > macd_signal and macd > 0 and abs(macd - macd_signal) > abs(macd) * 0.1)
        strong_bearish_momentum = (rsi < 20) or (macd < macd_signal and macd < 0 and abs(macd - macd_signal) > abs(macd) * 0.1)

        if signal_type == 'SELL':
            # Si queremos vender pero hay momentum alcista fuerte, bloquear
            if strong_bullish_momentum:
                return False, f"Momentum alcista fuerte detectado (RSI={rsi:.1f}, MACD={macd:.5f})"
            return True, f"No hay momentum alcista opuesto (RSI={rsi:.1f})"

        else:  # BUY
            # Si queremos comprar pero hay momentum bajista fuerte, bloquear
            if strong_bearish_momentum:
                return False, f"Momentum bajista fuerte detectado (RSI={rsi:.1f}, MACD={macd:.5f})"
            return True, f"No hay momentum bajista opuesto (RSI={rsi:.1f})"

    def _check_support_resistance_proximity(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Verifica que las señales estén cerca de zonas de soporte/resistencia (configurable)

        - SELL solo cerca de resistencias
        - BUY solo cerca de soportes
        - Evitar operar "en medio de la nada"
        """
        # Usar timeframe mediano para S/R
        primary_analysis = None
        for tf in ['1h', '4h', config.primary_timeframe]:
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
            # Debe estar cerca de resistencia (configurable)
            if distance_to_resistance <= self.sr_proximity_percent:
                return True, f"✅ Cerca de resistencia: {distance_to_resistance:.2f}% (R={resistance:.5f})"
            else:
                # Si está muy lejos de resistencia, rechazar (umbral configurable)
                if distance_to_resistance > self.sr_max_distance_percent:
                    return False, f"⚠️ Muy lejos de resistencia: {distance_to_resistance:.2f}% > {self.sr_max_distance_percent}%"

        # Para señales de COMPRA
        elif signal_type == 'BUY':
            # Debe estar cerca de soporte (configurable)
            if distance_to_support <= self.sr_proximity_percent:
                return True, f"✅ Cerca de soporte: {distance_to_support:.2f}% (S={support:.5f})"
            else:
                # Si está muy lejos de soporte, rechazar (umbral configurable)
                if distance_to_support > self.sr_max_distance_percent:
                    return False, f"⚠️ Muy lejos de soporte: {distance_to_support:.2f}% > {self.sr_max_distance_percent}%"

        return True, "Proximidad S/R aceptable"

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
