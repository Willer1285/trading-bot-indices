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
        # Check timeframe confluence (increased from 50% to 60% for better quality)
        confluence_result, confluence_ratio = self._check_timeframe_confluence(analyses, signal_type)
        if not confluence_result:
            logger.warning(f"{symbol}: ❌ Insufficient timeframe confluence ({confluence_ratio:.1%} < 60%)")
            return False
        logger.info(f"{symbol}: ✅ Timeframe confluence passed ({confluence_ratio:.1%} >= 60%)")

        # Check trend strength (ADX filter - NEW)
        trend_strength_result, adx_value = self._check_trend_strength(analyses)
        if not trend_strength_result:
            logger.warning(f"{symbol}: ❌ Weak trend strength - ADX {adx_value:.1f} < 25")
            return False
        logger.info(f"{symbol}: ✅ Trend strength passed (ADX {adx_value:.1f} >= 25)")

        # Check market regime (NEW - only BUY in uptrend, SELL in downtrend)
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

        # Check for conflicting signals
        if self._has_conflicting_recent_signal(symbol, signal_type):
            logger.warning(f"{symbol}: ❌ Conflicting recent signal")
            return False
        logger.info(f"{symbol}: ✅ No conflicting signals")

        # All quality checks passed (no execution limits checked here)
        logger.info(f"{symbol}: ✅✅✅ ALL QUALITY FILTERS PASSED - Signal will be sent to Telegram!")
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
        Only trade when ADX > 25 (strong trending market)
        Avoid ranging/choppy markets where scalping is difficult
        """
        from config import config

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

        # ADX > 25 indicates strong trend (good for scalping)
        # ADX < 25 indicates weak/ranging market (avoid)
        return adx >= 25, adx

    def _check_market_regime(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """
        Check if signal aligns with market regime
        Only BUY in clear uptrends, only SELL in clear downtrends
        More strict than trend_alignment check
        """
        from config import config

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

        # Determine market regime
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

        return False, "Unknown signal type"

    def _check_trend_alignment(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> tuple[bool, str]:
        """Check if signal aligns with higher timeframe trends"""
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

        if signal_type == 'BUY':
            # Allow BUY in uptrend or near support
            result = price >= sma_50 * 0.98  # Within 2% of SMA50
            reason = f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf} (need >= -2%)"
            return result, reason

        else:  # SELL
            # Allow SELL in downtrend or near resistance
            result = price <= sma_50 * 1.02  # Within 2% of SMA50
            reason = f"Price {percent_from_sma:+.2f}% from SMA50 on {selected_tf} (need <= +2%)"
            return result, reason

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

    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        return {
            'daily_signals': self.daily_signal_count,
            'max_daily': self.max_signals_per_day,
            'pair_counts': self.pair_signal_count,
            'recent_signals': len(self.recent_signals)
        }
