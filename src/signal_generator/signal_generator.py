"""
Signal Generator
Generates high-probability trading signals from AI analysis
Specialized for synthetic indices (PainX/GainX) scalping strategy
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis, MarketAnalyzer
from .signal_filter import SignalFilter
from .risk_manager import RiskManager
from .signal_tracker import SignalTracker
from config import config


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    confidence: float
    timeframe: str
    timestamp: datetime
    analysis: Dict
    risk_reward_ratio: float
    reason: str
    lot_size: float
    atr_at_signal: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def get_formatted_message(self) -> str:
        """Get formatted signal message for Telegram"""
        signal_emoji = "🟢" if self.signal_type == "BUY" else "🔴"

        message = f"""
{signal_emoji} **{self.signal_type} SIGNAL** {signal_emoji}

**Symbol:** {self.symbol}
**Timeframe:** {self.timeframe}
**Confidence:** {self.confidence:.1%}
**Strength:** {self.strength}/100

📊 **Entry Price:** ${self.entry_price:.4f}
🛑 **Stop Loss:** ${self.stop_loss:.4f}
🎯 **Take Profit:**
"""
        for i, tp in enumerate(self.take_profit_levels, 1):
            message += f"   TP{i}: ${tp:.4f}\n"

        message += f"""
📈 **Risk/Reward:** 1:{self.risk_reward_ratio:.2f}

💡 **Reason:** {self.reason}

⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return message


class SignalGenerator:
    """Generates trading signals from market analysis"""

    def __init__(
        self,
        analyzer: MarketAnalyzer,
        signal_filter: SignalFilter,
        risk_manager: RiskManager,
        min_confidence: float = 0.75,
        mt5_connector=None  # Optional MT5 connector for real position verification
    ):
        """
        Initialize Signal Generator

        Args:
            analyzer: Market analyzer instance
            signal_filter: Signal filter instance
            risk_manager: Risk manager instance
            min_confidence: Minimum confidence threshold
            mt5_connector: Optional MT5Connector for verifying real positions
        """
        self.analyzer = analyzer
        self.signal_filter = signal_filter
        self.risk_manager = risk_manager
        self.mt5_connector = mt5_connector  # Store MT5 connector
        self.signal_tracker = SignalTracker(
            min_time_between_signals=30,  # 30 minutes for scalping
            max_price_distance_percent=0.5,  # 0.5% price movement invalidates signal
            max_signal_age=120  # 2 hours max age
        )

        self.min_confidence = min_confidence

        self.generated_signals: List[TradingSignal] = []
        self.signal_history: Dict[str, List[TradingSignal]] = {}

        logger.info("Signal Generator initialized")

    def _is_painx_symbol(self, symbol: str) -> bool:
        """Check if symbol is a PainX index"""
        return 'PainX' in symbol or 'painx' in symbol.lower()

    def _is_gainx_symbol(self, symbol: str) -> bool:
        """Check if symbol is a GainX index"""
        return 'GainX' in symbol or 'gainx' in symbol.lower()

    def _validate_signal_direction(self, symbol: str, signal_type: str) -> bool:
        """
        Validate signal direction for synthetic indices

        PainX: Only SELL signals (spikes down)
        GainX: Only BUY signals (spikes up)

        Args:
            symbol: Trading symbol
            signal_type: BUY or SELL

        Returns:
            True if signal direction is valid for this index
        """
        if self._is_painx_symbol(symbol) and config.enforce_painx_sell_only:
            if signal_type != 'SELL':
                logger.warning(
                    f"{symbol}: ❌ PainX index - Only SELL signals allowed "
                    f"(spikes are downward). Blocking {signal_type} signal."
                )
                return False
            logger.info(f"{symbol}: ✅ PainX index - SELL signal is valid (spike direction)")

        elif self._is_gainx_symbol(symbol) and config.enforce_gainx_buy_only:
            if signal_type != 'BUY':
                logger.warning(
                    f"{symbol}: ❌ GainX index - Only BUY signals allowed "
                    f"(spikes are upward). Blocking {signal_type} signal."
                )
                return False
            logger.info(f"{symbol}: ✅ GainX index - BUY signal is valid (spike direction)")

        return True

    def generate_signal(
        self,
        symbol: str,
        multi_tf_analyses: Dict[str, Optional[MarketAnalysis]],
        current_price: float,
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from multi-timeframe analysis

        Args:
            symbol: Trading pair
            multi_tf_analyses: Multi-timeframe analysis results
            current_price: Current market price

        Returns:
            TradingSignal or None
        """
        try:
            # Get primary timeframe analysis from configuration
            primary_analysis = None

            # First try to get the configured primary timeframe
            if config.primary_timeframe in multi_tf_analyses and multi_tf_analyses[config.primary_timeframe]:
                primary_analysis = multi_tf_analyses[config.primary_timeframe]
                logger.info(f"{symbol}: Using configured primary timeframe: {config.primary_timeframe}")
            else:
                # Fallback: try other timeframes in order of configuration
                logger.warning(f"{symbol}: Primary timeframe {config.primary_timeframe} not available, trying fallback")
                for tf in config.timeframes:
                    if tf in multi_tf_analyses and multi_tf_analyses[tf]:
                        primary_analysis = multi_tf_analyses[tf]
                        logger.info(f"{symbol}: Using fallback timeframe: {tf}")
                        break

            if not primary_analysis:
                logger.warning(f"{symbol}: No primary analysis available")
                return None

            # Usar directamente la señal y confianza del análisis primario
            signal_type = primary_analysis.signal
            confidence = primary_analysis.confidence

            logger.info(f"Analysis result for {symbol}: Signal={signal_type}, Confidence={confidence:.2%}")

            # Check if signal is actionable
            if signal_type == 'HOLD':
                logger.info(f"{symbol}: ⏸️  Primary signal is HOLD, skipping signal generation")
                return None

            # Validate signal direction for synthetic indices (PainX/GainX)
            if not self._validate_signal_direction(symbol, signal_type):
                return None

            # Check for duplicate signals (with MT5 verification if available)
            if not self.signal_tracker.should_send_signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=current_price,
                current_price=current_price,
                mt5_connector=self.mt5_connector  # Pass MT5 connector for real verification
            ):
                return None

            # Check minimum thresholds
            if confidence < self.min_confidence:
                logger.warning(
                    f"{symbol}: ❌ Confidence {confidence:.2%} below threshold {self.min_confidence:.2%}"
                )
                return None

            logger.info(f"{symbol}: ✅ Passed confidence threshold (Confidence: {confidence:.2%} >= {self.min_confidence:.2%})")

            # Apply signal quality filters (for Telegram notification)
            # Note: Execution limits are checked separately in main_mt5.py before executing on MT5
            if not self.signal_filter.should_notify(symbol, signal_type, multi_tf_analyses):
                logger.warning(f"{symbol}: ❌ Signal filtered out by quality filters")
                return None

            logger.info(f"{symbol}: ✅ Passed quality filters - Signal will be generated")

            # Calculate entry, stop loss, and take profit
            risk_params = self.risk_manager.calculate_risk_parameters(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=current_price,
                analysis=primary_analysis,
                confidence=confidence,
                full_market_data=market_data[primary_analysis.timeframe] # Pass the correct dataframe
            )

            if not risk_params:
                logger.warning(f"{symbol}: Could not calculate risk parameters")
                return None

            # Generate signal ID
            signal_id = f"{symbol}_{signal_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Create reason
            reason = self._generate_reason(primary_analysis, multi_tf_analyses)

            # Create trading signal
            signal = TradingSignal(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                entry_price=risk_params['entry_price'],
                stop_loss=risk_params['stop_loss'],
                take_profit_levels=risk_params['take_profit_levels'],
                confidence=confidence,
                timeframe=primary_analysis.timeframe,
                timestamp=datetime.utcnow(),
                analysis=primary_analysis.to_dict(),
                risk_reward_ratio=risk_params['risk_reward_ratio'],
                reason=reason,
                lot_size=risk_params['lot_size'],
                atr_at_signal=risk_params['atr_at_signal']
            )

            # Store signal
            self.generated_signals.append(signal)
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(signal)

            # Track signal to prevent duplicates
            self.signal_tracker.track_signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=risk_params['entry_price'],
                stop_loss=risk_params['stop_loss'],
                take_profit=risk_params['take_profit_levels'][0] if risk_params['take_profit_levels'] else 0,
                signal_id=signal_id
            )

            logger.info(
                f"Generated {signal_type} signal for {symbol} "
                f"(confidence: {confidence:.2%})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _generate_reason(
        self,
        primary_analysis: MarketAnalysis,
        multi_tf_analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> str:
        """Generate human-readable reason for signal"""
        reasons = []

        # Multi-timeframe alignment
        aligned_count = sum(
            1 for analysis in multi_tf_analyses.values()
            if analysis and analysis.signal == primary_analysis.signal
        )
        total_count = len([a for a in multi_tf_analyses.values() if a is not None])

        if aligned_count / total_count > 0.7:
            reasons.append(f"{aligned_count}/{total_count} timeframes aligned")

        # Technical indicators
        indicators = primary_analysis.indicators

        if primary_analysis.signal == 'BUY':
            if indicators.get('rsi_14', 50) < 40:
                reasons.append("RSI oversold")
            if indicators.get('price', 0) > indicators.get('ema_50', 0):
                reasons.append("Above EMA50")
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                reasons.append("MACD bullish")
        else:  # SELL
            if indicators.get('rsi_14', 50) > 60:
                reasons.append("RSI overbought")
            if indicators.get('price', 0) < indicators.get('ema_50', 0):
                reasons.append("Below EMA50")
            if indicators.get('macd', 0) < indicators.get('macd_signal', 0):
                reasons.append("MACD bearish")

        # Patterns
        patterns = primary_analysis.patterns
        bullish_patterns = ['bullish_engulfing', 'morning_star', 'hammer']
        bearish_patterns = ['bearish_engulfing', 'evening_star', 'shooting_star']

        for pattern in bullish_patterns:
            if patterns.get(pattern) and primary_analysis.signal == 'BUY':
                reasons.append(f"{pattern.replace('_', ' ').title()}")

        for pattern in bearish_patterns:
            if patterns.get(pattern) and primary_analysis.signal == 'SELL':
                reasons.append(f"{pattern.replace('_', ' ').title()}")

        return ", ".join(reasons) if reasons else "AI consensus signal"

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[TradingSignal]:
        """
        Get recent signals

        Args:
            symbol: Filter by symbol (None for all)
            limit: Maximum number of signals

        Returns:
            List of recent signals
        """
        if symbol:
            signals = self.signal_history.get(symbol, [])
        else:
            signals = self.generated_signals

        return signals[-limit:]

    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals"""
        if not self.generated_signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0,
                'avg_strength': 0
            }

        buy_signals = [s for s in self.generated_signals if s.signal_type == 'BUY']
        sell_signals = [s for s in self.generated_signals if s.signal_type == 'SELL']

        return {
            'total_signals': len(self.generated_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': sum(s.confidence for s in self.generated_signals) / len(self.generated_signals),
            'avg_strength': sum(s.strength for s in self.generated_signals) / len(self.generated_signals),
            'symbols': list(self.signal_history.keys())
        }

    def clear_old_signals(self, days: int = 7):
        """Clear signals older than specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        self.generated_signals = [
            s for s in self.generated_signals
            if s.timestamp > cutoff_time
        ]

        for symbol in self.signal_history:
            self.signal_history[symbol] = [
                s for s in self.signal_history[symbol]
                if s.timestamp > cutoff_time
            ]

        logger.info(f"Cleared signals older than {days} days")
