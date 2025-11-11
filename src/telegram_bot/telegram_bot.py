"""
Telegram Bot
Handles sending messages and charts to Telegram
"""

import asyncio
from typing import Optional, List, Dict
from telegram import Bot, Update
from telegram.error import TelegramError
from telegram.constants import ParseMode
from loguru import logger
import io

from signal_generator.signal_generator import TradingSignal
from .message_formatter import MessageFormatter
from .chart_generator import ChartGenerator


class TelegramBot:
    """Telegram bot for sending trading signals"""

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        enable_charts: bool = True
    ):
        """
        Initialize Telegram Bot

        Args:
            bot_token: Telegram bot token
            channel_id: Channel ID to send messages to
            enable_charts: Whether to generate and send charts
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.enable_charts = enable_charts

        self.bot: Optional[Bot] = None
        self.formatter = MessageFormatter()
        self.chart_generator = ChartGenerator() if enable_charts else None

        self.message_queue: List[Dict] = []
        self.is_running = False

        self._initialize_bot()

    def _initialize_bot(self):
        """Initialize Telegram bot"""
        try:
            if not self.bot_token or not self.channel_id:
                logger.warning("Telegram bot token or channel ID not configured")
                return

            self.bot = Bot(token=self.bot_token)
            logger.info("Telegram bot initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None

    async def send_signal(
        self,
        signal: TradingSignal,
        market_data: Optional[Dict] = None
    ) -> bool:
        """
        Send trading signal to Telegram channel

        Args:
            signal: Trading signal to send
            market_data: Optional market data for chart

        Returns:
            True if sent successfully
        """
        if not self.bot:
            logger.warning("Telegram bot not initialized")
            return False

        try:
            # Format message
            message = self.formatter.format_signal(signal)

            # Send text message
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

            # Send chart if enabled and data available
            if self.enable_charts and self.chart_generator and market_data and 'data' in market_data:
                chart_image = self.chart_generator.generate_signal_chart(
                    signal, market_data['data']
                )

                if chart_image:
                    await self.bot.send_photo(
                        chat_id=self.channel_id,
                        photo=chart_image,
                        caption=f"Chart for {signal.symbol}"
                    )

            logger.info(f"Sent {signal.signal_type} signal for {signal.symbol} to Telegram")
            return True

        except TelegramError as e:
            logger.error(f"Telegram error sending signal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {e}")
            return False

    async def send_message(self, message: str, parse_mode: str = ParseMode.MARKDOWN) -> bool:
        """
        Send a simple text message

        Args:
            message: Message text
            parse_mode: Parse mode (MARKDOWN or HTML)

        Returns:
            True if sent successfully
        """
        if not self.bot:
            return False

        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=parse_mode
            )
            return True

        except Exception as e:
            logger.error(f"Error sending message to Telegram: {e}")
            return False

    async def send_daily_summary(self, summary: Dict) -> bool:
        """
        Send daily performance summary

        Args:
            summary: Summary statistics

        Returns:
            True if sent successfully
        """
        if not self.bot:
            return False

        try:
            message = self.formatter.format_daily_summary(summary)

            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

            logger.info("Sent daily summary to Telegram")
            return True

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False

    async def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """
        Send error alert

        Args:
            error_message: Error message
            context: Additional context

        Returns:
            True if sent successfully
        """
        if not self.bot:
            return False

        try:
            message = f"""
⚠️ **ERROR ALERT** ⚠️

**Error:** {error_message}

**Context:** {context}

**Time:** {self.formatter.get_timestamp()}
"""

            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

            return True

        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return False

    async def send_performance_update(self, performance: Dict) -> bool:
        """
        Send performance metrics update

        Args:
            performance: Performance metrics

        Returns:
            True if sent successfully
        """
        if not self.bot:
            return False

        try:
            message = self.formatter.format_performance(performance)

            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

            logger.info("Sent performance update to Telegram")
            return True

        except Exception as e:
            logger.error(f"Error sending performance update: {e}")
            return False

    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection

        Returns:
            True if connection is working
        """
        if not self.bot:
            logger.error("Bot not initialized")
            return False

        try:
            me = await self.bot.get_me()
            logger.info(f"Connected to Telegram as @{me.username}")

            # Send test message
            await self.send_message("✅ Bot connection test successful!")

            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def queue_message(self, message_data: Dict):
        """
        Add message to queue for async sending

        Args:
            message_data: Message data dictionary
        """
        self.message_queue.append(message_data)

    async def process_queue(self):
        """Process queued messages"""
        while self.message_queue:
            message_data = self.message_queue.pop(0)

            message_type = message_data.get('type', 'text')

            if message_type == 'signal':
                await self.send_signal(
                    message_data['signal'],
                    message_data.get('market_data')
                )
            elif message_type == 'text':
                await self.send_message(message_data['message'])
            elif message_type == 'summary':
                await self.send_daily_summary(message_data['summary'])

            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

    async def start_queue_processor(self):
        """Start continuous queue processing"""
        self.is_running = True

        while self.is_running:
            try:
                if self.message_queue:
                    await self.process_queue()

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Stop queue processor"""
        self.is_running = False
        logger.info("Telegram bot stopped")

    async def send_break_even_notification(self, position: Dict, new_sl: float) -> bool:
        """Sends a break even notification."""
        message = self.formatter.format_break_even(position, new_sl)
        return await self.send_message(message)

    async def send_trailing_stop_notification(self, position: Dict, new_sl: float) -> bool:
        """Sends a trailing stop notification."""
        message = self.formatter.format_trailing_stop(position, new_sl)
        return await self.send_message(message)

    async def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a photo to Telegram channel

        Args:
            photo_path: Path to the image file
            caption: Optional caption for the photo

        Returns:
            True if sent successfully
        """
        if not self.bot:
            logger.warning("Telegram bot not initialized")
            return False

        try:
            with open(photo_path, 'rb') as photo_file:
                await self.bot.send_photo(
                    chat_id=self.channel_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN if caption else None
                )

            logger.info(f"Sent photo to Telegram: {photo_path}")
            return True

        except Exception as e:
            logger.error(f"Error sending photo to Telegram: {e}")
            return False
