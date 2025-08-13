import os
from typing import List, Dict
import asyncio

# Try to import telegram bot, but make it optional
try:
    from telegram import Bot, Update
    from telegram.ext import Updater, CommandHandler, CallbackContext
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Create dummy types for when telegram is not available
    class Update:
        pass
    class CallbackContext:
        pass
    print("Warning: python-telegram-bot not available. Telegram alerts will be disabled.")

class TelegramAlertBot:
    def __init__(self, token: str = None):
        if not TELEGRAM_AVAILABLE:
            print("Telegram bot disabled - python-telegram-bot not installed")
            self.enabled = False
            return
            
        self.enabled = True
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        
        if not self.token:
            print("Warning: No Telegram bot token provided. Telegram alerts will be disabled.")
            self.enabled = False
            return
            
        try:
            self.bot = Bot(token=self.token)
            self.updater = Updater(token=self.token, use_context=True)
            self.subscribers = set()  # In production, this would be stored in DB
            
            # Add command handlers
            self.updater.dispatcher.add_handler(CommandHandler("start", self.start_command))
            self.updater.dispatcher.add_handler(CommandHandler("subscribe", self.subscribe_command))
            self.updater.dispatcher.add_handler(CommandHandler("unsubscribe", self.unsubscribe_command))
        except Exception as e:
            print(f"Error initializing Telegram bot: {e}")
            self.enabled = False

    def start(self):
        """Start the bot"""
        if not self.enabled:
            return
        try:
            self.updater.start_polling()
        except Exception as e:
            print(f"Error starting Telegram bot: {e}")

    def stop(self):
        """Stop the bot"""
        if not self.enabled:
            return
        try:
            self.updater.stop()
        except Exception as e:
            print(f"Error stopping Telegram bot: {e}")

    def start_command(self, update: Update, context: CallbackContext):
        """Handle /start command"""
        if not self.enabled:
            return
            
        try:
            welcome_message = """
🚨 Welcome to NOESIS Alert Bot!

This bot provides real-time alerts about civil unrest and protests.

Commands:
/subscribe - Subscribe to alerts
/unsubscribe - Unsubscribe from alerts

You'll receive alerts for verified incidents with medium or high severity.
            """
            update.message.reply_text(welcome_message)
        except Exception as e:
            print(f"Error in start command: {e}")

    def subscribe_command(self, update: Update, context: CallbackContext):
        """Handle /subscribe command"""
        if not self.enabled:
            return
            
        try:
            chat_id = update.effective_chat.id
            self.subscribers.add(chat_id)
            update.message.reply_text("✅ You've been subscribed to NOESIS alerts!")
        except Exception as e:
            print(f"Error in subscribe command: {e}")

    def unsubscribe_command(self, update: Update, context: CallbackContext):
        """Handle /unsubscribe command"""
        if not self.enabled:
            return
            
        try:
            chat_id = update.effective_chat.id
            self.subscribers.discard(chat_id)
            update.message.reply_text("❌ You've been unsubscribed from NOESIS alerts.")
        except Exception as e:
            print(f"Error in unsubscribe command: {e}")

    def send_alert(self, chat_id: int, message: str):
        """Send alert to specific chat"""
        if not self.enabled:
            return
            
        try:
            self.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")

    def broadcast_incident(self, incident: Dict):
        """Broadcast incident to all subscribers"""
        if not self.enabled:
            return
            
        if incident.get("severity") in ["medium", "high"] and incident.get("status") in ["verified", "medium"]:
            message = self.format_incident_message(incident)
            
            for chat_id in self.subscribers:
                self.send_alert(chat_id, message)

    def format_incident_message(self, incident: Dict) -> str:
        """Format incident for Telegram message"""
        severity_emoji = {
            "low": "🟡",
            "medium": "🟠", 
            "high": "🔴"
        }
        
        emoji = severity_emoji.get(incident.get("severity", "low"), "🟡")
        
        message = f"""
{emoji} <b>NOESIS Alert</b>

<b>Location:</b> {incident.get('location', 'Unknown')}
<b>Severity:</b> {incident.get('severity', 'Unknown').title()}
<b>Status:</b> {incident.get('status', 'Unknown').title()}
<b>Confidence:</b> {incident.get('confidence_score', 0)}%

<b>Description:</b>
{incident.get('description', 'No description available')}

<b>Sources:</b> {incident.get('source_count', 0)} posts across {incident.get('platform_diversity', 0)} platforms
        """
        
        return message.strip() 