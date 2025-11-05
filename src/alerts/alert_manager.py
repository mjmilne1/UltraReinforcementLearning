import smtplib
import json
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
import requests
from enum import Enum

class AlertLevel(Enum):
    '''Alert severity levels'''
    INFO = 'info'       # General information
    SUCCESS = 'success' # Successful trades
    WARNING = 'warning' # Risk warnings
    CRITICAL = 'critical' # Critical errors
    EMERGENCY = 'emergency' # System failures

class AlertChannel:
    '''Base class for alert channels'''
    def send(self, message: str, level: AlertLevel) -> bool:
        raise NotImplementedError

class EmailAlert(AlertChannel):
    '''Email alert channel'''
    def __init__(self, smtp_server: str = 'smtp.gmail.com', 
                 smtp_port: int = 587,
                 sender_email: str = '',
                 sender_password: str = '',
                 recipient_emails: List[str] = []):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails
        
    def send(self, message: str, level: AlertLevel) -> bool:
        '''Send email alert'''
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = f'[{level.value.upper()}] Trading Alert - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            
            # Add emoji based on level
            emoji = {
                AlertLevel.INFO: '📊',
                AlertLevel.SUCCESS: '✅',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.CRITICAL: '🚨',
                AlertLevel.EMERGENCY: '🆘'
            }
            
            body = f'{emoji.get(level, "")} {message}'
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f'Email alert failed: {e}')
            return False

class DiscordAlert(AlertChannel):
    '''Discord webhook alerts'''
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send(self, message: str, level: AlertLevel) -> bool:
        '''Send Discord alert'''
        try:
            # Color codes for embed
            colors = {
                AlertLevel.INFO: 3447003,      # Blue
                AlertLevel.SUCCESS: 3066993,    # Green
                AlertLevel.WARNING: 15844367,   # Yellow
                AlertLevel.CRITICAL: 15158332,  # Red
                AlertLevel.EMERGENCY: 10038562  # Dark Red
            }
            
            embed = {
                'embeds': [{
                    'title': f'{level.value.upper()} Alert',
                    'description': message,
                    'color': colors.get(level, 0),
                    'timestamp': datetime.utcnow().isoformat(),
                    'footer': {'text': 'Ultra RL Trading System'}
                }]
            }
            
            response = requests.post(self.webhook_url, json=embed)
            return response.status_code == 204
        except Exception as e:
            print(f'Discord alert failed: {e}')
            return False

class TelegramAlert(AlertChannel):
    '''Telegram bot alerts'''
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f'https://api.telegram.org/bot{bot_token}'
        
    def send(self, message: str, level: AlertLevel) -> bool:
        '''Send Telegram alert'''
        try:
            # Add emoji prefix
            emoji = {
                AlertLevel.INFO: '📊',
                AlertLevel.SUCCESS: '✅',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.CRITICAL: '🚨',
                AlertLevel.EMERGENCY: '🆘'
            }
            
            text = f'{emoji.get(level, "")} *{level.value.upper()}*\n\n{message}'
            
            params = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(f'{self.api_url}/sendMessage', data=params)
            return response.json().get('ok', False)
        except Exception as e:
            print(f'Telegram alert failed: {e}')
            return False

class SMSAlert(AlertChannel):
    '''SMS alerts via Twilio'''
    def __init__(self, account_sid: str, auth_token: str, 
                 from_number: str, to_numbers: List[str]):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
        
    def send(self, message: str, level: AlertLevel) -> bool:
        '''Send SMS alert'''
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            for to_number in self.to_numbers:
                client.messages.create(
                    body=f'[{level.value.upper()}] {message[:160]}',
                    from_=self.from_number,
                    to=to_number
                )
            
            return True
        except Exception as e:
            print(f'SMS alert failed: {e}')
            return False

class AlertManager:
    '''Central alert management system'''
    def __init__(self, config_file: str = 'alert_config.json'):
        self.channels = []
        self.config_file = config_file
        self.alert_history = []
        self.alert_rules = {}
        
        # Load configuration
        self.load_config()
        
    def load_config(self):
        '''Load alert configuration'''
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
                # Setup channels based on config
                if config.get('email', {}).get('enabled'):
                    self.add_email_channel(config['email'])
                
                if config.get('discord', {}).get('enabled'):
                    self.add_discord_channel(config['discord'])
                
                if config.get('telegram', {}).get('enabled'):
                    self.add_telegram_channel(config['telegram'])
                    
                # Load alert rules
                self.alert_rules = config.get('rules', {})
    
    def add_channel(self, channel: AlertChannel):
        '''Add alert channel'''
        self.channels.append(channel)
        
    def add_email_channel(self, config: Dict):
        '''Add email channel from config'''
        channel = EmailAlert(
            sender_email=config.get('sender_email'),
            sender_password=config.get('sender_password'),
            recipient_emails=config.get('recipient_emails', [])
        )
        self.add_channel(channel)
    
    def add_discord_channel(self, config: Dict):
        '''Add Discord channel from config'''
        channel = DiscordAlert(webhook_url=config.get('webhook_url'))
        self.add_channel(channel)
    
    def add_telegram_channel(self, config: Dict):
        '''Add Telegram channel from config'''
        channel = TelegramAlert(
            bot_token=config.get('bot_token'),
            chat_id=config.get('chat_id')
        )
        self.add_channel(channel)
    
    def send_alert(self, message: str, level: AlertLevel = AlertLevel.INFO):
        '''Send alert to all channels'''
        timestamp = datetime.now()
        
        # Check if alert should be throttled
        if self._should_throttle(message, level):
            return
        
        # Send to all channels
        success_count = 0
        for channel in self.channels:
            if channel.send(message, level):
                success_count += 1
        
        # Log alert
        self.alert_history.append({
            'timestamp': timestamp,
            'message': message,
            'level': level.value,
            'channels_notified': success_count
        })
        
        # Save history
        self._save_history()
        
        return success_count > 0
    
    def _should_throttle(self, message: str, level: AlertLevel) -> bool:
        '''Check if alert should be throttled'''
        # Don't throttle emergency alerts
        if level == AlertLevel.EMERGENCY:
            return False
        
        # Check recent alerts
        if self.alert_history:
            recent = [a for a in self.alert_history[-10:] 
                     if a['message'] == message]
            if len(recent) >= 3:
                return True
        
        return False
    
    def _save_history(self):
        '''Save alert history to file'''
        history_file = 'alert_history.json'
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Convert to JSON-serializable format
        history_data = []
        for alert in self.alert_history:
            history_data.append({
                'timestamp': alert['timestamp'].isoformat(),
                'message': alert['message'],
                'level': alert['level'],
                'channels_notified': alert['channels_notified']
            })
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
