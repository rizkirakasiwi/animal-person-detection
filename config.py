import logging
import os.path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")  # <<< SET IN .ENV
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_DEFAULT_CHAT_ID_HERE")  # <<< SET IN .ENV
