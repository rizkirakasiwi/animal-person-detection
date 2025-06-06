import threading

from telegram.constants import ParseMode

import config
from core.helper.telegram_bot import TelegramBot


def _init_telegram_bot():
    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        raise ValueError("Telegram Bot Token is not configured.")
    return TelegramBot(
        token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
        parse_mode=ParseMode.HTML
    )


class Report:
    def __init__(self):
        self.bot = _init_telegram_bot()

    def send_notif(self, message: str, image: str):
        try:
            print(f"[Report] Sending notification with image: {image}")
            return self.bot.send_media_sync(caption=message, media=image)
        except Exception as e:
            print(f"[Report] Error sending image notification: {e}")
            return None

    def send_video(self, video: str, message: str):
        try:
            print(f"[Report] Sending notification with video: {video}")
            return self.bot.send_media_sync(caption=message, media=video, media_type="video")
        except Exception as e:
            print(f"[Report] Error sending video notification: {e}")
            return None

    def send_video_async(self, video: str, message: str):

        def wrapper():
            try:
                self.send_video(video, message)
            except Exception as e:
                print(f"[Report] send_video_async failed: {e}")

        threading.Thread(target=wrapper, daemon=True).start()
