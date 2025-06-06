import logging
from typing import Any, Optional, Dict, Union

import aiohttp
import asyncio
from telegram import Bot
from telegram.constants import ParseMode
import os

from core.helper.compress_video import CompressVideo

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, token: str, chat_id: str, parse_mode: ParseMode = ParseMode.MARKDOWN_V2):
        self.chat_id = chat_id
        self.token = token
        self.bot = Bot(token=self.token)
        self.parse_mode = parse_mode

    async def _post(self, url: str, data=None, files=None) -> Optional[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                if files:
                    data_form = aiohttp.FormData()
                    for key, value in data.items():
                        data_form.add_field(key, str(value))
                    for name, file_obj in files.items():
                        data_form.add_field(name, file_obj, filename=name)

                    async with session.post(url, data=data_form, timeout=30) as response:
                        return await response.json()
                else:
                    async with session.post(url, data=data, timeout=15) as response:
                        return await response.json()
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return None

    async def send_message(self, text: str, **kwargs) -> Union[bool, int]:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': self.parse_mode.value,
            **kwargs
        }
        result = await self._post(url, data=payload)
        if result and result.get("ok"):
            logger.info(f"✅ Message sent: {result['result']['message_id']}")
            return result['result']['message_id']
        logger.error(f"❌ Failed to send message: {result}")
        return False

    async def send_photo(self, photo_path: str, caption: str = None, **kwargs) -> Union[bool, int]:
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        payload = {
            'chat_id': self.chat_id,
            'caption': caption,
            'parse_mode': self.parse_mode.value,
            **kwargs
        }

        try:
            with open(photo_path, 'rb') as f:
                files = {'photo': f}
                result = await self._post(url, data=payload, files=files)
                if result and result.get("ok"):
                    return result['result']['message_id']
        except Exception as e:
            logger.error(f"❌ Error sending photo: {e}")
        return False

    async def send_video(self, video_path: str, caption: str = None, **kwargs) -> Union[bool, int]:
        url = f"https://api.telegram.org/bot{self.token}/sendVideo"
        payload = {
            'chat_id': self.chat_id,
            'caption': caption,
            'parse_mode': self.parse_mode.value,
            **kwargs
        }

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            with open(video_path, 'rb') as f:
                files = {'video': f}
                result = await self._post(url, data=payload, files=files)
                if result and result.get("ok"):
                    return result['result']['message_id']
                else:
                    print(f"Sending video failed {result}")
                    return False
        except Exception as e:
            logger.error(f"❌ Error sending video: {e}")
        return False

    async def edit_message_text(self, message_id: int, text: str, **kwargs) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/editMessageText"
        payload = {
            'chat_id': self.chat_id,
            'message_id': message_id,
            'text': text,
            'parse_mode': self.parse_mode.value,
            **kwargs
        }
        result = await self._post(url, data=payload)
        return bool(result and result.get("ok"))

    async def edit_caption(self, message_id: int, caption: str, **kwargs) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/editMessageCaption"
        payload = {
            'chat_id': self.chat_id,
            'message_id': message_id,
            'caption': caption,
            'parse_mode': self.parse_mode.value,
            **kwargs
        }
        result = await self._post(url, data=payload)
        return bool(result and result.get("ok"))

    async def get_me(self) -> Optional[Dict[str, Any]]:
        try:
            user = await self.bot.get_me()
            return user.to_dict()
        except Exception as e:
            logger.error(f"get_me failed: {e}")
            return None

    def send_media_sync(self, caption: str, media: str, media_type: str = "photo"):
        return asyncio.run(self.send_media(caption=caption, media=media, media_type=media_type))

    async def send_media(self, caption: str, media: str, media_type: str = "photo"):
        if media_type == "photo":
            return await self.send_photo(photo_path=media, caption=caption)
        elif media_type == "video":
            compressed_media = await CompressVideo.compress_video(input_path=media)
            return await self.send_video(video_path=compressed_media, caption=caption)
        else:
            raise ValueError(f"Unsupported media_type: {media_type}")



def escape_markdown_v2(text: str) -> str:
    """
    Escapes text for Telegram MarkdownV2 parse mode.
    Special characters: '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
    must be escaped with the preceding character '\'.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join(['\\' + c if c in escape_chars else c for c in text])


def escape_html(text: str) -> str:
    """
    Escapes text for Telegram HTML parse mode.
    Replaces '<', '>', '&' with their HTML entities.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
