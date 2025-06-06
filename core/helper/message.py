import html
from datetime import datetime
from typing import Optional


class Message:

    emoji_map = {
        "fire": "🔥",
        "smoke": "💨",
        "person": "🧍",
    }

    @staticmethod
    def generate_message(
        detections: list[dict],
        video_url: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        if not detections:
            return "⚠️ <b>Video detections report.</b>"

        top_detection = max(detections, key=lambda d: d["confidence"])
        top_class = top_detection["class_name"]
        confidence_percent = Message._format_confidence(top_detection["confidence"])
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

        parts = [
            Message._format_header(top_class, confidence_percent),
            "",
            Message._format_top_detection(top_class, location, current_time, confidence_percent),
            Message._format_summary(detections),
            Message._format_video_link(video_url),
        ]

        return "\n".join(parts)

    @staticmethod
    def _format_confidence(conf: float) -> float:
        return round(conf * 100 if conf <= 1 else conf, 1)

    @staticmethod
    def _format_header(top_class: str, confidence: float) -> str:
        return f"🚨🔴🚨 <b>{top_class.upper()} DETECTED WITH CONFIDENCE {confidence}%</b> 🚨🔴🚨"

    @staticmethod
    def _format_top_detection(top_class: str, location: Optional[str], time_str: str, confidence: float) -> str:
        emoji = Message.emoji_map.get(top_class.lower(), "⚠️")
        lines = [
            f"{emoji} <b>Top Detection:</b> {top_class}",
            f"📍 <b>Location:</b> {location}" if location else "",
            f"⏰ <b>Time:</b> {time_str}",
            f"🎯 <b>Confidence:</b> {confidence}%",
        ]
        return "\n".join(filter(None, lines))

    @staticmethod
    def _format_summary(detections: list[dict]) -> str:
        class_conf_map = {}
        for det in detections:
            cls = det["class_name"]
            class_conf_map.setdefault(cls, []).append(det["confidence"])

        summary_lines = []
        for cls, confs in class_conf_map.items():
            avg = sum(confs) / len(confs)
            summary_lines.append(f"• {cls}: avg {avg * 100:.1f}%, count {len(confs)}")

        return "<b>📌 Detection Summary:</b>\n\n" + "\n".join(summary_lines)

    @staticmethod
    def _format_video_link(video_url: Optional[str]) -> str:
        if video_url:
            safe_url = html.escape(video_url)
            return f"\n📹 <a href='{safe_url}'><b>► VIEW DETECTION EVIDENCE</b></a>"
        return "\n📹 <i>Video evidence processing...</i>"
