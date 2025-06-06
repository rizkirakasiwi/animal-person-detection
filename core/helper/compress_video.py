import os
import subprocess
import asyncio
import shutil

class CompressVideo:
    @staticmethod
    async def compress_video(input_path: str, crf: int = 28, resolution: str = "1280x720") -> str:
        """
        Asynchronously compress a video using ffmpeg and return the new path.
        If compression fails, returns the original path.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        if not shutil.which("ffmpeg"):
            print("[CompressVideo] ffmpeg not found in PATH. Skipping compression.")
            return input_path  # Fallback: return original

        print(f"[CompressVideo] Compressing {input_path}...")

        abs_input_path = os.path.abspath(input_path)
        output_path = abs_input_path.replace(".mp4", "_compressed.mp4")

        command = [
            "ffmpeg", "-y",
            "-i", abs_input_path,
            "-vcodec", "libx264",
            "-crf", str(crf),
            "-vf", f"scale={resolution}",
            output_path
        ]

        def run_compression():
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[CompressVideo] Compressed video saved to: {output_path}")

                # Only delete the original if compression succeeded
                os.remove(abs_input_path)
                print(f"[CompressVideo] Original video deleted: {abs_input_path}")
                return output_path
            except subprocess.CalledProcessError as e:
                print(f"[CompressVideo] Compression failed: {e}")
                return abs_input_path

        return await asyncio.to_thread(run_compression)
