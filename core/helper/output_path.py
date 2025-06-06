import os
from datetime import datetime, timezone

def get_output_path(path, extension):
    os.makedirs(f"output/{path}", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return os.path.join("output", path, f"{timestamp}.{extension}")