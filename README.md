## 🔥 Object Detection & Alert System

This project is a real-time video analysis tool built with OpenCV and YOLOv8. It supports detection of fire, smoke, people, and more. When an object is detected, the system can:

* Record video evidence
* Take screenshots
* Send Telegram notifications

---

## 📦 Features

* ✅ Modular detector architecture (`FireDetector`, `GeneralDetector`, etc.)
* ✅ Support for polygon segmentation & bounding boxes
* ✅ Image & video evidence capture
* ✅ Telegram bot integration for alerting
* ✅ CLI-based video input

---

## ⚙️ Requirements

* Python 3.9–3.11 (recommended)
* `pip` or `poetry`
* [YOLOv11 models](https://github.com/ultralytics/ultralytics)

---

## 🚀 Installation

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/object-detection-alert.git
cd object-detection-alert
```

2. **Set up virtual environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

maybe you need to install cuda to run in GPU locally, please refer to this [link](https://pytorch.org/get-started/locally/) and download your cuda version for example
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

4. **Download models**

Place your YOLOv8 models in the `model/` directory.

Example:

```
model/
  ├── fire_v5.pt
  └── yolov8n.pt
```

---

## 🎬 Running the Project

Use the CLI script to run detection on a video:

```bash
python run_detection.py --video assets/videos/wildfire.mp4
```

Press `q` to quit during playback.

---

## 📁 Project Structure

```
core/
│
├── engine/               # Detection engine (frame processing logic)
│   └── detection_engine.py
│
├── helper/              # Utilities
│   ├── capture.py       # Screenshot logic
│   ├── recorder.py      # Video recording logic
│   ├── detector.py      # YOLO detector wrapper
│   └── extensions.py    # Utilities (e.g., to_camel_case)
│
├── report/              # Telegram reporting logic
│   └── report.py
│
usecase/
│   ├── fire_detector.py
│   ├── general_detector.py
│   └── road_dmg_detector.py (optional)
│
assets/
│   └── videos/          # Sample video inputs
│
model/                   # YOLOv8 models (.pt files)
│
run_detection.py         # CLI entry point
requirements.txt
```

---

## 🧠 Adding Your Own Detector

1. Create a new file in `usecase/` like `my_detector.py`.
2. Inherit from `BaseDetector`.
3. Override the `detect()` method using `ObjectDetector`.

Example:

```python
class MyDetector(BaseDetector):
    def __init__(self):
        self.detector = ObjectDetector("model/my_model.pt", allowed_classes=[0, 1])

    def detect(self, frame: np.ndarray) -> List[Dict]:
        return self.detector.detect(frame)
```

---

## 📲 Telegram Alert Setup (Optional)

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Add your bot token and chat ID inside `core/report/report.py`
3. Supports sending:

   * Detection messages
   * Screenshots
   * Video evidence

---

## 🛠 Future Ideas

* Add email or WhatsApp support
* Deploy on Jetson Nano or Raspberry Pi
* Export as Docker image for cloud/edge use

---

## 📄 License

MIT License. Use freely with attribution.
