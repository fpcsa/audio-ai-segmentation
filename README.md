# AI Audio Segmentation API

This project is a **FastAPI-based** service for **audio and video file analysis** using [Google's YAMNet model](https://tfhub.dev/google/yamnet/1).  
It can process both **uploaded files** and **remote files via URL** to:

- **Extract or convert audio** to WAV format
- **Segment** audio into fixed-length chunks
- **Classify** segments into sound categories
- **Optionally detect dynamic segments** for a specific target label (e.g., "Music")
- Return **JSON results** containing:
  - `timeline`: Classification results for all segments
  - `dynamic_segments`: Detected target label segments

---

## 🚀 Features

- Works with both **video** (`.mp4`, `.mov`, `.avi`, `.mkv`) and **audio** (`.wav`, `.mp3`, etc.) files
- Supports file input via:
  - **File upload** (`/process-file-from-upload`)
  - **Remote file URL** (`/process-file-from-url`)
- **Automatic audio extraction/conversion** to mono, 16kHz WAV using ffmpeg
- **Optional** dynamic segmentation for target labels
- Runs **fully offline** if YAMNet model is stored locally
- Detailed logging for each processing step

---

## 📦 Requirements

- Python 3.9+
- Virtual environment recommended
- ffmpeg (via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), no system installation required)

---

## 🛠 Installation

```bash
# Clone repository
git clone <this_git_repo_url>
cd ai-audio-segmentation-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YAMNet model (local copy)
# Place it in the project root as: yamnet-tensorflow2-yamnet-v1/
```
---

## ▶️ Running the API

```bash
uvicorn main_server:app --host 0.0.0.0 --port 8000
```

Open **Swagger UI** for interactive documentation:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoints

### 1️⃣ `POST /process-file-from-upload`

Upload an audio or video file.

**Parameters**

| Name             | Type       | Description                                             | Required |
| ---------------- | ---------- | ------------------------------------------------------- | -------- |
| `file`           | UploadFile | The file to process                                     | ✅        |
| `segment_length` | int        | Segment length in seconds (default: `10`)               | ❌        |
| `target_label`   | str        | Target label for dynamic segmentation (e.g., `"Music"`) | ❌        |
| `threshold`      | float      | Confidence threshold for dynamic segmentation           | ❌        |

**Example (via `curl`):**

```bash
curl -X POST "http://127.0.0.1:8000/process-file-from-upload" \
  -F "file=@test.mp4" \
  -F "segment_length=10" \
  -F "target_label=Music" \
  -F "threshold=0.5"
```

**Example Response:**

```json
{
  "timeline": [
    {"start": 0, "end": 10, "label": "Speech", "score": 0.92},
    {"start": 10, "end": 20, "label": "Music", "score": 0.88}
  ],
  "dynamic_segments": [
    {"start": 10, "end": 20, "label": "Music", "score": 0.88}
  ]
}
```

---

### 2️⃣ `POST /process-file-from-url`

Process a file located at a remote URL.

**Request Body**

| Field                    | Type  | Description                                   | Required |
| ------------------------ | ----- | --------------------------------------------- | -------- |
| `file_url`               | str   | Direct link to audio/video file               | ✅        |
| `segment_length_seconds` | int   | Segment length in seconds (default: `10`)     | ❌        |
| `target_label`           | str   | Target label for dynamic segmentation         | ❌        |
| `threshold`              | float | Confidence threshold for dynamic segmentation | ❌        |

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/process-file-from-url" \
  -H "Content-Type: application/json" \
  -d '{
        "file_url": "https://example.com/test.mp4",
        "segment_length_seconds": 10,
        "target_label": "Music",
        "threshold": 0.5
      }'
```

---

## ⚙️ Dynamic Segmentation Rules

* If `target_label` is omitted or empty → Dynamic segmentation is **skipped**
* If skipped → `"dynamic_segments": []` in the response
* If `threshold` is omitted → defaults to **`0.5`**

---

## 📝 Logging

Processing steps are logged both to the console and to `server.log`, including:

* File reception/download
* Audio extraction/conversion
* Segmentation & classification
* Dynamic segment detection
* Cleanup of temp files

---

## 📂 Project Structure

```bash
.
├── main_server.py           # FastAPI application
├── utils.py                 # Audio processing utilities
├── yamnet-tensorflow2-yamnet-v1/  # Local YAMNet model
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License. See **LICENSE** for details.

