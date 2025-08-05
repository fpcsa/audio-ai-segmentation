from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import tensorflow_hub as hub
import mimetypes
import os
import shutil
import logging
import tempfile
from utils import (
    load_yamnet_labels,
    download_file,
    cleanup_temp_files,
    process_audio_pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Audio Segmentation API",
    description="""
Processes audio or video files to extract and analyze their audio tracks using Google's YAMNet model.

**Main features:**
- Extracts or converts audio to WAV
- Segments audio into fixed-length chunks
- Classifies each segment
- Detects dynamic segments for specific labels
- Returns results as JSON
""",
    version="1.0.0",
)

class ProcessFileFromUrlRequest(BaseModel):
    file_url: str
    segment_length_seconds: int = 10
    target_label: Optional[str] = None
    threshold: Optional[float] = None

class ProcessingResponse(BaseModel):
    timeline: List[Dict[str, Any]]
    dynamic_segments: List[Dict[str, Any]]

# Load YAMNet model offline
MODEL_PATH = "yamnet-tensorflow2-yamnet-v1"
model = hub.load(MODEL_PATH)
yamnet_labels = load_yamnet_labels(MODEL_PATH)

@app.post("/process-file-from-upload", response_model=ProcessingResponse)
async def process_file_from_upload(
    file: UploadFile,
    segment_length: int = Form(10),
    target_label: str = Form("Music"),
    threshold: float = Form(0.5)
):
    """
    Upload an audio or video file for audio segmentation, classification, and dynamic segment detection.

    **Workflow:**
    1. Save the uploaded file to a temporary location.
    2. Convert or extract the audio track to WAV format if needed.
    3. Segment the audio into fixed-length chunks.
    4. Classify each audio segment using YAMNet.
    5. Detect dynamic segments containing the specified `target_label` above the given `threshold`.

    **Args:**
    - **file**: The uploaded audio or video file (MP4, MP3, WAV, etc.).
    - **segment_length**: Length of each segment in seconds (default: 10).
    - **target_label**: The label to detect dynamically (default: `"Music"`).
    - **threshold**: Confidence threshold for dynamic detection (default: 0.5).

    **Returns:**
    - **timeline**: A list of classified audio segments with timestamps and labels.
    - **dynamic_segments**: A list of detected dynamic segments for the given `target_label`.
    """
    temp_file_path = None
    audio_path = None
    try:
        logger.info(f"Received request to process file upload: {file.filename}")

        # Guess extension
        ext = os.path.splitext(file.filename.split("?")[0])[1].lower()
        if not ext:
            mime_type, _ = mimetypes.guess_type(file.filename)
            ext = mimetypes.guess_extension(mime_type) if mime_type else ".dat"

        # Create temp file for uploaded data
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_video:
            temp_file_path = tmp_video.name
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {temp_file_path}")

        # Process pipeline
        timeline, dynamic_segments, audio_path = process_audio_pipeline(
            input_path=temp_file_path,
            ext=ext,
            segment_length=segment_length,
            target_label=target_label if target_label else None,
            threshold=threshold if threshold is not None else None,
            model=model,
            yamnet_labels=yamnet_labels
        )

        return JSONResponse(content={
            "timeline": timeline,
            "dynamic_segments": dynamic_segments
        })

    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        cleanup_temp_files(temp_file_path, audio_path)

@app.post("/process-file-from-url", response_model=ProcessingResponse)
async def process_file_from_url(request: ProcessFileFromUrlRequest):
    """
    Process an audio or video file located at a remote URL for audio segmentation, classification,
    and dynamic segment detection.

    **Workflow:**
    1. Download the file from the provided URL to a temporary location.
    2. Convert or extract the audio track to WAV format if needed.
    3. Segment the audio into fixed-length chunks.
    4. Classify each audio segment using YAMNet.
    5. Detect dynamic segments containing the specified `target_label` above the given `threshold`.

    **Args:**
    - **file_url**: The full URL to the file (supports audio/video formats like MP4, MP3, WAV).
    - **segment_length_seconds**: Length of each segment in seconds (default: 10).
    - **target_label**: The label to detect dynamically (default: `"Music"`).
    - **threshold**: Confidence threshold for dynamic detection (default: 0.5).

    **Returns:**
    - **timeline**: A list of classified audio segments with timestamps and labels.
    - **dynamic_segments**: A list of detected dynamic segments for the given `target_label`.
    """
    temp_file_path = None
    audio_path = None
    try:
        file_url = request.file_url
        segment_length = request.segment_length_seconds
        target_label = request.target_label
        threshold = request.threshold

        logger.info(f"Received request to process file from URL: {file_url}")

        # Guess extension
        ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if not ext:
            mime_type, _ = mimetypes.guess_type(file_url)
            ext = mimetypes.guess_extension(mime_type) if mime_type else ".dat"

        # Create temp file for downloaded data
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_video:
            temp_file_path = tmp_video.name

        logger.info(f"Downloading file to: {temp_file_path}")
        download_file(file_url, temp_file_path)

        # Process pipeline
        timeline, dynamic_segments, audio_path = process_audio_pipeline(
            input_path=temp_file_path,
            ext=ext,
            segment_length=segment_length,
            target_label=target_label if target_label else None,
            threshold=threshold if threshold is not None else None,
            model=model,
            yamnet_labels=yamnet_labels
        )

        return JSONResponse(content={
            "timeline": timeline,
            "dynamic_segments": dynamic_segments
        })

    except Exception as e:
        logger.error(f"Error processing file from URL: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        cleanup_temp_files(temp_file_path, audio_path)
