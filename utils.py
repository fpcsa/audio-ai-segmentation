import subprocess
import imageio_ffmpeg as ffmpeg
import os
import json
import requests
import logging
# from moviepy import VideoFileClip
import librosa
import numpy as np
import tempfile

logger = logging.getLogger(__name__)

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path

def convert_to_wav_ffmpeg(input_path, output_path):
    """Convert any audio/video file to WAV using embedded ffmpeg binary."""
    ffmpeg_binary = ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_binary,
        "-y",  # always overwrite output
        "-i", input_path,
        "-vn",                # no video
        "-acodec", "pcm_s16le", # WAV format
        "-ar", "16000",       # 16kHz sample rate
        "-ac", "1",           # mono
        output_path
    ]
    subprocess.run(command, check=True)

"""
def extract_audio_old(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()
"""

def segment_audio(audio_path, segment_length=2):
    y, sr = librosa.load(audio_path, sr=None)
    segments = []
    for i in range(0, len(y), int(segment_length * sr)):
        segments.append(y[i:i + int(segment_length * sr)])
    return segments, sr

def load_yamnet_labels(model_path):
    """
    Load YAMNet class labels and create a mapping of class IDs to human-readable names.
    """
    # labels_path = "yamnet_class_map.csv"  # Path to the class map file
    labels_path = os.path.join(model_path, "assets", "yamnet_class_map.csv")
    labels = {}
    with open(labels_path, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",")
            class_id = parts[1]  # Class ID
            display_name = parts[2]  # Human-readable name
            labels[class_id] = display_name
    return labels

def classify_segment(model, yamnet_labels, segment, sr):
    # Resample to 16 kHz if the sample rate is different
    if sr != 16000:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

    # Ensure the segment is a 1D numpy array
    segment = np.array(segment, dtype=np.float32)

    # Run the segment through the YAMNet model
    scores, _, _ = model(segment)
    scores = scores.numpy().mean(axis=0)

    # Load the YAMNet class map
    # yamnet_labels = load_yamnet_labels(model_path)

    # Find the highest-scoring label
    top_index = np.argmax(scores)
    top_id = list(yamnet_labels.keys())[top_index]  # Get the class ID
    top_label = yamnet_labels[top_id]  # Map to human-readable label
    top_score = float(scores[top_index])  # Convert to native Python float

    # print(f"Top Label: {top_label}, Score: {top_score:.4f}")  # Log the label and confidence score

    return {"label": top_label, "score": top_score}

def classify_segment_with_scores(model, yamnet_labels, segment, sr):
    # Resample to 16 kHz if the sample rate is different
    if sr != 16000:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

    # Ensure the segment is a 1D numpy array
    segment = np.array(segment, dtype=np.float32)

    # Run the segment through the YAMNet model
    scores, _, _ = model(segment)
    scores = scores.numpy().mean(axis=0)

    # Load the YAMNet class map
    # yamnet_labels = load_yamnet_labels(model_path)

    # Find the highest-scoring label
    top_index = np.argmax(scores)
    top_id = list(yamnet_labels.keys())[top_index]
    top_label = yamnet_labels[top_id]
    top_score = float(scores[top_index])

    # print(f"Top Label: {top_label}, Score: {top_score:.4f}")

    # Return full scores along with the top label
    return {"label": top_label, "score": top_score, "scores": scores}


def map_segments_to_timeline(results, segment_length):
    timeline = []
    for i, result in enumerate(results):
        start = i * segment_length
        end = start + segment_length
        timeline.append({
            "start": start,
            "end": end,
            "label": result["label"],  # Human-readable label
            "score": result["score"]
        })
    return timeline

def save_timeline_to_json(timeline, output_file):
    """
    Save the timeline to a formatted JSON file.
    
    Args:
        timeline (list): List of timeline entries (start, end, label).
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(timeline, f, indent=4)  # Use indent=4 for pretty formatting

def detect_dynamic_segments(audio_path, sr, model, yamnet_labels, target_label="Music", threshold=0.5, frame_length=0.5):
    """
    Detect dynamic segments for a specific class (e.g., "Music").
    
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate of the audio.
        model (tfhub.Module): Preloaded YAMNet model.
        yamnet_labels (dict): Mapping of class IDs to human-readable labels.
        target_label (str): The class to detect (e.g., "Music").
        threshold (float): Confidence threshold for detecting the class.
        frame_length (float): Length of each frame in seconds.

    Returns:
        list: List of dynamic segments with start and end times.
    """
    # Load the audio file
    y, _ = librosa.load(audio_path, sr=sr)
    frame_size = int(frame_length * sr)
    
    # Initialize variables
    segments = []
    in_segment = False
    current_start = None
    
    for i in range(0, len(y), frame_size):
        # Extract the current frame
        frame = y[i:i + frame_size]
        if len(frame) < frame_size:
            break  # Skip incomplete frames at the end

        # Resample the frame to 16 kHz
        frame = librosa.resample(frame, orig_sr=sr, target_sr=16000)
        frame = np.array(frame, dtype=np.float32)

        # Run the frame through YAMNet
        scores, _, _ = model(frame)
        scores = scores.numpy().mean(axis=0)

        # Get the confidence score for the target label
        target_index = list(yamnet_labels.values()).index(target_label)
        target_score = scores[target_index]

        # Detect transitions
        if target_score > threshold and not in_segment:
            in_segment = True
            current_start = i / sr
        elif target_score <= threshold and in_segment:
            in_segment = False
            segments.append({"start": current_start, "end": i / sr})
            current_start = None

    # Handle case where audio ends during a segment
    if in_segment:
        segments.append({"start": current_start, "end": len(y) / sr})

    return segments


def detect_dynamic_segments_with_scores(audio_path, sr, model, yamnet_labels, target_label="Music", threshold=0.5, frame_length=0.5):
    """
    Detect dynamic segments for a specific class (e.g., "Music") with confidence scores.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate of the audio.
        model (tfhub.Module): Preloaded YAMNet model.
        yamnet_labels (dict): Mapping of class IDs to human-readable labels.
        target_label (str): The class to detect (e.g., "Music").
        threshold (float): Confidence threshold for detecting the class.
        frame_length (float): Length of each frame in seconds.

    Returns:
        list: List of dynamic segments with start time, end time, label, and score.
    """
    # Load the audio file
    y, _ = librosa.load(audio_path, sr=sr)
    frame_size = int(frame_length * sr)

    # Initialize variables
    segments = []
    in_segment = False
    current_start = None
    cumulative_scores = []
    frames_in_segment = 0

    # Get the target index for the desired label
    target_index = list(yamnet_labels.values()).index(target_label)

    for i in range(0, len(y), frame_size):
        # Extract the current frame
        frame = y[i:i + frame_size]
        if len(frame) < frame_size:
            break  # Skip incomplete frames at the end

        # Resample the frame to 16 kHz
        frame = librosa.resample(frame, orig_sr=sr, target_sr=16000)
        frame = np.array(frame, dtype=np.float32)

        # Run the frame through YAMNet
        scores, _, _ = model(frame)
        scores = scores.numpy().mean(axis=0)

        # Get the confidence score for the target label
        target_score = float(scores[target_index])  # Convert to Python float

        # Detect transitions and aggregate scores
        if target_score > threshold:
            if not in_segment:
                # Start a new segment
                in_segment = True
                current_start = i / sr
                cumulative_scores = [target_score]  # Reset cumulative scores
                frames_in_segment = 1
            else:
                # Continue the current segment
                cumulative_scores.append(target_score)
                frames_in_segment += 1
        elif in_segment:
            # End the current segment
            avg_score = sum(cumulative_scores) / frames_in_segment
            segments.append({
                "start": current_start,
                "end": i / sr,
                "label": target_label,
                "score": float(avg_score)  # Ensure avg_score is a Python float
            })
            in_segment = False
            cumulative_scores = []
            frames_in_segment = 0

    # Handle case where audio ends during a segment
    if in_segment:
        avg_score = sum(cumulative_scores) / frames_in_segment
        segments.append({
            "start": current_start,
            "end": len(y) / sr,
            "label": target_label,
            "score": float(avg_score)  # Ensure avg_score is a Python float
        })

    return segments

def cleanup_temp_files(video_path, audio_path):
    """Safely remove temporary video and audio files without double-deleting."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Removed temp file: {video_path}")

        # Remove audio only if it's different from video path
        if audio_path and os.path.exists(audio_path) and audio_path != video_path:
            os.remove(audio_path)
            logger.info(f"Removed temp file: {audio_path}")

    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def process_audio_pipeline(input_path: str, ext: str, segment_length: int, target_label: str, threshold: float,
                           model=None, yamnet_labels=None):
    """
    Process an audio or video file:
    - Convert or extract audio to WAV
    - Segment audio
    - Classify segments
    - Detect dynamic segments

    Args:
        input_path (str): Path to the input file
        ext (str): File extension (lowercase, with dot)
        segment_length (int): Segment length in seconds
        target_label (str, optional): Label to detect in dynamic segmentation
        threshold (float, optional): Confidence threshold for dynamic segmentation        
        model: Preloaded YAMNet model
        yamnet_labels (dict): YAMNet labels dictionary

    Returns:
        tuple: (timeline, dynamic_segments, audio_path)
    """
    # Generate temp audio path (without creating the file)
    audio_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".wav")

    # Extract or convert to WAV
    if ext == ".wav":
        logger.info("WAV file detected, skipping conversion")
        audio_path = input_path
    else:
        convert_to_wav_ffmpeg(input_path, audio_path)

    logger.info(f"Extracted audio to: {audio_path}")

    # Segment audio
    segments, sr = segment_audio(audio_path, segment_length)
    logger.info(f"Segmented audio into {len(segments)} segments")

    # Classify segments
    results = [classify_segment_with_scores(model, yamnet_labels, segment, sr) for segment in segments]
    timeline = map_segments_to_timeline(results, segment_length)
    # save_timeline_to_json(timeline, "timeline.json")
    logger.info("Classification completed")

    # Detect dynamic segments
    dynamic_segments = []
    if target_label and target_label.strip(): # Only run if not empty string
        logger.info(f"Detecting dynamic segments for label '{target_label}' with threshold={threshold}")
        dynamic_segments = detect_dynamic_segments_with_scores(
            audio_path=audio_path,
            sr=16000,
            model=model,
            yamnet_labels=yamnet_labels,
            target_label=target_label,
            threshold=threshold if threshold is not None else 0.5,  # default if missing
            frame_length=0.5
        )
        # save_timeline_to_json(dynamic_segments, "dynamic_segments.json")
        logger.info(f"Detected {len(dynamic_segments)} dynamic segments")
    else:
        logger.info("No target_label provided — skipping dynamic segmentation")

    return timeline, dynamic_segments, audio_path