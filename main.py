# main.py

import threading
import uvicorn
import listener.signal_listener as signal_listener

from camera.camera_manager import CameraManager
from processor.model_loader import load_model
from config.settings import generate_session_id
from processor.video_processor import process_video


STATE = "IDLE"

camera = CameraManager()
predictor = load_model()

current_session_id = None
current_video_path = None


def start_system():
    global STATE, current_session_id, current_video_path

    if STATE != "IDLE":
        print("⚠ System not idle. Cannot start.")
        return

    print("🚦 START signal received")

    if not camera.is_camera_available():
        print("❌ Camera not available")
        return

    current_session_id = generate_session_id()
    current_video_path = camera.start_recording(current_session_id)

    STATE = "RECORDING"


def stop_system():
    global STATE, current_video_path, current_session_id

    if STATE != "RECORDING":
        print("⚠ System not recording. Cannot stop.")
        return

    print("🛑 STOP signal received")

    video_path = camera.stop_recording()
    STATE = "PROCESSING"

    threading.Thread(
        target=run_processing,
        args=(video_path, current_session_id),
    ).start()


def run_processing(video_path, session_id):
    global STATE

    print("🧠 Starting analysis...")
    process_video(video_path, session_id, predictor)
    print("✅ Processing completed")

    STATE = "IDLE"
    print("🟢 System back to IDLE")


# 🔥 Properly inject callbacks into module variables
signal_listener.start_callback = start_system
signal_listener.stop_callback = stop_system


if __name__ == "__main__":
    print("🚀 Locomotive Inspection System Started")
    uvicorn.run(signal_listener.app, host="0.0.0.0", port=8000)
