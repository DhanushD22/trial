# main.py

import threading
import time
import uvicorn

import listener.signal_listener as signal_listener

from camera.camera_manager import CameraManager
from processor.model_loader import load_model
from processor.video_processor import process_video
from config.settings import generate_session_id

from monitoring.metrics import (
    listener_active,
    model_loaded,
    system_state,
    model_running,
    sessions_processed,
    processing_time,
    system_errors,
)
from monitoring.system_monitor import start_system_monitor


# ================================
# SYSTEM STATE
# ================================
STATE = "IDLE"

camera = CameraManager()
predictor = None

current_session_id = None
current_video_path = None


# ================================
# START SYSTEM
# ================================
def start_system():
    global STATE, current_session_id, current_video_path

    if STATE != "IDLE":
        print("⚠ System not idle. Cannot start.")
        return

    print("🚦 START signal received")

    try:
        if not camera.is_camera_available():
            print("❌ Camera not available")
            system_errors.inc()
            return

        current_session_id = generate_session_id()
        current_video_path = camera.start_recording(current_session_id)

        STATE = "RECORDING"
        system_state.set(1)  # RECORDING

    except Exception as e:
        print("❌ Error during start:", e)
        system_errors.inc()


# ================================
# STOP SYSTEM
# ================================
def stop_system():
    global STATE, current_video_path, current_session_id

    if STATE != "RECORDING":
        print("⚠ System not recording. Cannot stop.")
        return

    print("🛑 STOP signal received")

    try:
        video_path = camera.stop_recording()
        STATE = "PROCESSING"
        system_state.set(2)  # PROCESSING

        threading.Thread(
            target=run_processing,
            args=(video_path, current_session_id),
        ).start()

    except Exception as e:
        print("❌ Error during stop:", e)
        system_errors.inc()


# ================================
# PROCESSING THREAD
# ================================
def run_processing(video_path, session_id):
    global STATE

    try:
        print("🧠 Starting analysis...")
        model_running.set(1)

        start_time = time.time()

        process_video(video_path, session_id, predictor)

        duration = time.time() - start_time
        processing_time.set(duration)

        model_running.set(0)
        sessions_processed.inc()

        print("✅ Processing completed")

    except Exception as e:
        print("❌ Processing error:", e)
        system_errors.inc()

    finally:
        STATE = "IDLE"
        system_state.set(0)  # IDLE
        print("🟢 System back to IDLE")


# ================================
# MAIN ENTRY
# ================================
if __name__ == "__main__":

    print("🚀 Locomotive Inspection System Started")

    # Load model once
    predictor = load_model()
    model_loaded.set(1)

    # Start monitoring thread
    start_system_monitor()

    # Mark listener active
    listener_active.set(1)
    system_state.set(0)  # IDLE

    # Inject callbacks
    signal_listener.start_callback = start_system
    signal_listener.stop_callback = stop_system

    # Start API
    uvicorn.run(signal_listener.app, host="0.0.0.0", port=8000)
