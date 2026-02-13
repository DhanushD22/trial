# camera/camera_manager.py

import cv2
import os
import threading
from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    RAW_VIDEO_DIR,
)

class CameraManager:
    def __init__(self):
        self.cap = None
        self.out = None
        self.recording = False
        self.video_path = None
        self.thread = None

    def is_camera_available(self):
        test_cap = cv2.VideoCapture(CAMERA_INDEX)
        if not test_cap.isOpened():
            return False
        test_cap.release()
        return True

    def _record_loop(self):
        while self.recording:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            self.out.write(frame)

            # Show preview
            cv2.imshow("Live Recording", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recording = False
                break

        cv2.destroyAllWindows()

    def start_recording(self, session_id):
        if self.recording:
            print("⚠ Already recording")
            return None

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available")

        os.makedirs(RAW_VIDEO_DIR, exist_ok=True)

        self.video_path = os.path.join(
            RAW_VIDEO_DIR, f"{session_id}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.video_path,
            fourcc,
            FPS,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )

        self.recording = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

        print(f"🎥 Recording started: {self.video_path}")
        return self.video_path

    def stop_recording(self):
        if not self.recording:
            print("⚠ Not recording")
            return None

        self.recording = False
        self.thread.join()

        if self.cap:
            self.cap.release()

        if self.out:
            self.out.release()

        print("🛑 Recording stopped")

        return self.video_path
