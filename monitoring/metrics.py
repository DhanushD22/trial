# monitoring/metrics.py

from prometheus_client import Counter, Gauge

# ---------------------------
# System State Metrics
# ---------------------------
listener_active = Gauge(
    "listener_active",
    "Listener service active status (1=active, 0=inactive)"
)

system_state = Gauge(
    "inspection_system_state",
    "System state (0=IDLE, 1=RECORDING, 2=PROCESSING)"
)

model_loaded = Gauge(
    "model_loaded",
    "Model loaded status (1=loaded)"
)

model_running = Gauge(
    "model_running",
    "Model currently running inference (1=running)"
)

camera_connected = Gauge(
    "camera_connected",
    "Camera connection status (1=connected)"
)

camera_fps = Gauge(
    "camera_fps",
    "Live camera FPS"
)

cpu_usage = Gauge(
    "cpu_usage_percent",
    "CPU usage percent"
)

gpu_usage = Gauge(
    "gpu_usage_percent",
    "GPU usage percent"
)

sessions_processed = Counter(
    "sessions_processed_total",
    "Total sessions processed"
)

defects_detected = Counter(
    "defects_detected_total",
    "Total defects detected"
)

processing_time = Gauge(
    "last_processing_time_seconds",
    "Time taken for last processing"
)

system_errors = Counter(
    "system_errors_total",
    "Total system errors"
)
