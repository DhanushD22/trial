# listener/signal_listener.py

from fastapi import FastAPI
from prometheus_client import generate_latest
from fastapi.responses import Response
import threading

app = FastAPI()

# These will be injected from main.py
start_callback = None
stop_callback = None


@app.post("/start")
def start_recording():
    if start_callback:
        threading.Thread(target=start_callback).start()
        return {"status": "START signal received"}
    return {"error": "Start callback not registered"}


@app.post("/stop")
def stop_recording():
    if stop_callback:
        threading.Thread(target=stop_callback).start()
        return {"status": "STOP signal received"}
    return {"error": "Stop callback not registered"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

