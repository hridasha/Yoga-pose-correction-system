import os
import signal
import subprocess
import time

PID_FILE = "fastapi_server.pid"

def start_fastapi_server():
    """ Start the FastAPI server if not running """
    if os.path.exists(PID_FILE):
        print(" FastAPI server is already running.")
        return

    print(" Starting FastAPI server...")

    process = subprocess.Popen(
        ["uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8001", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True  # Avoid zombie processes
    )

    # Save the PID to file
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    # Give the server time to start
    time.sleep(2)
    print(" FastAPI server started.")


def stop_fastapi_server():
    """ Stop the FastAPI server gracefully """
    if not os.path.exists(PID_FILE):
        print(" No FastAPI server is running.")
        return

    # Read the PID and terminate the process
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f" FastAPI server (PID {pid}) stopped.")
    except ProcessLookupError:
        print(" Failed to stop FastAPI server.")
    
    # Remove PID file
    os.remove(PID_FILE)
