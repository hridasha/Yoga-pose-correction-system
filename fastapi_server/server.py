from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import logging
import numpy as np
import io
from typing import List, Dict, Tuple
from pose_utils import PoseDetector
import signal
import sys


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add these imports at the top of your file
def shutdown_handler(signum, frame):
    logger.info("Shutting down server...")
    sys.exit(0)

# Add this before app creation
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

app = FastAPI()

# Add CORS Middleware to handle Django-FastAPI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Allow Django origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections = set()

# Initialize pose detector only when needed
pose_detector = None

def get_pose_detector():
    """
    Initializes the PoseDetector once and reuses the same instance
    to optimize performance.
    """
    global pose_detector
    if pose_detector is None:
        pose_detector = PoseDetector()
        logger.info("PoseDetector initialized")
    return pose_detector

async def process_frame(websocket: WebSocket):
    """
    Continuously captures frames from the camera, processes them for pose detection,
    draws landmarks, prints coordinates, and streams them to the client via WebSockets.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully")
    detector = get_pose_detector()

    try:
        while True:
            try:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Process frame
                landmarks = detector.process_frame(frame)
                
                # Draw landmarks
                if landmarks:
                    frame = detector.draw_pose_landmarks(frame, landmarks)

                # Print coordinates if stable
                detector.print_stable_coordinates()

                # Encode frame with optimized quality settings
                _, buffer = cv2.imencode(
                    ".jpg", 
                    frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90,  # Higher quality
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,   # Enable optimization
                     int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]  # Progressive encoding
                )
                
                frame_bytes = buffer.tobytes()
                
                # Send frame
                await websocket.send_bytes(frame_bytes)
                
                # Add small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)  # Reduced delay for smoother streaming
                
            except Exception as e:
                logger.error(f"Error in frame processing: {str(e)}")
                continue
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
    finally:
        cap.release()
        logger.info("Camera released")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that handles real-time video streaming.
    """
    logger.info("WebSocket connection attempt")
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        active_connections.add(websocket)
        logger.info(f"Active connections: {len(active_connections)}")
        
        try:
            await process_frame(websocket)
            
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            active_connections.remove(websocket)
            
        except Exception as e:
            logger.error(f"Error in WebSocket endpoint: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection: {str(e)}")


@app.get("/status")
async def server_status():
    """
    Endpoint to check server status.
    """
    return {"status": "running", "active_connections": len(active_connections)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)