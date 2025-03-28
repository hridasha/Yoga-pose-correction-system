from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import logging
import numpy as np
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

async def process_frame(websocket):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        # Resize frame to reduce bandwidth
        frame = cv2.resize(frame, (640, 480))
        
        # Encode frame with optimized quality settings
        _, buffer = cv2.imencode(
            ".jpg", 
            frame, 
            [int(cv2.IMWRITE_JPEG_QUALITY), 75,  # Medium quality
             int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,   # Enable optimization
             int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]  # Enable progressive encoding
        )
        frame_bytes = buffer.tobytes()
        
        try:
            logger.debug(f"Sending frame of size: {len(frame_bytes)} bytes")
            await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            break

    cap.release()
    logger.info("Camera released")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket connection attempt")
    
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    active_connections.add(websocket)
    logger.info(f"Active connections: {len(active_connections)}")
    
    try:
        await process_frame(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {str(e)}")
    finally:
        active_connections.remove(websocket)
        logger.info(f"Active connections: {len(active_connections)}")

@app.get("/status")
async def server_status():
    return {"status": "running", "active_connections": len(active_connections)}
