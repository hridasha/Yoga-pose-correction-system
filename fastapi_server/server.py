from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import logging
import numpy as np
import json
import signal
import sys

from pose_utils import PoseDetector
from pose_correction import PoseCorrection

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def shutdown_handler(signum, frame):
    logger.info("Shutting down server...")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = set()

MODEL_PATH = r"D:\YogaPC\ypc\datasets\final_student_model_35.keras"
POSE_CLASSES_PATH = r"D:\YogaPC\ypc\datasets\pose_classes.pkl"

pose_detector = PoseDetector(model_path=MODEL_PATH, pose_classes_path=POSE_CLASSES_PATH)
pose_correction = PoseCorrection()

async def process_frame(websocket: WebSocket):
    """
    Captures frames from the camera, processes them, and streams them to the client via WebSocket.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        await websocket.close()
        return

    logger.info("Camera opened successfully")
    detector = pose_detector
    detected_pose = False
    detected_view = False

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.resize(frame, (640, 480))

            landmarks = await detector.process_frame(frame)

            if landmarks:
                frame = detector.draw_pose_landmarks(frame, landmarks)

            await detector.print_stable_coordinates()

            if not detected_pose or not detected_view:
                if detector.current_pose and detector.current_view:
                    detected_pose = True
                    detected_view = True
                    angles = detector.calculate_pose_angles()
                    ideal_angles = await detector.get_ideal_angles(detector.current_pose)
                    if ideal_angles:
                        errors = detector.calculate_error(angles, ideal_angles)
                        feedback = detector.generate_feedback(errors)

                        await websocket.send_text(json.dumps({
                            "idealAngles": ideal_angles,
                            "corrections": feedback if feedback else "NO FEEDBACK",
                        }))

            _, buffer = cv2.imencode(".jpg", frame, 
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 90, 
                                      int(cv2.IMWRITE_JPEG_OPTIMIZE), 1, 
                                      int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
            
            frame_bytes = buffer.tobytes()

            await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        cap.release()
        logger.info("Camera released")

        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info("Removed WebSocket from active connections")

        await websocket.close()




async def process_websocket(websocket):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully")
    corrector = pose_correction
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        frame = cv2.resize(frame, (640, 480))
        
        landmarks = corrector.process_correction(frame)
        
        if landmarks:
            frame = corrector.draw_pose_landmarks(frame, landmarks)
        
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
            await asyncio.sleep(0.01) 
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            break

    cap.release()
    logger.info("Camera released")





@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    """
    logger.info("New WebSocket connection attempt")

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        active_connections.add(websocket)
        logger.info(f"Active connections: {len(active_connections)}")

        await process_frame(websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info("WebSocket removed from active connections")

        await websocket.close()
        logger.info("WebSocket connection closed")


@app.websocket("/ws/correction")
async def correction_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        await process_websocket(websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
    finally:
        await websocket.close()


@app.get("/status")
async def server_status():
    """
    Endpoint to check server status.
    """
    return {"status": "running", "active_connections": len(active_connections)}