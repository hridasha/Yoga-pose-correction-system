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
import time
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
    frame_count=0

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.resize(frame, (640, 480))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.pose.process(rgb_frame)
            landmarks = []

            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    confidence = landmark.visibility
                    landmarks.append((x, y, z, confidence))
            else:
                continue

            print(f"Current FPS: {detector.fps:.2f}")
            print(f"Pose Detected: {detector.detected_pose}, View Detected: {detector.detected_view}, Match Calculated: {detector.calculated_best_match}")
            
            # Print pose name if detected
            if detector.current_pose:
                print(f"Current Pose: {detector.current_pose}")

            # === Step 1: After pose and view are classified ===
            if detector.detected_pose and detector.detected_view and detector.ideal_angles_selected and not detector.calculated_best_match:
                current_angles = detector.calculate_pose_angles()
                detector.fixed_ideal_angles = await detector.get_ideal_angles(detector.current_pose)
                detector.calculated_best_match = True
                print("\n[INFO] Best angle match calculated.")

            if detector.calculated_best_match:
                if not detector.stability_disabled:
                    print("\n[INFO] Stability check disabled. Entering feedback loop.")
                    detector.fps = 2
                    detector.high_fps = False
                    detector.stability_disabled = True
                    detector.feedback_timer_start = time.time()

                elapsed = time.time() - detector.feedback_timer_start
                print(f"[INFO] Waiting for user adjustment... {elapsed:.2f}/5.00 seconds")

                if elapsed >= 5:
                    print("\n[INFO] Generating feedback after 5 seconds...")
                    detector.feedback_timer_start = time.time()  # Reset for next feedback round

                    if detector.last_frame_for_feedback:
                        angles = detector.calculate_pose_angles()
                        if detector.fixed_ideal_angles:
                            errors = detector.calculate_angle_errors(angles, detector.fixed_ideal_angles)
                            feedback = detector.process_feedback_queue(errors)
                            print("\n[DEBUG] Feedback:")
                            for angle_name, error in errors.items():
                                within_range = "✓" if error['within_range'] else "✗"
                                print(
                                    f"{angle_name}: Error={error['error']:.1f}°, "
                                    f"Actual={error['actual']:.1f}°, Target={error['target']:.1f}°, "
                                    f"Range=[{error['min']:.1f}°-{error['max']:.1f}°] ({within_range})"
                                )
                        else:
                            print("[WARNING] No ideal angles available for feedback")

            # === Step 2: Pre-classification stability check ===
            if detector.previous_landmarks:
                stable_points = 0
                for idx in detector.keypoints:
                    if landmarks[idx] and detector.previous_landmarks[idx]:
                        x, y = landmarks[idx][:2]
                        prev_x, prev_y = detector.previous_landmarks[idx][:2]
                        if (prev_x - detector.tolerance_range <= x <= prev_x + detector.tolerance_range and
                            prev_y - detector.tolerance_range <= y <= prev_y + detector.tolerance_range):
                            stable_points += 1

                print(f"\nStable Points: {stable_points}/{len(detector.keypoints)}")

                if stable_points >= 7:
                    detector.stable_time += 1 / detector.fps

                    if detector.stable_time >= detector.stability_threshold:
                        print("\n[INFO] Pose Stable!")
                        print(f"Stable for {detector.stable_time:.2f} seconds")
                        
                        detector.last_frame_for_feedback = landmarks
                        detector.stable_coordinates = landmarks
                        detector.stable_time = 0.0
                        await detector.print_stable_coordinates()

                        # Classify pose and view only once
                        if not detector.detected_pose and not detector.detected_view:
                            pose_class, confidence = await detector.classify_pose(landmarks)
                            print(f"\n[INFO] Pose Classified: {pose_class} (Confidence: {confidence:.2f})")
                            detector.current_pose = pose_class
                            detector.detected_pose = True

                            detector.current_view = detector.classify_view(landmarks)
                            print(f"[INFO] View Classified: {detector.current_view}")
                            detector.detected_view = True

                            detector.fixed_ideal_angles = await detector.get_ideal_angles(detector.current_pose)
                            detector.ideal_angles_selected = True
                            print("\n[INFO] Ideal angles loaded.")

            detector.previous_landmarks = landmarks

            frame = detector.draw_pose_landmarks(frame, landmarks)

            await detector.print_stable_coordinates()
            
            frame_count +=1
            if frame_count % 150 == 0:
                await websocket.send_text(json.dumps({
                    "pose_name": detector.current_pose,
                    "landmarks": landmarks if landmarks is not None else None,
                    "detected_pose": detector.current_pose,
                    "detected_view": detector.current_view,
                    "idealAngles": await detector.get_ideal_angles(detector.current_pose) if detector.current_pose else None,
                    "errors": detector.calculate_error(detector.calculate_pose_angles(), detector.get_ideal_angles(detector.current_pose)) if detector.current_pose else None
                }))
            # if not detected_pose or not detected_view:
            #     if detector.current_pose and detector.current_view:
            #         detected_pose = True
            #         detected_view = True
            #         angles = detector.calculate_pose_angles()
            #         ideal_angles = await detector.get_ideal_angles(detector.current_pose)
            #         if ideal_angles:
            #             errors = detector.calculate_error(angles, ideal_angles)
            #             feedback = detector.generate_feedback(errors)

            #             await websocket.send_text(json.dumps({
            #                 "idealAngles": ideal_angles,
            #                 "corrections": feedback if feedback else "NO FEEDBACK",
            #             }))

            _, buffer = cv2.imencode(".jpg", frame, 
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 90, 
                                      int(cv2.IMWRITE_JPEG_OPTIMIZE), 1, 
                                      int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
            
            frame_bytes = buffer.tobytes()

            try:
                await websocket.send_bytes(frame_bytes)
            except Exception as e:
                logger.error(f"Error sending frame: {str(e)}")
                break

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
        
        
        
import pyttsx3
# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate



async def process_websocket(websocket: WebSocket, pose_name: str):
    frame_count = 0
    feedback_interval = 10  # Initial 2 FPS processing
    cooldown_duration = 5  # 5 seconds cooldown
    last_feedback_time = 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully")
    corrector = pose_correction
    detected_view = False
    previous_landmarks = None
    stable_time = 0
    stability_threshold = 0.5  # 0.5 seconds
    tolerance_range = 5
    ideal_angles_selected = False
    fixed_ideal_angles = None
    last_frame_for_feedback = None
    first_feedback_given = False
    view_classified = False  # Track if view has been classified
     
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.resize(frame, (640, 480))
            
            frame_count += 1
            
            # Process every frame for video stream
            _, buffer = cv2.imencode(
                ".jpg", 
                frame, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 75, 
                 int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  
                 int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]  
            )
            frame_bytes = buffer.tobytes()
            
            try:
                await websocket.send_bytes(frame_bytes)
                await asyncio.sleep(0.01) 
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            
            # Only process correction every feedback_interval frames
            if frame_count % feedback_interval == 0:
                landmarks = await corrector.process_correction(frame, pose_name)
                
                if landmarks is not None:
                    # Draw landmarks on the frame
                    for idx, (x, y, z, confidence) in landmarks.items():
                        if confidence > 0.5:
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Draw connections between landmarks
                    for (p1, p2) in corrector.connections:
                        if p1 in landmarks and p2 in landmarks:
                            x1, y1, _, _ = landmarks[p1]
                            x2, y2, _, _ = landmarks[p2]
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Check for stability
                    if previous_landmarks:
                        stable_points = sum(
                            1 for idx in corrector.keypoints
                            if idx in landmarks and idx in previous_landmarks
                            and abs(landmarks[idx][0] - previous_landmarks[idx][0]) <= tolerance_range
                            and abs(landmarks[idx][1] - previous_landmarks[idx][1]) <= tolerance_range
                        )

                        if stable_points >= 7:  # 7 stable points required
                            stable_time += 1 / corrector.fps
                            if stable_time >= stability_threshold:
                                logger.info(f"Pose Stable! Stable for {stable_time:.2f} seconds")
                                
                                # Store stable coordinates
                                stable_coordinates = landmarks.copy()
                                
                                # Only classify view once when pose becomes stable
                                if not view_classified:
                                    view = corrector.classify_view(stable_coordinates)
                                    logger.info(f"View classified as: {view}")
                                    view_classified = True
                                
                                # Calculate angles and errors
                                angles = corrector.calculate_pose_angles(landmarks)
                                ideal_angles = await corrector.get_ideal_angles(pose_name, landmarks)
                                
                                if not ideal_angles_selected:
                                    fixed_ideal_angles = ideal_angles
                                    ideal_angles_selected = True
                                    last_frame_for_feedback = landmarks.copy()

                                errors = corrector.calculate_error(angles, ideal_angles)
                                
                                # Check if we're in cooldown period
                                current_time = time.time()
                                if current_time - last_feedback_time > cooldown_duration:
                                    if errors:
                                        highest_error = max(errors.items(), key=lambda x: x[1]['error']) if errors else None
                                        if highest_error:
                                            angle_name = highest_error[0]
                                            error_value = highest_error[1]['error']
                                            
                                            # Generate speech feedback
                                            if 'Elbow' in angle_name:
                                                speech_text = f"Bend your {angle_name} slightly. error with {error_value:.2f}"
                                            else:
                                                speech_text = f"Adjust your {angle_name} to be more closed. error with {error_value:.2f}"
                                            
                                            # Speak feedback and update last feedback time
                                            engine.say(speech_text)
                                            engine.runAndWait()
                                            last_feedback_time = current_time
                                            
                                            # After first feedback, switch to lower processing rate
                                            if not first_feedback_given:
                                                feedback_interval = 100  # Process every 100 frames (about 0.3 FPS)
                                                first_feedback_given = True
                                
                                # Add stability information to frame
                                cv2.putText(frame, f"Stable Points: {stable_points}/16", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(frame, f"Stable Time: {stable_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(frame, f"Pose: {pose_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                # Send detailed information in websocket message
                                await websocket.send_text(json.dumps({
                                    "pose_name": pose_name,
                                    "landmarks": landmarks,
                                    "corrections": corrector.generate_feedback(landmarks),
                                    "detected_pose": corrector.current_pose,
                                    "detected_view": corrector.current_view,
                                    "idealAngles": ideal_angles,
                                    "errors": errors,
                                    "stable_points": stable_points,
                                    "stable_time": stable_time,
                                    "is_stable": stable_time >= stability_threshold
                                }))
                    
                    # Update previous landmarks
                    previous_landmarks = landmarks
            
    finally:
        cap.release()
        engine.stop()
        logger.info("Camera released")


@app.websocket("/ws/correction/{pose_name:path}")
async def websocket_endpoint(websocket: WebSocket, pose_name: str):
    await websocket.accept()
    active_connections.add(websocket)
    
    pose_name = pose_name.replace('%20', ' ')  
    logger.info(f"WebSocket connected for pose: {pose_name}")
    
    try:
        await process_websocket(websocket, pose_name)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info("Removed WebSocket from active connections")


@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("WebSocket connected for live video stream")
    try:
        await process_frame(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info("Removed WEbsocket from active connections")
    
    
    


# @app.websocket("/ws/correction")
# async def correction_endpoint(websocket: WebSocket):
#     logger.info("New WebSocket connection attempt")

#     try:
#         await websocket.accept()
#         logger.info("WebSocket connection accepted")

#         query_params = dict(websocket.url.query)
#         pose_name = query_params.get('pose_name', 'unknown')
        
#         logger.info(f"Processing correction for pose: {pose_name}")
        
#         await process_frame(websocket, pose_name)

#     except WebSocketDisconnect:
#         logger.info("WebSocket disconnected")
#     except Exception as e:
#         logger.error(f"WebSocket error: {str(e)}", exc_info=True)
#     finally:
#         await websocket.close()


@app.get("/status")
async def server_status():
    """
    Endpoint to check server status.
    """
    return {"status": "running", "active_connections": len(active_connections)}