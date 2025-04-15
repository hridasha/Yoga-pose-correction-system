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
        
import pyttsx3
# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1)  # Set volume

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.getLogger("comtypes").setLevel(logging.WARNING)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

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


pose_corrector = PoseCorrection()


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
            
            # pose name
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
                                
                            # Find highest error and speak feedback
                            if errors:
                                highest_error = max(errors.items(), key=lambda x: x[1]['error']) if errors else None
                                if highest_error:
                                    angle_name = highest_error[0]
                                    error_value = highest_error[1]['error']
                                    actual_angle = highest_error[1]['actual']
                                    target_angle = highest_error[1]['target']
                                    
                                    # Generate speech feedback
                                    speech_text = f"Adjust your {angle_name.lower()} to {target_angle:.1f} degrees. Current angle is {actual_angle:.1f} degrees with an error of {error_value:.1f} degrees."
                                    
                                    # Speak feedback
                                    engine.say(speech_text)
                                    engine.runAndWait()
                                    
                                    # Add cooldown period
                                    detector.feedback_timer_start = time.time() + 5  # Add 5 seconds delay before next feedback
                        else:
                            print("[WARNING] No ideal angles available for feedback")

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
            if frame_count % 30 == 0:  # Send data more frequently
                try:
                    if detector.current_pose:
                        current_angles = detector.calculate_pose_angles()
                        if current_angles:
                            ideal_angles = await detector.get_ideal_angles(detector.current_pose)
                            if ideal_angles:
                                errors = detector.calculate_angle_errors(current_angles, ideal_angles)
                            else:
                                errors = None
                        else:
                            errors = None
                    else:
                        current_angles = None
                        ideal_angles = None
                        errors = None

                    await websocket.send_text(json.dumps({
                        "pose_name": detector.current_pose,
                        "landmarks": landmarks if landmarks is not None else None,
                        "detected_pose": detector.current_pose,
                        "detected_view": detector.current_view,
                        "idealAngles": ideal_angles,
                        "errors": errors,
                        "current_angles": current_angles
                    }, cls=NumpyJSONEncoder))
                except Exception as e:
                    print(f"Error sending data: {e}")
                    continue

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
        
        


async def process_websocket(websocket: WebSocket, pose_name: str):
    frame_count = 0
    feedback_interval = 10  # every 10th frame at 2 FPS = 5 seconds
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Camera opened successfully")
    corrector = pose_corrector

    previous_landmarks = None
    stable_time = 0
    stability_threshold = 0.5
    tolerance_range = 5
    stability_done = False
    ideal_angles_selected = False
    fixed_ideal_angles = None
    last_frame_for_feedback = None
    view_classified = False

    cooldown_start_time = None
    fps = 30
    delay = 1 / fps

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            print(f"\n=== FRAME {frame_count} ===")
            print(f"Current FPS: {fps:.2f}")
            print(f"Pose Name: {pose_name}")
            print(f"Stability Status: {stability_done}")
            print(f"View Classified: {view_classified}")
            print(f"Ideal Angles Selected: {ideal_angles_selected}")

            # Send frame to frontend
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            try:
                await websocket.send_bytes(frame_bytes)
                await asyncio.sleep(delay)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break

            if frame_count % feedback_interval != 0:
                print("[INFO] Skipping frame - not a feedback interval")
                continue

            print("\n=== PROCESSING FRAME ===")
            print(f"Processing frame {frame_count} for pose detection...")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = corrector.pose.process(rgb_frame)

            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected")
                continue

            # Extract landmarks
            landmarks = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i in corrector.keypoints:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    visibility = landmark.visibility
                    landmarks[i] = (x, y, z, visibility)

            if not landmarks:
                print("[WARNING] No valid landmarks extracted")
                continue

            # Debug: Print landmarks for 10th frame
            if frame_count == 10:
                print("\n=== LANDMARKS FOR FRAME 10 ===")
                for idx, (x, y, z, conf) in landmarks.items():
                    print(f"Landmark {idx}: X={x}, Y={y}, Z={z:.4f}, Confidence={conf:.2f}")

            # Draw keypoints and connections
            for idx, (x, y, z, conf) in landmarks.items():
                if conf > 0.5:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for p1, p2 in corrector.connections:
                if p1 in landmarks and p2 in landmarks:
                    x1, y1, _, _ = landmarks[p1]
                    x2, y2, _, _ = landmarks[p2]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # === Stability Check Once ===
            if not stability_done and previous_landmarks:
                stable_points = sum(
                    1 for idx in corrector.keypoints
                    if idx in landmarks and idx in previous_landmarks
                    and abs(landmarks[idx][0] - previous_landmarks[idx][0]) <= tolerance_range
                    and abs(landmarks[idx][1] - previous_landmarks[idx][1]) <= tolerance_range
                )
                print(f"\n=== STABILITY CHECK ===")
                print(f"Stable Points: {stable_points}/{len(corrector.keypoints)}")
                print(f"Stable Time: {stable_time:.2f} seconds")

                if stable_points >= 7:
                    stable_time += 1 / fps
                    print(f"Updated Stable Time: {stable_time:.2f} seconds")
                    if stable_time >= stability_threshold:
                        print("\n=== POSE STABLE ===")
                        print("Pose is stable --------->> STARTING CORRECTION")
                        stability_done = True
                        last_frame_for_feedback = landmarks.copy()

                        # View classification
                        if not view_classified:
                            corrector.current_view = corrector.classify_view(landmarks)
                            view_classified = True
                            print(f"\n=== VIEW CLASSIFICATION ===")
                            print(f"View classified as: {corrector.current_view}")

                        # Load ideal angles
                        if not ideal_angles_selected:
                            try:
                                print("\n=== GETTING IDEAL ANGLES ===")
                                ideal_data = await corrector.get_ideal_angles(pose_name, landmarks)
                                fixed_ideal_angles = ideal_data
                                ideal_angles_selected = True
                                print("Ideal angles selected")
                                
                                # Debug: Print ideal angles for 10th frame
                                if frame_count == 10:
                                    print("\n=== IDEAL ANGLES FOR FRAME 10 ===")
                                    for angle, data in ideal_data.items():
                                        print(f"{angle}: Mean={data['mean']:.1f}, Min={data['min']:.1f}, Max={data['max']:.1f}")

                            except Exception as e:
                                logger.error(f"Failed to get ideal angles: {e}")
                                continue

                        # First feedback
                        if last_frame_for_feedback and fixed_ideal_angles:
                            print("\n=== CALCULATING ANGLES ===")
                            angles = corrector.calculate_pose_angles(last_frame_for_feedback)
                            
                            # Debug: Print calculated angles for 10th frame
                            if frame_count == 10:
                                print("\n=== CALCULATED ANGLES FOR FRAME 10 ===")
                                for angle, value in angles.items():
                                    print(f"{angle}: {value:.1f} degrees")

                            errors = corrector.calculate_angle_errors(angles, fixed_ideal_angles)
                            corrector.process_feedback_queue(errors)
                            print("\n=== FEEDBACK ===")
                            print("Initial feedback sent")

                        # Begin cooldown
                        cooldown_start_time = time.time()

                        # Switch to 2 FPS
                        fps = 2
                        delay = 1 / fps
                        frame_count = 0
                        print("\n=== FPS SWITCH ===")
                        print("Switched to 2 FPS for feedback mode")
                        continue

            # Wait 5 seconds for user adjustment after feedback
            if stability_done and cooldown_start_time:
                elapsed = time.time() - cooldown_start_time
                if elapsed < 5:
                    print(f"\n=== WAITING ===")
                    print(f"Waiting {5 - elapsed:.2f} seconds for user to adjust pose...")
                    continue
                else:
                    cooldown_start_time = None
                    print("\n=== RESUMING FEEDBACK ===")
                    print("User adjustment time complete. Resuming feedback.")

            # Feedback every 5 seconds (i.e., every 10th frame at 2 FPS)
            if stability_done and ideal_angles_selected and view_classified:
                last_frame_for_feedback = landmarks.copy()
                angles = corrector.calculate_pose_angles(last_frame_for_feedback)
                errors = corrector.calculate_angle_errors(angles, fixed_ideal_angles)
                corrector.process_feedback_queue(errors)
                print(f"\n=== FEEDBACK SENT ===")
                print(f"Feedback sent at {time.strftime('%X')}")
                try:
                    await websocket.send_text(json.dumps({
                        "pose_name": pose_name,
                        "landmarks": landmarks,
                        "correction": corrector.generate_feedback(landmarks),
                        "idealAngles": fixed_ideal_angles,
                        "errors": errors,
                        "stable_points": stable_points,
                        "stable_time": stable_time,
                        "is_stable": stable_time >= stability_threshold
                    }, cls=NumpyJSONEncoder))
                except Exception as e:
                    logger.error(f"Error sending websocket message: {str(e)}")

            previous_landmarks = landmarks.copy()

    except Exception as e:
        logger.error(f"Unhandled error in process_websocket: {e}")
    finally:
        cap.release()
        engine.stop()
        logger.info("WebSocket closed")


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