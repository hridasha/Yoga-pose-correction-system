from asyncio.log import logger
from typing import List, Tuple, Dict ,Optional
import numpy as np
import cv2
import tensorflow as tf
import os
from django.conf import settings
from django.db import connection
from django.http import JsonResponse
import os
import django
import sys
from asgiref.sync import sync_to_async
import time
import logging
import pyttsx3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ypc.settings')
django.setup()
# from pose_selection.models import YogaPoseIdealAngle

try:
    from pose_selection.models import YogaPoseIdealAngle
except ImportError as e:
    print(f"Error importing pose_selection models: {e}")

_mp_pose = None

def _get_mp_pose():
    """Lazy load MediaPipe Pose module."""
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
    return _mp_pose


class PoseCorrection:
    def __init__(self):
        """Initialize MediaPipe pose model."""
        self.stable_coordinates: List[Tuple[float, float, float]] = []  
        self.needs_printing = True
        self.current_pose = None
        self.current_view = None
        self.ideal_angles = None
        self.connections = [
            # Head connections
            (0, 2), (2, 5), (5, 0),   # Nose ↔ Eyes ↔ Nose
            (2, 7), (5, 8),           # Eyes ↔ Ears
            
            # Torso connections
            (11, 12),                 # Shoulders
            (23, 24),                 # Hips
            (11, 23), (12, 24),       # Shoulders ↔ Hips
            
            # Arms connections
            (11, 13), (13, 15),       # Left Shoulder → Elbow → Wrist
            (12, 14), (14, 16),       # Right Shoulder → Elbow → Wrist
            
            # Legs connections
            (23, 25), (25, 27),       # Left Hip → Knee → Ankle
            (24, 26), (26, 28)      # Right Hip → Knee → Ankle
        ]
        self.keypoints = {
            0: "Nose",
            2: "Left Eye (Inner)",
            5: "Right Eye (Inner)",
            7: "Left Ear",
            8: "Right Ear",
            11: "Left Shoulder",
            12: "Right Shoulder",
            13: "Left Elbow",
            14: "Right Elbow",
            15: "Left Wrist",
            16: "Right Wrist",
            23: "Left Hip",
            24: "Right Hip",
            25: "Left Knee",
            26: "Right Knee",
            27: "Left Ankle",
            28: "Right Ankle"
        }

        mp_pose = _get_mp_pose()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6
        )

        self.previous_landmarks = None
        self.stable_time = 0
        self.stability_threshold = 0.5  # 0.5 second stable time
        self.tolerance_range = 5
        self.pause_stability = False
        self.pause_time = 0
        self.detected_pose = False
        self.detected_view = False
        self.pose_detection_time = 0
        self.view_detection_time = 0
        self.pose_detection_timeout = 5 
        self.view_detection_timeout = 5 

        self.feedback_cooldown = 5  
        self.last_correction = None
        self.last_correction_time = 0
        self.last_correction_error = float('inf')
        self.current_correction = None
        self.last_time = time.time()
        self.fps = 30.0
        self.feedback_interval = 5  # seconds
        self.last_feedback_time = time.time()
        self.last_frame_for_feedback = None
        self.ideal_angles_selected = False
        self.fixed_ideal_angles = None 

        
        self.feedback_queue = []
        self.error_tracking = {}
        self.high_fps = True  
        self.view_classified = False 

    def text_to_speech(self, text):
        """Convert text to speech."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        engine.setProperty('volume', 1) 
        engine.say(text)
        engine.runAndWait()

    def print_stable_keypoints(self, landmarks: Dict[int, Tuple[float, float, float, float]]) -> None:
        """Print stable keypoints coordinates when more than 7 keypoints are stable."""
        if not landmarks:
            return

        print("\nStable Keypoints:")
        for idx, (x, y, z, confidence) in landmarks.items():
            if idx in self.keypoints:
                print(f"{self.keypoints[idx]}: X={x}, Y={y}, Z={z:.2f}, Confidence={confidence:.2f}")

    def print_pose_angles(self, angles: Dict[str, float]) -> None:
        """Print all calculated pose angles."""
        if not angles:
            return

        print("\nCurrent Pose Angles:")
        for angle_name, angle_value in angles.items():
            print(f"{angle_name}: {angle_value:.1f} degrees")

    def get_2d_coords(self, idx: int, landmarks: Dict[int, Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
        """Get 2D coordinates (x, y) from landmarks."""
        if idx in landmarks:
            x, y, _, _ = landmarks[idx]
            return (x, y)
        return None

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            ba = a - b
            bc = c - b
            dot_product = np.dot(ba, bc)
            magnitude_ba = np.linalg.norm(ba)
            magnitude_bc = np.linalg.norm(bc)

            if magnitude_ba == 0 or magnitude_bc == 0:
                return 0

            cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
            
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.degrees(np.arccos(cosine_angle))
            return angle
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0
        

    def calculate_pose_angles(self, landmarks: Dict[int, Tuple[float, float, float, float]]) -> Dict[str, float]:
        """Calculate angles between key body parts."""
        if not landmarks:
            return {}

        angles = {}
        left_elbow_coords = [self.get_2d_coords(i, landmarks) for i in [11, 13, 15]]
        if all(left_elbow_coords):
            left_elbow = self.calculate_angle(*left_elbow_coords)
            angles['Left_Elbow_Angle'] = left_elbow

        right_elbow_coords = [self.get_2d_coords(i, landmarks) for i in [12, 14, 16]]
        if all(right_elbow_coords):
            right_elbow = self.calculate_angle(*right_elbow_coords)
            angles['Right_Elbow_Angle'] = right_elbow

        left_shoulder_coords = [self.get_2d_coords(i, landmarks) for i in [13, 11, 23]]
        if all(left_shoulder_coords):
            left_shoulder = self.calculate_angle(*left_shoulder_coords)
            angles['Left_Shoulder_Angle'] = left_shoulder

        right_shoulder_coords = [self.get_2d_coords(i, landmarks) for i in [14, 12, 24]]
        if all(right_shoulder_coords):
            right_shoulder = self.calculate_angle(*right_shoulder_coords)
            angles['Right_Shoulder_Angle'] = right_shoulder

        left_hip_coords = [self.get_2d_coords(i, landmarks) for i in [11, 23, 25]]
        if all(left_hip_coords):
            left_hip = self.calculate_angle(*left_hip_coords)
            angles['Left_Hip_Angle'] = left_hip

        right_hip_coords = [self.get_2d_coords(i, landmarks) for i in [12, 24, 26]]
        if all(right_hip_coords):
            right_hip = self.calculate_angle(*right_hip_coords)
            angles['Right_Hip_Angle'] = right_hip

        left_knee_coords = [self.get_2d_coords(i, landmarks) for i in [23, 25, 27]]
        if all(left_knee_coords):
            left_knee = self.calculate_angle(*left_knee_coords)
            angles['Left_Knee_Angle'] = left_knee

        right_knee_coords = [self.get_2d_coords(i, landmarks) for i in [24, 26, 28]]
        if all(right_knee_coords):
            right_knee = self.calculate_angle(*right_knee_coords)
            angles['Right_Knee_Angle'] = right_knee

        left_ankle_coords = [self.get_2d_coords(i, landmarks) for i in [25, 27, 11]]
        if all(left_ankle_coords):
            left_ankle = self.calculate_angle(*left_ankle_coords)
            angles['Left_Ankle_Angle'] = left_ankle

        right_ankle_coords = [self.get_2d_coords(i, landmarks) for i in [26, 28, 12]]
        if all(right_ankle_coords):
            right_ankle = self.calculate_angle(*right_ankle_coords)
            angles['Right_Ankle_Angle'] = right_ankle

        return angles
    
    
    def classify_view(self, stable_coordinates):
        """
        Enhanced View Classification with more sub-categories based on keypoints' coordinates.
        """
        try:
            
            left_shoulder = stable_coordinates[11]
            right_shoulder = stable_coordinates[12]
            left_hip = stable_coordinates[23]
            right_hip = stable_coordinates[24]
            left_knee = stable_coordinates[25]
            right_knee = stable_coordinates[26]
            left_wrist = stable_coordinates[15]
            right_wrist = stable_coordinates[16]
            left_elbow = stable_coordinates[13]
            right_elbow = stable_coordinates[14]

            # required_keypoints = [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_wrist, right_wrist, left_elbow, right_elbow]
            # if any(not kp for kp in required_keypoints):
            #     return "Partial Pose"

            
            shoulder_depth_diff = abs(left_shoulder[2] - right_shoulder[2])
            hip_depth_diff = abs(left_hip[2] - right_hip[2])
            knee_depth_diff = abs(left_knee[2] - right_knee[2])
            
            wrist_depth_diff = abs(left_wrist[2] - right_wrist[2])
            elbow_depth_diff = abs(left_elbow[2] - right_elbow[2])

            shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
            hip_height_diff = abs(left_hip[1] - right_hip[1])
            knee_height_diff = abs(left_knee[1] - right_knee[1])
            
            shoulder_hip_dist = abs(left_shoulder[0] - left_hip[0])
            knee_hip_dist = abs(left_knee[0] - left_hip[0])
            
            
            if (shoulder_depth_diff < 0.1 and hip_depth_diff < 0.1) and \
               (shoulder_height_diff < 0.1 and hip_height_diff < 0.1):

                if knee_depth_diff < 0.1:
                    return "Front View (Perfect)"

                elif knee_depth_diff < 0.2:
                    return "Front View (Partial)"

                else:
                    return "Front View (Mixed)"
            elif (shoulder_depth_diff > 0.5 and hip_depth_diff > 0.5) and \
                 (shoulder_height_diff < 0.2 and hip_height_diff < 0.2):

                if knee_depth_diff > 0.5:
                    return "Back View (Full)"

                elif knee_depth_diff > 0.3:
                    return "Back View (Partial)"

                else:
                    return "Back View (Mixed)"

            elif (shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3) and \
                 (shoulder_height_diff < 0.1 and hip_height_diff < 0.1):

                # if shoulder_depth_diff > 0.6 and hip_depth_diff > 0.6:
                #     return "Side View (Perfect - Full Profile)"

                if shoulder_depth_diff > 0.4 and hip_depth_diff > 0.4:
                    return "Side View (Perfect - Near Full)"

                elif shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3:
                    return "Side View (Perfect - Partial)"

                else:
                    return "Side View (Intermediate)"

            elif (shoulder_depth_diff > 0.2 and hip_depth_diff > 0.2) and \
                 (shoulder_height_diff > 0.1 and hip_height_diff > 0.1):

                if shoulder_depth_diff > 0.4 and hip_depth_diff > 0.4:
                    return "Oblique View (Strong)"

                elif shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3:
                    return "Oblique View (Moderate)"

                else:
                    return "Oblique View (Mild)"

            elif (wrist_depth_diff > 0.4 or elbow_depth_diff > 0.4):

                if wrist_depth_diff > 0.6 and elbow_depth_diff > 0.6:
                    return "Arm-Specific (Extended Side View)"

                elif wrist_depth_diff > 0.4:
                    return "Arm-Specific (Partial Extension)"

                else:
                    return "Arm-Specific (Mixed)"
            else: 
                return "Rare or Mixed View"
        except Exception as e:
            print(f"Error in classify_view: {e}")
            return "Unknown View"



    async def get_ideal_angles(self, pose_name: str, landmarks: Dict[int, Tuple[float, float, float, float]]) -> Dict[str, Dict[str, float]]:
        """Get ideal angles for a pose from the database and calculate errors."""
        try:
            if not pose_name or not landmarks:
                return {}

            from asgiref.sync import sync_to_async

            @sync_to_async
            def get_angles(pose_name, view, is_flipped):
                try:
                    angles = YogaPoseIdealAngle.objects.get(
                        pose_name=pose_name,
                        view=view,
                        is_flipped=is_flipped
                    )
                    return {
                        'Left_Elbow_Angle': {
                            'mean': angles.left_elbow_angle_mean,
                            'min': angles.left_elbow_angle_mean - 10,
                            'max': angles.left_elbow_angle_mean + 10
                        },
                        'Right_Elbow_Angle': {
                            'mean': angles.right_elbow_angle_mean,
                            'min': angles.right_elbow_angle_mean - 10,
                            'max': angles.right_elbow_angle_mean + 10
                        },
                        'Left_Shoulder_Angle': {
                            'mean': angles.left_shoulder_angle_mean,
                            'min': angles.left_shoulder_angle_mean - 10,
                            'max': angles.left_shoulder_angle_mean + 10
                        },
                        'Right_Shoulder_Angle': {
                            'mean': angles.right_shoulder_angle_mean,
                            'min': angles.right_shoulder_angle_mean - 10,
                            'max': angles.right_shoulder_angle_mean + 10
                        },
                        'Left_Hip_Angle': {
                            'mean': angles.left_hip_angle_mean,
                            'min': angles.left_hip_angle_mean - 10,
                            'max': angles.left_hip_angle_mean + 10
                        },
                        'Right_Hip_Angle': {
                            'mean': angles.right_hip_angle_mean,
                            'min': angles.right_hip_angle_mean - 10,
                            'max': angles.right_hip_angle_mean + 10
                        },
                        'Left_Knee_Angle': {
                            'mean': angles.left_knee_angle_mean,
                            'min': angles.left_knee_angle_mean - 10,
                            'max': angles.left_knee_angle_mean + 10
                        },
                        'Right_Knee_Angle': {
                            'mean': angles.right_knee_angle_mean,
                            'min': angles.right_knee_angle_mean - 10,
                            'max': angles.right_knee_angle_mean + 10
                        },
                        'Left_Ankle_Angle': {
                            'mean': angles.left_ankle_angle_mean,
                            'min': angles.left_ankle_angle_mean - 10,
                            'max': angles.left_ankle_angle_mean + 10
                        },
                        'Right_Ankle_Angle': {
                            'mean': angles.right_ankle_angle_mean,
                            'min': angles.right_ankle_angle_mean - 10,
                            'max': angles.right_ankle_angle_mean + 10
                        }
                    }
                except YogaPoseIdealAngle.DoesNotExist:
                    return None

            # Get current pose angles
            current_angles = self.calculate_pose_angles(landmarks)
            if not current_angles:
                print("No angles calculated for current pose")
                return {}

            original_angles = await get_angles(pose_name, self.current_view, False)
            flipped_angles = await get_angles(pose_name, self.current_view, True)

            if original_angles or flipped_angles:
                original_errors = self.calculate_error(current_angles, original_angles) if original_angles else None
                flipped_errors = self.calculate_error(current_angles, flipped_angles) if flipped_angles else None

                original_total_error = sum(abs(error['error']) for error in original_errors.values()) if original_errors else float('inf')
                flipped_total_error = sum(abs(error['error']) for error in flipped_errors.values()) if flipped_errors else float('inf')

                if original_total_error <= flipped_total_error:
                    print(f"Using original view: Error={original_total_error:.2f}")
                    return original_angles
                else:
                    print(f"Using flipped view: Error={flipped_total_error:.2f}")
                    return flipped_angles

            @sync_to_async
            def get_all_views(pose_name):
                try:
                    return list(YogaPoseIdealAngle.objects.filter(
                        pose_name=pose_name,
                        is_flipped=False
                    ).values_list('view', flat=True).distinct())
                except Exception as e:
                    print(f"Error fetching all views: {e}")
                    return []

            all_views = await get_all_views(pose_name)
            if not all_views:
                print(f"No views found in database for pose: {pose_name}")
                return {}

            #  all views and find the best match
            best_view = None
            best_angles = None
            min_error = float('inf')

            for view in all_views:
                for is_flipped in [False, True]:
                    angles = await get_angles(pose_name, view, is_flipped)
                    if angles:
                        errors = self.calculate_error(current_angles, angles)
                        total_error = sum(abs(error['error']) for error in errors.values())
                        
                        if total_error < min_error:
                            min_error = total_error
                            best_view = f"{view} (Flipped={is_flipped})"
                            best_angles = angles
                            
                        print(f"View: {view} (Flipped={is_flipped}) - Error: {total_error:.2f}")

            if best_angles:
                print(f"Best matching view found: {best_view} with error: {min_error:.2f}")
                return best_angles
            else:
                print("No suitable view found in database")
                return {}

        except Exception as e:
            print(f"Error fetching ideal angles: {e}")
            return {}
    
    async def draw_pose(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose landmarks and angles on the frame."""
        if self.stable_coordinates:
            for idx, landmark in enumerate(self.stable_coordinates):
                if idx in self.keypoints:
                    cv2.circle(
                        frame,
                        (int(landmark[0]), int(landmark[1])),
                        5,
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        self.keypoints.get(idx, ""),
                        (int(landmark[0]) + 10, int(landmark[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255), 
                        1
                    )

        if self.ideal_angles and self.current_pose:
            try:
                angles = self.calculate_pose_angles()
                y_position = 30 
                
                cv2.putText(
                    frame,
                    f"Current Pose: {self.current_pose}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                y_position += 30
                
                for angle_name in self.ideal_angles:
                    if angle_name in angles:
                        current = angles[angle_name]
                        ideal = self.ideal_angles[angle_name]
                        
                        if current < ideal['min']:
                            color = (0, 0, 255)  # Red for too low
                        elif current > ideal['max']:
                            color = (0, 0, 255)  # Red for too high
                        else:
                            color = (0, 255, 0)  # Green for within range

                        # Display angle with color-coded guidance
                        cv2.putText(
                            frame,
                            f"{angle_name}: {current:.1f}° (Target: {ideal['mean']:.1f}°)",
                            (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                        y_position += 25

            except Exception as e:
                print(f"Error displaying angles: {e}")

        return frame

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Draw proper stick figure on the frame with ideal angles and guidance."""

        for p1, p2 in self.connections:
            if landmarks[p1] and landmarks[p2]:
                cv2.line(
                    frame,
                    (int(landmarks[p1][0]), int(landmarks[p1][1])),
                    (int(landmarks[p2][0]), int(landmarks[p2][1])),
                    (0, 255, 0),  
                    3
                )

        for idx, landmark in enumerate(landmarks):
            if landmark:
                cv2.circle(
                    frame,
                    (int(landmark[0]), int(landmark[1])),
                    7,
                    (0, 0, 255), 
                    -1
                )
                cv2.putText(
                    frame,
                    self.keypoints.get(idx, ""),
                    (int(landmark[0]) + 10, int(landmark[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255), 
                    1
                )

        if self.ideal_angles and self.current_pose:
            try:
                angles = self.calculate_pose_angles()
                y_position = 30 
                
                cv2.putText(
                    frame,
                    f"Current Pose: {self.current_pose}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                y_position += 30
                
                # Display each angle's guidance
                for angle_name in self.ideal_angles:
                    if angle_name in angles:
                        current = angles[angle_name]
                        ideal = self.ideal_angles[angle_name]
                        
                        # Determine guidance color
                        if current < ideal['min']:
                            color = (0, 0, 255)  # Red for too low
                            guidance = f"Too low ({ideal['min']:.1f}° - {ideal['max']:.1f}°)"
                        elif current > ideal['max']:
                            color = (0, 0, 255)  # Red for too high
                            guidance = f"Too high ({ideal['min']:.1f}° - {ideal['max']:.1f}°)"
                        else:
                            color = (0, 255, 0)  # Green for within range
                            guidance = "Good position!"
                        
                        # Display angle information
                        cv2.putText(
                            frame,
                            f"{angle_name}: {current:.1f}°",  # Current angle
                            (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                        y_position += 20
                        
                        # Display guidance
                        cv2.putText(
                            frame,
                            guidance,
                            (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1
                        )
                        y_position += 20
            except Exception as e:
                print(f"Error displaying angles: {str(e)}")

        return frame

    def calculate_error(self, actual: Dict[str, float], ideal: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate the angle errors between actual and ideal angles with tolerance range."""
        errors = {}
        
        for angle in actual:
            if angle in ideal:
                detected_value = actual[angle]
                ideal_value = ideal[angle]['mean']
                min_value = ideal[angle]['min']
                max_value = ideal[angle]['max']
                
                # Calculate error
                error = abs(detected_value - ideal_value)
                
                # Check if angle is within range
                within_range = min_value <= detected_value <= max_value
                
                errors[angle] = {
                    'error': error,
                    'within_range': within_range,
                    'actual': detected_value,
                    'target': ideal_value,
                    'min': min_value,
                    'max': max_value
                }
        
        return errors
    
    
    def generate_feedback(self, errors):
        """Generate feedback based on angle errors."""
        feedback = []
        detailed_errors = []

        if isinstance(errors, dict):
            for joint, data in errors.items():
                # Convert joint index to name if it's an integer
                if isinstance(joint, int):
                    joint_name = self.keypoints.get(joint, str(joint))
                else:
                    joint_name = str(joint).lower().replace('_', ' ')

                if isinstance(data, dict):
                    detected = data.get("detected", 0)
                    ideal = data.get("ideal", 0)
                    error = data.get("error", 0)
                else:  # Handle tuple format
                    detected = data[0]
                    ideal = data[1]
                    error = abs(detected - ideal)

                if error > 10:
                    # Get more specific joint name from keypoints if available
                    if isinstance(joint, int):
                        joint_name = self.keypoints.get(joint, str(joint))
                    else:
                        joint_name = str(joint).lower().replace('_', ' ')

                    if "elbow" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Extend your {joint_name} fully. error with {error:.2f}")
                        else:
                            feedback.append(f"Bend your {joint_name} slightly. error with {error:.2f}")
                    elif "hip" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Drop your {joint_name} down a little to balance. error with {error:.2f}")
                        else:
                            feedback.append(f"Lift your {joint_name} up slightly to balance. error with {error:.2f}")
                    elif "knee" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Bend your {joint_name} slightly for balance. error with {error:.2f}")
                        else:
                            feedback.append(f"Straighten your {joint_name} slightly. error with {error:.2f}")
                    elif "ankle" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Shift your weight on {joint_name} slightly forward onto your toes. error with {error:.2f}")
                        else:
                            feedback.append(f"Shift your weight on {joint_name} slightly back onto your heels. error with {error:.2f}")
                    else:
                        # For other joints, provide generic feedback
                        if detected < ideal:
                            feedback.append(f"Adjust your {joint_name} to be more open. error with {error:.2f}")
                        else:
                            feedback.append(f"Adjust your {joint_name} to be more closed. error with {error:.2f}")

                    detailed_errors.append(f"{joint_name}: Detected={detected:.1f}°, Ideal={ideal:.1f}°, Error={error:.1f}°")

        elif isinstance(errors, list):
            for joint, detected, ideal in errors:
                error = abs(detected - ideal)
                if error > 10:
                    # Get joint name from keypoints if it's an index
                    if isinstance(joint, int):
                        joint_name = self.keypoints.get(joint, str(joint))
                    else:
                        joint_name = str(joint).lower().replace('_', ' ')

                    if "elbow" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Extend your {joint_name} fully. error with {error:.2f}")
                        else:
                            feedback.append(f"Bend your {joint_name} slightly. error with {error:.2f}")
                    elif "hip" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Drop your {joint_name} down a little to balance. error with {error:.2f}")
                        else:
                            feedback.append(f"Lift your {joint_name} up slightly to balance. error with {error:.2f}")
                    elif "knee" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Bend your {joint_name} slightly for balance. error with {error:.2f}")
                        else:
                            feedback.append(f"Straighten your {joint_name} slightly. error with {error:.2f}")
                    elif "ankle" in joint_name.lower():
                        if detected < ideal:
                            feedback.append(f"Shift your weight on {joint_name} slightly forward onto your toes. error with {error:.2f}")
                        else:
                            feedback.append(f"Shift your weight on {joint_name} slightly back onto your heels. error with {error:.2f}")
                    else:
                        # For other joints, provide generic feedback
                        if detected < ideal:
                            feedback.append(f"Adjust your {joint_name} to be more open. error with {error:.2f}")
                        else:
                            feedback.append(f"Adjust your {joint_name} to be more closed. error with {error:.2f}")

                    detailed_errors.append(f"{joint_name}: Detected={detected:.1f}°, Ideal={ideal:.1f}°, Error={error:.1f}°")

        if feedback:
            print("\n---------------------------------FEEDBACK:--------------------------")
            print("Detailed Angle Errors:")
            for error in detailed_errors:
                print(error)
            print("\nCorrections:")
            for correction in feedback:
                print(correction)
            print("---------------------------------")

        return feedback

    async def process_correction(self, frame: np.ndarray, pose_name: str) -> Dict[int, Tuple[float, float, float, float]]:
        """Process frame and return pose landmarks."""
        try:
            if frame is None:
                print("\nERROR: Frame is None - Camera connection issue")
                return {}

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"\nERROR: Failed to convert frame to RGB: {str(e)}")
                return {}

            results = self.pose.process(rgb_frame)

            landmarks_dict = {}
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i in self.keypoints:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        z = landmark.z
                        confidence = landmark.visibility
                        landmarks_dict[i] = (x, y, z, confidence)

            if not landmarks_dict:
                print("\nNo keypoints detected")
                return {}  

            if self.high_fps:
                print(f"Current FPS: {self.fps:.2f}")
                print(f"Current Pose: {pose_name}")
            
            # if self.pause_stability:
            #     if time.time() - self.pause_time >= 300:
            #         self.pause_stability = False
            #         print("\nResuming stability checking...")
            #     return landmarks_dict
            
            if self.pause_stability:
                current_time = time.time()

                if current_time - self.pause_time >= 300:
                    self.pause_stability = False
                    print("\nResuming stability checking...")
                else:
                    # print(f"\n[INFO] Time since last feedback: {abs(current_time - self.last_feedback_time:).2f} seconds")
                    print(f"\n[INFO] Time since last feedback: {abs(current_time - self.last_feedback_time):.2f} seconds")

                    if current_time - self.last_feedback_time >= self.feedback_interval:
                        self.last_feedback_time = current_time
                        print(f"\n[INFO] Providing feedback at {time.strftime('%X')}...")

                        if self.last_frame_for_feedback:
                            angles = self.calculate_pose_angles(self.last_frame_for_feedback)
                            errors = self.calculate_angle_errors(angles, self.fixed_ideal_angles)
                            self.process_feedback_queue(errors)
                        else:
                            print("[WARN] No stable frame available for feedback.")

                    return landmarks_dict 

            if self.previous_landmarks:
                stable_points = sum(
                    1 for idx in self.keypoints
                    if idx in landmarks_dict and idx in self.previous_landmarks
                    and abs(landmarks_dict[idx][0] - self.previous_landmarks[idx][0]) <= self.tolerance_range
                    and abs(landmarks_dict[idx][1] - self.previous_landmarks[idx][1]) <= self.tolerance_range
                )

                print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

                if stable_points >= 7:
                    self.stable_time += 1 / self.fps
                    if self.stable_time >= self.stability_threshold:
                        print("\nPose Stable!")
                        print(f"Stable for {self.stable_time:.2f} seconds")
                        self.stable_coordinates = landmarks_dict.copy()

                        print("-------------------------------------------------->Stable coordinates:",self.stable_coordinates)
                        
                        
                        self.pause_stability = True
                        self.pause_time = time.time()
                        
                        
                        if self.high_fps:
                            self.fps = 2
                            self.high_fps = False
                            print("\nSwitching to 2 FPS for continuous feedback")
                            
                        # angles = self.calculate_pose_angles(landmarks_dict)
                        # self.ideal_angles = await self.get_ideal_angles(pose_name, landmarks_dict)
                        # errors = self.calculate_angle_errors(angles, self.ideal_angles)
                        
                        
                        # current_time = time.time()
                        # if current_time - self.last_feedback_time >= 5:
                        #     self.last_feedback_time = current_time
                        #     self.process_feedback_queue(errors)
                        # Store the last stable frame to use every 5s for feedback
                        self.last_frame_for_feedback = landmarks_dict.copy()

                        # if not self.ideal_angles_selected:
                        #     self.fixed_ideal_angles = await self.get_ideal_angles(pose_name, landmarks_dict)
                        #     self.ideal_angles_selected = True
                        
                        # get ideal angles
                        if not self.ideal_angles_selected:
                            ideal_data = await self.get_ideal_angles(pose_name, landmarks_dict)
                            for angle, val in ideal_data.items():
                                if not all(k in val for k in ("mean", "min", "max")):
                                    print(f"[ERROR] Incomplete ideal angle data for {angle}: {val}")
                            self.fixed_ideal_angles = ideal_data
                            self.ideal_angles_selected = True
                            
                        #check time for feedback
                        current_time = time.time()
                        if current_time - self.last_feedback_time >= self.feedback_interval:
                            self.last_feedback_time = current_time

                            #recalculate angles using last saved frame
                            if self.last_frame_for_feedback:
                                angles = self.calculate_pose_angles(self.last_frame_for_feedback)
                                errors = self.calculate_angle_errors(angles, self.fixed_ideal_angles)
                                self.process_feedback_queue(errors)

                        if self.high_fps:
                            print("\nCalculated Angles:")
                            for angle_name, angle_value in angles.items():
                                print(f"{angle_name}: {angle_value:.1f} degrees")
                            
                            if self.ideal_angles:
                                print("\nIdeal Angles:")
                                for angle_name, ideal in self.ideal_angles.items():
                                    print(f"{angle_name}: Target={ideal['target']:.1f}, Min={ideal['min']:.1f}, Max={ideal['max']:.1f}")
                            
                            if errors:
                                print("\nAngle Errors:")
                                for angle_name, error in errors.items():
                                    within_range = "✓" if error['within_range'] else "✗"
                                    print(f"{angle_name}: Error={error['error']:.1f}°, Actual={error['actual']:.1f}°, Target={error['target']:.1f}°, Range=[{error['min']:.1f}°-{error['max']:.1f}°] ({within_range})")
                        
                        self.print_stable_keypoints(landmarks_dict)

                        # Only classify view once
                        if not self.view_classified:
                            self.current_view = self.classify_view(self.stable_coordinates)
                            print(f"View classified as: {self.current_view}")
                            self.view_classified = True

            self.previous_landmarks = landmarks_dict
            return landmarks_dict

        except Exception as e:
            print(f"\nERROR in process_correction: {str(e)}")
            return {}



    def process_feedback_queue(self, errors):
        """Process errors and provide feedback based on highest errors."""
        if not errors:
            return

        current_time = time.time()
        
        new_errors = {}
        for angle_name, error in errors.items():
            if not error['within_range']:
                if angle_name not in self.error_tracking:
                    self.error_tracking[angle_name] = {
                        'error_sum': error['error'],
                        'count': 1,
                        'last_error': error['error'],
                        'detected': error['actual'],
                        'ideal': error['target'],
                        'last_speech_time': 0
                    }
                else:
                    self.error_tracking[angle_name]['error_sum'] += error['error']
                    self.error_tracking[angle_name]['count'] += 1
                    self.error_tracking[angle_name]['last_error'] = error['error']
                    self.error_tracking[angle_name]['detected'] = error['actual']
                    self.error_tracking[angle_name]['ideal'] = error['target']
                new_errors[angle_name] = self.error_tracking[angle_name]

        # Find top 3 joints with the highest average error
        sorted_errors = sorted(
            [(k, v) for k, v in new_errors.items()],
            key=lambda x: x[1]['error_sum'] / x[1]['count'],
            reverse=True
        )

        # Prepare simplified error dictionary for generate_feedback()
        top_errors = {}
        for angle_name, stats in sorted_errors[:1]:
            top_errors[angle_name] = {
                "error": stats['last_error'],
                "detected": stats['detected'],
                "ideal": stats['ideal']
            }

        # Generate feedback only for the top 3 highest errors
        feedback = self.generate_feedback(top_errors)
        
        # Only speak feedback if enough time has passed since last speech
        if not self.last_correction or (current_time - self.last_correction_time > self.feedback_cooldown):
            # Update the last speech time for all tracked errors
            for angle_name in self.error_tracking:
                self.error_tracking[angle_name]['last_speech_time'] = current_time
            
            self.text_to_speech(feedback)
            self.last_correction = feedback
            self.last_correction_time = current_time
        
        # Print and send feedback only if new feedback exists
        if top_errors:
            self.feedback_queue = [
                {
                    'angle': joint,
                    'avg_error': self.error_tracking[joint]['error_sum'] / self.error_tracking[joint]['count'],
                    'last_error': self.error_tracking[joint]['last_error']
                } for joint in top_errors
            ]

            # Raw Feedback Summary
            print("\nPose Correction Feedback:")
            for item in self.feedback_queue:
                print(f"- {item['angle']}: Average Error = {item['avg_error']:.1f}°, Last Error = {item['last_error']:.1f}°")

        # If all joints are within range, clear feedback and reset tracking
        if all(error['within_range'] for error in errors.values()):
            self.feedback_queue.clear()
            self.error_tracking.clear()
            print("\nPose is correct! No further corrections needed.")


    def calculate_angle_errors(self, angles: Dict[str, float], ideal_angles: Dict[str, float]) -> Dict[str, float]:
        """Calculate errors between calculated angles and ideal angles."""
        if not angles or not ideal_angles:
            return {}

        errors = {}
        for angle_name, angle_value in angles.items():
            if angle_name in ideal_angles:
                ideal = ideal_angles[angle_name]
                target = ideal['mean']
                min_angle = ideal['min']
                max_angle = ideal['max']
                
                # Calculate error
                error = abs(angle_value - target)
                
                # Check if angle is within range
                within_range = min_angle <= angle_value <= max_angle
                
                errors[angle_name] = {
                    'error': error,
                    'within_range': within_range,
                    'actual': angle_value,
                    'target': target,
                    'min': min_angle,
                    'max': max_angle
                }
        
        return errors

    def calculate_fps(self):
        """Dynamically calculate FPS based on frame time."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        if frame_time > 0:
            self.fps = 1.0 / frame_time
        else:
            self.fps = 30.0

        return self.fps
    def calculate_angle_errors(self, angles: Dict[str, float], ideal_angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate errors between calculated angles and ideal angles.
        
        Args:
            angles: Dictionary of calculated angles
            ideal_angles: Dictionary of ideal angles with mean, min, max values
            
        Returns:
            Dictionary of angle errors with the following structure:
            {
                "angle_name": {
                    "error": float,        # Difference between actual and target
                    "actual": float,       # Actual angle value
                    "target": float,       # Target angle value
                    "min": float,          # Minimum acceptable angle
                    "max": float,          # Maximum acceptable angle
                    "within_range": bool   # Whether angle is within acceptable range
                }
            }
        """
        errors = {}
        
        if not angles or not ideal_angles:
            return errors
            
        for angle_name, angle_value in angles.items():
            if angle_name in ideal_angles:
                ideal = ideal_angles[angle_name]
                target = ideal['mean']
                error = angle_value - target
                
                errors[angle_name] = {
                    "error": error,
                    "actual": angle_value,
                    "target": target,
                    "min": ideal['min'],
                    "max": ideal['max'],
                    "within_range": ideal['min'] <= angle_value <= ideal['max']
                }
        
        return errors