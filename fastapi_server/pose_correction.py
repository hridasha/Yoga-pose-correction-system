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

        self.last_feedback_time = 0
        self.feedback_cooldown = 5  
        self.last_correction = None
        self.last_correction_time = 0
        self.last_correction_error = float('inf')
        self.current_correction = None
        self.last_time = time.time()
        self.fps = 30.0
        
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
            if not stable_coordinates:
                return "No Pose Detected"

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

            @sync_to_async
            def get_all_views(pose_name):
                try:
                    return list(YogaPoseIdealAngle.objects.filter(
                        pose_name=pose_name,
                        is_flipped=False
                    ).values_list('view', flat=True))
                except Exception as e:
                    print(f"Error fetching all views: {e}")
                    return []
            
            #current psoe ang;es
            current_angles = self.calculate_pose_angles(landmarks)
            
            if not current_angles:
                print("No angles calculated for current pose")
                return {}

            # First try to get angles for the classified view
            view = self.classify_view(self.stable_coordinates)
            print(f"\nSearching for ideal angles for view: {view}")
            
            # Try both flipped and non-flipped versions
            views_to_check = [(view, False), (view, True)]
            
            all_views = await get_all_views(pose_name)
            
            for v in all_views:
                if v != view:  
                    views_to_check.extend([(v, False), (v, True)])

            min_error = float('inf')
            best_view = None
            best_angles = None
            best_errors = None

            for view_name, is_flipped in views_to_check:
                try:
                    # Get angles for this view
                    ideal_angles = await get_angles(pose_name, view_name, is_flipped)
                    
                    if ideal_angles:
                        # Calculate errors
                        errors = self.calculate_error(current_angles, ideal_angles)
                        
                        # Calculate total error
                        total_error = sum(error['error'] for error in errors.values())
                        
                        # Check if this is the best match
                        if total_error < min_error:
                            min_error = total_error
                            best_view = f"{view_name} (Flipped={is_flipped})"
                            best_angles = ideal_angles
                            best_errors = errors
                        
                        print("="*50)
                        print(f"\nView: {view_name} (Flipped={is_flipped})")
                        print(f"Total Error: {total_error:.2f}")
                        print("Angle Errors:")
                        for angle, error in errors.items():
                            print(f"{angle}: Detected={error['detected']:.2f}°, Ideal={error['ideal']:.2f}°, Error={error['error']:.2f}°")
                        print("="*50)
                except Exception as e:
                    print(f"Error processing view {view_name}: {e}")
                    continue

            if best_angles:
                print("="*50)
                print(f"\nBest Matching View: {best_view}")
                print(f"Total Error: {min_error:.2f}")
                print("\nBest Angle Errors:")
                for angle, error in best_errors.items():
                    print(f"{angle}: Detected={error['detected']:.2f}°, Ideal={error['ideal']:.2f}°, Error={error['error']:.2f}°")
                print("="*50)
                feedback = self.generate_feedback(best_errors)
                print("\nAdjustment Feedback:")
                print(feedback)
                
                return best_angles
            else:
                print(f"No ideal angles found for pose: {pose_name}")
                return {}

        except Exception as e:
            print(f"Error fetching ideal angles: {e}")
            return {}

    def calculate_error(self, actual: Dict[str, float], ideal: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate the angle errors between actual and ideal angles."""
        errors = {}
        
        for angle in actual:
            if angle in ideal:
                detected_value = actual[angle]
                ideal_value = ideal[angle]['mean']
                error_value = abs(detected_value - ideal_value)

                errors[angle] = {
                    "detected": round(detected_value, 2),
                    "ideal": round(ideal_value, 2),
                    "error": round(error_value, 2)
                }
        
        return errors
    
    
    def generate_feedback(self, errors):
        """Generate feedback based on angle errors."""
        feedback = []
        detailed_errors = []

        for joint, data in errors.items():
            detected = data["detected"]
            ideal = data["ideal"]
            error = data["error"]

            
            if error > 10:
                joint_name = joint.lower().replace('_', ' ')
        
                if "elbow" in joint.lower():
                    if detected < ideal:
                        feedback.append(f"Extend your {joint_name} fully. error with {error}")
                    else:
                        feedback.append(f"Relax your {joint_name} slightly. error with {error}")

                elif "shoulder" in joint.lower():
                    if detected < ideal:
                        feedback.append(f"Lift your {joint_name} to align with your arm.error with {error}")
                    else:
                        feedback.append(f"Drop your {joint_name} slightly to relax.error with {error}")

                elif "hip" in joint.lower():
                    if detected < ideal:
                        feedback.append(f"Push your {joint_name} upward slightly.error with {error}")
                    else:
                        feedback.append(f"Drop your {joint_name} down a little to balance.error with {error}")

                elif "knee" in joint.lower():
                    if detected < ideal:
                        feedback.append(f"Try straightening your {joint_name}.error with    {error}")
                    else:
                        feedback.append(f"Bend your {joint_name} slightly for balance.errro with {error}")

                elif "ankle" in joint.lower():
                    if detected < ideal:
                        feedback.append(f"Shift your weight on {joint_name} slightly forward onto your toes.error with {error}")
                    else:
                        feedback.append(f"Shift your weight  {joint_name} slightly back onto your heel.error with {error}")
        
        print("\n---------------------------------FEEDBACK:--------------------------")
        print("Detailed Angle Errors:")
        for error in detailed_errors:
            print(error)
        print("\nCorrections:")
        for correction in feedback:
            print(correction)
        print("---------------------------------")

        if not feedback:
            return []

        return feedback


    async def process_correction(self, frame: np.ndarray, pose_name: str) -> Dict[int, Tuple[float, float, float, float]]:
        """Process frame and return pose landmarks."""
        try:
            if frame is None:
                print("\nERROR: Frame is None - Camera connection issue")
                return {}

            # Convert frame to RGB
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

            print(f"Current FPS: {self.fps:.2f}")
            print(f"Current Pose: {pose_name}")
        
            if self.pause_stability:
                if time.time() - self.pause_time >= 300:
                    self.pause_stability = False
                    print("\nResuming stability checking...")
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
                        
                        self.pause_stability = True
                        self.pause_time = time.time()
                        # Calculate angles
                        angles = self.calculate_pose_angles(landmarks_dict)
                        
                        # Get ideal angles for this pose
                        self.ideal_angles = await self.get_ideal_angles(pose_name, landmarks_dict)
                        
                        # Calculate errors
                        errors = self.calculate_angle_errors(angles, self.ideal_angles)
                        
                        # Print results
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


            self.previous_landmarks = landmarks_dict
            return landmarks_dict

        except Exception as e:
            print(f"\nERROR in process_correction: {str(e)}")
            return {}

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

    def calculate_angle_errors(self, angles: Dict[str, float], ideal_angles: Dict[str, float]) -> Dict[str, float]:
        """Calculate errors between calculated angles and ideal angles."""
        if not angles or not ideal_angles:
            return {}

        errors = {}
        for angle_name, angle_value in angles.items():
            if angle_name in ideal_angles:
                ideal = ideal_angles[angle_name]
                target = ideal['target']
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
