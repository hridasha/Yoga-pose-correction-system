from typing import List, Tuple, Dict
import numpy as np
import cv2
import time
import tensorflow as tf
import pickle
import os
from django.conf import settings
from django.db import connection
from django.http import JsonResponse
import os
import django
import sys
from asgiref.sync import sync_to_async



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ypc.settings')
django.setup()
# from pose_selection.models import YogaPoseIdealAngle

try:
    from pose_selection.models import YogaPoseIdealAngle
except ImportError as e:
    print(f"Error importing model: {e}")


_mp_pose = None

def _get_mp_pose():
    """Lazy load MediaPipe Pose module."""
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
    return _mp_pose

class PoseDetector:
    def __init__(self, model_path: str, pose_classes_path: str):
        """Initialize the PoseDetector."""
        self.model = tf.keras.models.load_model(model_path)
        self.pose_classes = self.load_pose_classes(pose_classes_path)
        self.stable_coordinates: List[Tuple[float, float, float, float]] = []
        self.needs_printing = True
        self.current_pose = None
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
        self.stability_threshold = 0.5  
        self.tolerance_range = 5  
        self.pause_stability = False
        self.pause_time = 0
        
        self.last_time = time.time()
        self.fps = 30.0

    def load_pose_classes(self, path: str) -> List[str]:
        """Load pose classes from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)



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
    
    # async def classify_pose(self, landmarks):
    #     """Classify the pose using the loaded model."""
    #     if not landmarks:
    #         return "No pose detected"
            
    #     # Prepare the input data for the model
    #     input_data = []
    #     for landmark in landmarks:
    #         if landmark:
    #             input_data.extend(landmark[:3])  # Add x, y, z coordinates
    #         else:
    #             input_data.extend([0, 0, 0])  # Add zeros for missing landmarks
        
    #     # Convert to numpy array and reshape
    #     input_array = np.array([input_data])
        
    #     # Get predictions
    #     predictions = self.model.predict(input_array)
    #     predicted_class_idx = np.argmax(predictions[0])
    #     predicted_class = self.pose_classes[predicted_class_idx]
    #     confidence = predictions[0][predicted_class_idx]
        
    #     return predicted_class, confidence
    async def classify_pose(self, landmarks):
        """Classify the pose using the loaded model."""
        if not landmarks:
            return "No pose detected", 0.0
            
        try:
            model_keypoints = [0,2 ,5 ,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
            
            input_data = []
            for idx in model_keypoints:
                if idx < len(landmarks) and landmarks[idx]:
                    input_data.extend(landmarks[idx][:3])
                else:
                    input_data.extend([0, 0, 0])  # Add zeros for missing landmarks
            
            # Ensure we have exactly 71 features
            if len(input_data) > 71:
                input_data = input_data[:71]
            elif len(input_data) < 71:
                input_data.extend([0] * (71 - len(input_data)))
            
            # Convert to numpy array and reshape
            input_array = np.array([input_data])
            
            # Get predictions
            predictions = self.model.predict(input_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.pose_classes[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return "Unknown", 0.0
    
    async def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Process frame and return pose landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        results = self.pose.process(rgb_frame)

        # Extract landmarks
        landmarks = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = landmark.z
                confidence = landmark.visibility
                landmarks.append((x, y, z, confidence))
        else:
            # If no landmarks detected, return empty
            return []

        # Debug: Print FPS
        print(f"Current FPS: {self.fps:.2f}")

        # Skip stability checking during pause period
        if self.pause_stability:
            if time.time() - self.pause_time >= 20.0:  
                self.pause_stability = False
                print("\nResuming stability checking...")
            return landmarks

        # Stability check
        if self.previous_landmarks is not None:
            stable_points = 0

            for idx in self.keypoints:
                if landmarks[idx] and self.previous_landmarks[idx]:
                    x, y = landmarks[idx][:2]
                    prev_x, prev_y = self.previous_landmarks[idx][:2]

                    if (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
                            prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
                        stable_points += 1

            # Print stability status
            print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

            if stable_points >= 5:  # More than 5 stable keypoints
                self.stable_time += 1 / self.fps

                if self.stable_time >= self.stability_threshold:
                    print("\nPose Stable!")
                    print(f"Stable for {self.stable_time:.2f} seconds")

                    # Store stable coordinates
                    self.stable_coordinates = landmarks
                    self.needs_printing = True

                    # Pause stability checking
                    self.pause_stability = True
                    self.pause_time = time.time()

                    # Print stable coordinates and angles
                    await self.print_stable_coordinates()
                    
                    
                    
                    
                    
                    self.stability_pause = True
                    
                    
                    
                    
                    
                    
                    pose_class, confidence = await self.classify_pose(landmarks)
                    print(f"\nDetected Pose: {pose_class}")
                    print(f"Confidence: {confidence:.2f}")

                    view = self.classify_view(landmarks)
                    print(f"\nView Classification:")
                    print(f"Current View: {view}")

                    try:
                        ideal_angles = await self.get_ideal_angles(pose_class)
                        print("\nIdeal Angles:")
                        if ideal_angles:
                            for angle_name, angle_value in ideal_angles.items():
                                print(f"{angle_name}: {angle_value:.1f} degrees")
                        else:
                            print("No ideal angles found in database")
                            print("checking for other closest views")
                            ideal_angles = await self.get_ideal_angles(pose_class)
                            if ideal_angles :
                                for angle_name, angle_value in ideal_angles.items():
                                    print(f"{angle_name}: {angle_value:.1f} degrees")
                            else:
                                print("No ideal angles found for this pose-----------------2")
                    except YogaPoseIdealAngle.DoesNotExist:
                        print("No ideal angles found in database-------------------2")
                            
                    except Exception as e:
                        print(f"Error fetching ideal angles: {e}")

            else:
                self.stable_time = 0

        self.previous_landmarks = landmarks
        return landmarks



    def classify_view(self, stable_coordinates):
        """
        Enhanced View Classification with more sub-categories based on keypoints' coordinates.
        """
        try:
            if not stable_coordinates:
                return "No Pose Detected"

            # Extract specific keypoints
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

            # Check if all required keypoints are detected
            required_keypoints = [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee]
            if any(not kp for kp in required_keypoints):
                return "Partial Pose"

            
            # Calculate depth differences with more tolerance
            shoulder_depth_diff = abs(left_shoulder[2] - right_shoulder[2])
            hip_depth_diff = abs(left_hip[2] - right_hip[2])
            knee_depth_diff = abs(left_knee[2] - right_knee[2])
            
            wrist_depth_diff = abs(left_wrist[2] - right_wrist[2])
            elbow_depth_diff = abs(left_elbow[2] - right_elbow[2])

            # Calculate height and width differences
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
        
    def calculate_error(self,actual, ideal):
        error={}
        for angle in actual:
            detected_value = actual.get(angle, 0)
            ideal_value = ideal.get(angle, 0)
            error_value = abs(detected_value - ideal_value)

            error[angle] = {
                "detected": round(detected_value, 2),
                "ideal": round(ideal_value, 2),
                "error": round(error_value, 2)
            }
        return error
    
    
        
    async def get_ideal_angles(self, pose_name: str) -> Dict[str, float]:
        """Get ideal angles for a pose from the database and calculate errors."""
        try:
            if not pose_name:
                return {}


            @sync_to_async
            def get_angles(pose_name, view, is_flipped):
                try:
                    angles = YogaPoseIdealAngle.objects.get(
                        pose_name=pose_name,
                        view=view,
                        is_flipped=is_flipped
                    )
                    return {
                        'Left_Elbow_Angle': angles.left_elbow_angle_mean,
                        'Right_Elbow_Angle': angles.right_elbow_angle_mean,
                        'Left_Shoulder_Angle': angles.left_shoulder_angle_mean,
                        'Right_Shoulder_Angle': angles.right_shoulder_angle_mean,
                        'Left_Hip_Angle': angles.left_hip_angle_mean,
                        'Right_Hip_Angle': angles.right_hip_angle_mean,
                        'Left_Knee_Angle': angles.left_knee_angle_mean,
                        'Right_Knee_Angle': angles.right_knee_angle_mean,
                        'Left_Ankle_Angle': angles.left_ankle_angle_mean,
                        'Right_Ankle_Angle': angles.right_ankle_angle_mean
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

            # Get current angles
            current_angles = self.calculate_pose_angles()
            
            if not current_angles:
                print("No angles calculated for current pose")
                return {}

            # First try to get angles for the classified view
            view = self.classify_view(self.stable_coordinates)
            print(f"\nSearching for ideal angles for view: {view}")

            # Try both flipped and non-flipped versions
            views_to_check = [(view, False), (view, True)]
            
            # Also check for other views in the database
            all_views = await get_all_views(pose_name)
            
            for v in all_views:
                if v != view:  # Don't duplicate the current view
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
                        print("------------------------------------------------------------------")
                        print(f"\nView: {view_name} (Flipped={is_flipped})")
                        print(f"Total Error: {total_error:.2f}")
                        print("Angle Errors:")
                        for angle, error in errors.items():
                            print(f"{angle}: Detected={error['detected']}°, Ideal={error['ideal']}°, Error={error['error']}°")
                        print("---------------------------------------------------------")
                except Exception as e:
                    print(f"Error processing view {view_name}: {e}")
                    continue
                
            if best_angles:
                print("====================="*20)
                print(f"\nBest Matching View: {best_view}")
                print(f"Total Error: {min_error:.2f}")
                print("\nBest Angle Errors:")
                for angle, error in best_errors.items():
                    print(f"{angle}: Detected={error['detected']}°, Ideal={error['ideal']}°, Error={error['error']}°")
                print("====================="*20)
                
                return best_angles
            else:
                print(f"No ideal angles found for pose: {pose_name}")
                return {}

        except Exception as e:
            print(f"Error fetching ideal angles: {e}")
            return {}
        


    async def print_stable_coordinates(self):
        """Print stable pose X, Y, Z coordinates, angles, and ideal angles."""
        if self.stable_coordinates:
            print("\nStable Coordinates:")
            for idx, (x, y, z, confidence) in enumerate(self.stable_coordinates):
                if idx in self.keypoints:
                    print(f"{self.keypoints[idx]}: X={x}, Y={y}, Z={z:.2f}, Confidence={confidence:.2f}")

            # Calculate and print angles
            angles = self.calculate_pose_angles()
            print("\nCurrent Angles:")
            for angle_name, angle_value in angles.items():
                print(f"{angle_name}: {angle_value:.1f} degrees")

            # Get and print ideal angles
            if self.current_pose:
                print(f"\nIdeal Angles for this pose:")
                try:
                    if not self.ideal_angles:
                        self.ideal_angles = await self.get_ideal_angles(self.current_pose)
                    
                    for angle_name, ideal_value in self.ideal_angles.items():
                        if angle_name in angles:
                            current_angle = angles[angle_name]
                            print(f"{angle_name}: Current={current_angle:.1f}, Ideal={ideal_value:.1f}")
                except Exception as e:
                    print(f"Error displaying ideal angles: {e}")




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
                            color = (0, 255, 0)  # Green for good
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
                print(f"Error displaying ideal angles: {str(e)}")

        return frame

    def calculate_pose_angles(self) -> Dict[str, float]:
        """Calculate angles from stable coordinates."""
        angle_definitions = {
            "Left_Elbow_Angle": [11, 13, 15],
            "Right_Elbow_Angle": [12, 14, 16],
            "Left_Shoulder_Angle": [23, 11, 13],
            "Right_Shoulder_Angle": [24, 12, 14],
            "Left_Hip_Angle": [11, 23, 25],
            "Right_Hip_Angle": [12, 24, 26],
            "Left_Knee_Angle": [23, 25, 27],
            "Right_Knee_Angle": [24, 26, 28],
            "Left_Ankle_Angle": [25, 27, 28],
            "Right_Ankle_Angle": [26, 28, 27]
        }

        angles = {}

        if self.stable_coordinates:
            for angle_name, points in angle_definitions.items():
                if all(p < len(self.stable_coordinates) for p in points):
                    a = self.stable_coordinates[points[0]][:2]
                    b = self.stable_coordinates[points[1]][:2]
                    c = self.stable_coordinates[points[2]][:2]
                    

                    angle = self.calculate_angle(a, b, c)
                    angles[angle_name] = angle

        return angles

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
