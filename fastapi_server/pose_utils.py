from typing import List, Tuple, Dict, Optional  
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
    print(f"Error importing pose_selection models: {e}")


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
        self.stability_threshold = 0.5  # Reduced stability threshold
        self.tolerance_range = 5
        self.pause_stability = False
        self.pause_time = 0
        self.detected_pose = False
        self.detected_view = False
        self.pose_detection_time = 0
        self.view_detection_time = 0
        self.pose_detection_timeout = 5 
        self.view_detection_timeout = 5 
        self.calculated_best_match = False

        self.last_feedback_time = 0
        self.feedback_cooldown = 5  # 5 seconds cooldown
        self.last_correction = None
        self.last_correction_time = 0
        self.last_correction_error = float('inf')
        self.current_correction = None
        self.last_time = time.time()
        self.feedback_counter = 0
        self.stability_disabled = False  # New flag to disable stability once classification is done

        self.fps = 30.0
        self.feedback_interval = 5  # seconds
        self.last_feedback_time = time.time()
        self.last_frame_for_feedback = None
        self.ideal_angles_selected = False
        self.fixed_ideal_angles = None 

        
        self.feedback_queue = []
        self.error_tracking = {}
        self.high_fps = True  

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
                    input_data.extend([0, 0, 0])  
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
    
    # async def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
    #     """Process frame and return pose landmarks."""
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     results = self.pose.process(rgb_frame)

    #     landmarks = []
    #     if results.pose_landmarks:
    #         for i, landmark in enumerate(results.pose_landmarks.landmark):
    #             x = int(landmark.x * frame.shape[1])
    #             y = int(landmark.y * frame.shape[0])
    #             z = landmark.z
    #             confidence = landmark.visibility
    #             landmarks.append((x, y, z, confidence))
    #     else:
    #         return []

    #     print(f"Current FPS: {self.fps:.2f}")

    #     # Skip stability checking during pause period
    #     if self.pause_stability:
    #         if time.time() - self.pause_time >= 300:
    #             self.pause_stability = False
    #             print("\nResuming stability checking...")
    #         return landmarks

    #     # Stability check
    #     if self.previous_landmarks is not None:
    #         stable_points = 0

    #         for idx in self.keypoints:
    #             if landmarks[idx] and self.previous_landmarks[idx]:
    #                 x, y = landmarks[idx][:2]
    #                 prev_x, prev_y = self.previous_landmarks[idx][:2]

    #                 if (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
    #                         prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
    #                     stable_points += 1

    #         # Print stability status
    #         print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

    #         if stable_points >= 7:  # More than 5 stable keypoints
    #             self.stable_time += 1 / self.fps

    #             if self.stable_time >= self.stability_threshold:
    #                 print("\nPose Stable!")
    #                 print(f"Stable for {self.stable_time:.2f} seconds")

    #                 self.stable_coordinates = landmarks
    #                 self.needs_printing = True

    #                 self.pause_stability = True
    #                 self.pause_time = time.time()

    #                 await self.print_stable_coordinates()

    #                 # Only classify pose and view once
    #                 if not self.detected_pose:
    #                     pose_class, confidence = await self.classify_pose(landmarks)
    #                     print(f"\nDetected Pose: {pose_class}")
    #                     print(f"Confidence: {confidence:.2f}")
    #                     self.current_pose = pose_class
    #                     self.detected_pose = True

    #                 if not self.detected_view:
    #                     self.current_view = self.classify_view(landmarks)
    #                     print(f"\nView Classification:")
    #                     print(f"Current View: {self.current_view}")
    #                     self.detected_view = True

    #                 # Get and process ideal angles
    #                 try:
    #                     if self.current_pose and self.current_view:
    #                         ideal_angles = await self.get_ideal_angles(self.current_pose)
    #                         if ideal_angles:
    #                             current_angles = self.calculate_pose_angles()
    #                             errors = self.calculate_error(current_angles, ideal_angles)
                                
    #                             # Generate feedback if cooldown has passed
    #                             current_time = time.time()
    #                             if (current_time - self.last_feedback_time >= 2 and  # Check every 2 seconds
    #                                 (not self.last_correction or 
    #                                  (current_time - self.last_correction_time >= self.feedback_cooldown and
    #                                   self.last_correction_error > 10))):  # Only repeat if error is significant
                                    
    #                                 feedback, error = self.generate_feedback(errors)
    #                                 if feedback and feedback != self.last_correction:
    #                                     print(f"\nCorrection Feedback:")
    #                                     print(feedback)
    #                                     self.last_correction = feedback
    #                                     self.last_correction_time = current_time
    #                                     self.last_correction_error = error
    #                                     self.current_correction = feedback
                                    
    #                                 # Clear correction if error is fixed
    #                                 elif self.current_correction and error <= 10:
    #                                     print("\nGreat job! Correction complete.")
    #                                     self.current_correction = None
    #                                     self.last_correction = None
    #                                     self.last_correction_error = float('inf')
                                
    #                 except Exception as e:
    #                     print(f"Error processing angles: {e}")

    #         else:
    #             self.stable_time = 0

    #     self.previous_landmarks = landmarks
    #     return landmarks
    
    


    def get_2d_coords(self, idx):
        """Get 2D coordinates (x, y) from stable coordinates."""
        if idx < len(self.stable_coordinates):
            try:
                coords = self.stable_coordinates[idx]
                if len(coords) >= 2:
                    return (coords[0], coords[1])
            except (IndexError, TypeError):
                pass
        return (0, 0)

    def calculate_pose_angles(self) -> Dict[str, float]:
        """Calculate angles between key body parts."""
        if not self.stable_coordinates:
            return {}

        angles = {}
        
        # Calculate elbow angles
        try:
            left_elbow = self.calculate_angle(
                self.get_2d_coords(11),  # Left shoulder
                self.get_2d_coords(13),  # Left elbow
                self.get_2d_coords(15)   # Left wrist
            )
            angles['Left_Elbow_Angle'] = left_elbow

            right_elbow = self.calculate_angle(
                self.get_2d_coords(12),  # Right shoulder
                self.get_2d_coords(14),  # Right elbow
                self.get_2d_coords(16)   # Right wrist
            )
            angles['Right_Elbow_Angle'] = right_elbow
        except:
            pass

        # Calculate shoulder angles
        try:
            left_shoulder = self.calculate_angle(
                self.get_2d_coords(13),  # Left elbow
                self.get_2d_coords(11),  # Left shoulder
                self.get_2d_coords(23)   # Left hip
            )
            angles['Left_Shoulder_Angle'] = left_shoulder

            right_shoulder = self.calculate_angle(
                self.get_2d_coords(14),  # Right elbow
                self.get_2d_coords(12),  # Right shoulder
                self.get_2d_coords(24)   # Right hip
            )
            angles['Right_Shoulder_Angle'] = right_shoulder
        except:
            pass

        # Calculate hip angles
        try:
            left_hip = self.calculate_angle(
                self.get_2d_coords(11),  # Left shoulder
                self.get_2d_coords(23),  # Left hip
                self.get_2d_coords(25)   # Left knee
            )
            angles['Left_Hip_Angle'] = left_hip

            right_hip = self.calculate_angle(
                self.get_2d_coords(12),  # Right shoulder
                self.get_2d_coords(24),  # Right hip
                self.get_2d_coords(26)   # Right knee
            )
            angles['Right_Hip_Angle'] = right_hip
        except:
            pass

        # Calculate knee angles
        try:
            left_knee = self.calculate_angle(
                self.get_2d_coords(23),  # Left hip
                self.get_2d_coords(25),  # Left knee
                self.get_2d_coords(27)   # Left ankle
            )
            angles['Left_Knee_Angle'] = left_knee

            right_knee = self.calculate_angle(
                self.get_2d_coords(24),  # Right hip
                self.get_2d_coords(26),  # Right knee
                self.get_2d_coords(28)   # Right ankle
            )
            angles['Right_Knee_Angle'] = right_knee
        except:
            pass

        # Calculate ankle angles
        try:
            left_ankle = self.calculate_angle(
                self.get_2d_coords(25),  # Left knee
                self.get_2d_coords(27),  # Left ankle
                self.get_2d_coords(11)   # Left shoulder
            )
            angles['Left_Ankle_Angle'] = left_ankle

            right_ankle = self.calculate_angle(
                self.get_2d_coords(26),  # Right knee
                self.get_2d_coords(28),  # Right ankle
                self.get_2d_coords(12)   # Right shoulder
            )
            angles['Right_Ankle_Angle'] = right_ankle
        except:
            pass

        return angles

    async def print_stable_coordinates(self):
        """Print stable pose X, Y, Z coordinates, angles, and ideal angles."""
        if self.stable_coordinates and self.needs_printing:
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
                            print(f"{angle_name}: Current={current_angle:.1f}, Ideal={ideal_value['mean']:.1f}")
                except Exception as e:
                    print(f"Error displaying ideal angles: {e}")

            # Reset needs_printing flag
            self.needs_printing = False

    async def get_ideal_angles(self, pose_name: str) -> Dict[str, Dict[str, float]]:
        """Get ideal angles for a pose from the database and calculate errors."""
        try:
            if not pose_name:
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
                        total_error = sum(error['error'] for error in errors)
                        
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
                        for error in errors:
                            angle_name = error['angle_name']
                            print(f"{angle_name}: Detected={error['detected']:.1f} degrees, Ideal={error['ideal']:.1f} degrees, Error={error['error']:.1f} degrees")

                        print("="*50)
                except Exception as e:
                    print(f"Error processing view {view_name}: {e}")
                    continue

            if best_angles:
                print("="*50)
                print(f"\nBest Matching View: {best_view}")
                print(f"Total Error: {min_error:.2f}")
                print("\nBest Angle Errors:")
                for error in best_errors:
                    angle_name = error['angle_name']
                    print(f"{angle_name}: Detected={error['detected']:.1f} degrees, Ideal={error['ideal']:.1f} degrees, Error={error['error']:.1f} degrees")
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

    # def calculate_error(self, actual: Dict[str, float], ideal: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    #     """Calculate the angle errors between actual and ideal angles."""
    #     errors = {}
        
    #     for angle in actual:
    #         if angle in ideal:
    #             detected_value = actual[angle]
    #             ideal_value = ideal[angle]['mean']
    #             error_value = abs(detected_value - ideal_value)

    #             errors[angle] = {
    #                 "detected": round(detected_value, 2),
    #                 "ideal": round(ideal_value, 2),
    #                 "error": round(error_value, 2)
    #             }
        
    #     return errors
    
    def calculate_error(self, actual: Dict[str, float], ideal: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
        """Calculate angle errors between actual and ideal angles."""
        errors = []

        for angle in actual:
            if angle in ideal:
                detected_value = actual[angle]
                ideal_value = ideal[angle]['mean']
                error_value = abs(detected_value - ideal_value)

                errors.append({
                    'angle_name': angle,
                    'detected': round(detected_value, 2),
                    'ideal': round(ideal_value, 2),
                    'error': round(error_value, 2),
                    'within_range': ideal[angle]['min'] <= detected_value <= ideal[angle]['max']
                })

        return errors


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
                
                # Display each angle's guidance
                for angle_name in self.ideal_angles:
                    if angle_name in angles:
                        current = angles[angle_name]
                        ideal = self.ideal_angles[angle_name]
                        
                        # Determine guidance color
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

    def generate_feedback(self, errors):
        """Generate feedback based on angle errors."""
        feedback = []
        detailed_errors = []

        # for joint, data in errors.items():
        #     detected = data["detected"]
        #     ideal = data["ideal"]
        #     error = data["error"]

        for e in errors:
            joint = e["angle_name"]
            detected = e["detected"]
            ideal = e["ideal"]
            error = e["error"]

            
            
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

            # required_keypoints = [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_wrist, right_wrist, left_elbow, right_elbow]
            # if any(not kp for kp in required_keypoints):
            #     return "Partial Pose"

            
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
        

    def process_feedback_queue(self, errors):
        """Process errors and provide feedback based on highest errors."""
        if not errors:
            return []
            
        # Handle case where errors is a dictionary
        if isinstance(errors, dict):
            # Convert dictionary to list of errors
            errors = [{
                'angle_name': angle,
                'detected': data.get('detected', 0),
                'ideal': data.get('ideal', 0),
                'error': data.get('error', 0)
            } for angle, data in errors.items()]
            
        # If we received feedback strings directly, just return them
        if isinstance(errors, list) and errors and isinstance(errors[0], str):
            feedback = errors
        else:
            # Otherwise process the errors to generate feedback
            feedback = self.generate_feedback(errors)
            
        # Print feedback
        print("\nAdjustment Feedback:")
        for f in feedback:
            print(f)
            
        return feedback
    
    # async def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
    #     """Process a video frame, detect landmarks, and provide feedback if applicable."""

    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.pose.process(rgb_frame)
    #     landmarks = []

    #     if results.pose_landmarks:
    #         for landmark in results.pose_landmarks.landmark:
    #             x = int(landmark.x * frame.shape[1])
    #             y = int(landmark.y * frame.shape[0])
    #             z = landmark.z
    #             confidence = landmark.visibility
    #             landmarks.append((x, y, z, confidence))
    #     else:
    #         return []

    #     print(f"Current FPS: {self.fps:.2f}")
    #     print(f"Pose Detected: {self.detected_pose}, View Detected: {self.detected_view}, Match Calculated: {self.calculated_best_match}")

    #     # === Step 1: After pose and view are classified ===
    #     if self.detected_pose and self.detected_view and self.ideal_angles_selected and not self.calculated_best_match:
    #         # Select best match now that both pose and view are set
    #         current_angles = self.calculate_pose_angles()
    #         self.fixed_ideal_angles = await self.get_ideal_angles(self.current_pose)
    #         self.calculated_best_match = True
    #         print("\n[INFO] Best angle match calculated.")

    #     if self.calculated_best_match:
    #         if not self.stability_disabled:
    #             print("\n[INFO] Stability check disabled. Entering feedback loop.")
    #             self.fps = 2
    #             self.high_fps = False
    #             self.stability_disabled = True
    #             self.feedback_counter = 0

    #         self.feedback_counter += 1

    #         if self.feedback_counter < 10:
    #             print(f"[INFO] Waiting for user adjustment... Frame {self.feedback_counter}/10")
    #             return landmarks

    #         self.feedback_counter = 0
    #         print("\n[INFO] Generating feedback on frame 10...")

    #         if self.last_frame_for_feedback:
    #             angles = self.calculate_pose_angles()
    #             errors = self.calculate_angle_errors(angles, self.fixed_ideal_angles)
    #             self.process_feedback_queue(errors)

    #             print("\n[DEBUG] Feedback:")
    #             for angle_name, error in errors.items():
    #                 within_range = "✓" if error['within_range'] else "✗"
    #                 print(
    #                     f"{angle_name}: Error={error['error']:.1f}°, "
    #                     f"Actual={error['actual']:.1f}°, Target={error['target']:.1f}°, "
    #                     f"Range=[{error['min']:.1f}°-{error['max']:.1f}°] ({within_range})"
    #                 )

    #         return landmarks

    #     # === Step 2: Pre-classification stability check ===
    #     if self.previous_landmarks:
    #         stable_points = 0
    #         for idx in self.keypoints:
    #             if landmarks[idx] and self.previous_landmarks[idx]:
    #                 x, y = landmarks[idx][:2]
    #                 prev_x, prev_y = self.previous_landmarks[idx][:2]
    #                 if (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
    #                     prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
    #                     stable_points += 1

    #         print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

    #         if stable_points >= 7:
    #             self.stable_time += 1 / self.fps

    #             if self.stable_time >= self.stability_threshold:
    #                 print("\n[INFO] Pose Stable!")
    #                 print(f"Stable for {self.stable_time:.2f} seconds")

    #                 self.stable_coordinates = landmarks
    #                 self.last_frame_for_feedback = landmarks
    #                 self.stable_time = 0.0
    #                 await self.print_stable_coordinates()

    #                 if self.high_fps:
    #                     self.fps = 2
    #                     self.high_fps = False
    #                     print("\n[INFO] Switching to 2 FPS for feedback.")

    #                 if not self.detected_pose:
    #                     pose_class, confidence = await self.classify_pose(landmarks)
    #                     print(f"\n[INFO] Pose Classified: {pose_class} (Confidence: {confidence:.2f})")
    #                     self.current_pose = pose_class
    #                     self.detected_pose = True

    #                 if not self.detected_view:
    #                     self.current_view = self.classify_view(landmarks)
    #                     print(f"[INFO] View Classified: {self.current_view}")
    #                     self.detected_view = True

    #                 if not self.ideal_angles_selected:
    #                     self.fixed_ideal_angles = await self.get_ideal_angles(self.current_pose)
    #                     self.ideal_angles_selected = True
    #                     print("\n[INFO] Ideal angles loaded.")

    #     self.previous_landmarks = landmarks
    #     return landmarks

    async def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Process a video frame, detect landmarks, and provide feedback if applicable."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        landmarks = []

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = landmark.z
                confidence = landmark.visibility
                landmarks.append((x, y, z, confidence))
        else:
            return []

        print(f"Current FPS: {self.fps:.2f}")
        print(f"Pose Detected: {self.detected_pose}, View Detected: {self.detected_view}, Match Calculated: {self.calculated_best_match}")

        if self.detected_pose and self.detected_view and self.ideal_angles_selected and not self.calculated_best_match:
            current_angles = self.calculate_pose_angles()
            self.fixed_ideal_angles = await self.get_ideal_angles(self.current_pose)
            self.calculated_best_match = True
            print("\n[INFO] Best angle match calculated.")

        if self.calculated_best_match:
            if not self.stability_disabled:
                print("\n[INFO] Stability check disabled. Entering feedback loop.")
                self.fps = 2
                self.high_fps = False
                self.stability_disabled = True
                self.feedback_timer_start = time.time()

            elapsed = time.time() - self.feedback_timer_start
            print(f"[INFO] Waiting for user adjustment... {elapsed:.2f}/5.00 seconds")

            if elapsed < 5:
                return landmarks 

            print("\n[INFO] Generating feedback after 5 seconds...")
            self.feedback_timer_start = time.time()  # Reset for next feedback round

            if self.last_frame_for_feedback:
                angles = self.calculate_pose_angles()
                errors = self.calculate_angle_errors(angles, self.fixed_ideal_angles)
                self.process_feedback_queue(errors)

                print("\n[DEBUG] Feedback:")
                for angle_name, error in errors.items():
                    within_range = "✓" if error['within_range'] else "✗"
                    print(
                        f"{angle_name}: Error={error['error']:.1f}°, "
                        f"Actual={error['actual']:.1f}°, Target={error['target']:.1f}°, "
                        f"Range=[{error['min']:.1f}°-{error['max']:.1f}°] ({within_range})"
                    )

            return landmarks

        # === Step 2: Pre-classification stability check ===
        if self.previous_landmarks:
            stable_points = 0
            for idx in self.keypoints:
                if landmarks[idx] and self.previous_landmarks[idx]:
                    x, y = landmarks[idx][:2]
                    prev_x, prev_y = self.previous_landmarks[idx][:2]
                    if (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
                        prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
                        stable_points += 1

            print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

            if stable_points >= 7:
                self.stable_time += 1 / self.fps

                if self.stable_time >= self.stability_threshold:
                    print("\n[INFO] Pose Stable!")
                    print(f"Stable for {self.stable_time:.2f} seconds")

                    
                    self.last_frame_for_feedback = landmarks
                    self.stable_coordinates = landmarks
                    self.stable_time = 0.0
                    await self.print_stable_coordinates()

                    if self.high_fps:
                        self.fps = 2
                        self.high_fps = False
                        print("\n[INFO] Switching to 2 FPS for feedback.")

                    if not self.detected_pose:
                        pose_class, confidence = await self.classify_pose(landmarks)
                        print(f"\n[INFO] Pose Classified: {pose_class} (Confidence: {confidence:.2f})")
                        self.current_pose = pose_class
                        self.detected_pose = True

                    if not self.detected_view:
                        self.current_view = self.classify_view(landmarks)
                        print(f"[INFO] View Classified: {self.current_view}")
                        self.detected_view = True

                    if not self.ideal_angles_selected:
                        self.fixed_ideal_angles = await self.get_ideal_angles(self.current_pose)
                        self.ideal_angles_selected = True
                        print("\n[INFO] Ideal angles loaded.")

        self.previous_landmarks = landmarks
        return landmarks
