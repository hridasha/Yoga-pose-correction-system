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

# Lazy import MediaPipe
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

        # Load MediaPipe pose model
        mp_pose = _get_mp_pose()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6
        )

        # Stability settings
        self.previous_landmarks = None
        self.stable_time = 0
        self.stability_threshold = 0.5  
        self.tolerance_range = 5  
        self.pause_stability = False
        self.pause_time = 0

        # FPS tracking
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

    def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Process frame and return pose landmarks."""
        # Convert BGR to RGB
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
            if time.time() - self.pause_time >= 2.0:  # Pause for 1 second
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

                    # Check stability
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
                    self.print_stable_coordinates()

            else:
                self.stable_time = 0

        self.previous_landmarks = landmarks
        return landmarks



    def classify_view(stable_coordinates,database_poses=None):
        """
        Enhanced View Classification with more sub-categories based on keypoints' coordinates.
        """
        try:
            # Extract coordinates from stable pose data (from print_stable_coordinates output)
            shoulder_angle = float(stable_coordinates[11][0])  # Left Shoulder X
            hip_angle = float(stable_coordinates[23][0])  # Left Hip X
            knee_angle = float(stable_coordinates[25][0])  # Left Knee X

            # Depth differences
            shoulder_depth_diff = abs(float(stable_coordinates[11][2]) - float(stable_coordinates[12][2]))
            hip_depth_diff = abs(float(stable_coordinates[23][2]) - float(stable_coordinates[24][2]))
            knee_depth_diff = abs(float(stable_coordinates[25][2]) - float(stable_coordinates[26][2]))

            wrist_depth_diff = abs(float(stable_coordinates[15][2]) - float(stable_coordinates[16][2]))
            elbow_depth_diff = abs(float(stable_coordinates[13][2]) - float(stable_coordinates[14][2]))
            
            # Height differences
            shoulder_height_diff = abs(float(stable_coordinates[11][1]) - float(stable_coordinates[12][1]))
            hip_height_diff = abs(float(stable_coordinates[23][1]) - float(stable_coordinates[24][1]))
            knee_height_diff = abs(float(stable_coordinates[25][1]) - float(stable_coordinates[26][1]))

            # Distance measures
            shoulder_hip_dist = abs(float(stable_coordinates[11][0]) - float(stable_coordinates[23][0]))
            knee_hip_dist = abs(float(stable_coordinates[25][0]) - float(stable_coordinates[23][0]))

            # Pose classification logic based on keypoints
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
                if database_poses:
                    closest_match = find_closest_rare_mixed_view(stable_coordinates, database_poses)
                    return f"Closest Match Rare/Mixed View: {closest_match}"

        except Exception as e:
            print(f"Error in classify_view: {e}")
            
    def find_closest_rare_mixed_view(stable_coordinates, database_poses):
        """
        Find the closest match for a rare or mixed view from the database.
        This function compares the current pose with stored database poses.
        """
        closest_match = None
        min_error = float('inf')

        for db_pose in database_poses:
            error = calculate_pose_error(stable_coordinates, db_pose)
            if error < min_error:
                min_error = error
                closest_match = db_pose["VIEW"] 

        return closest_match

    def calculate_pose_error(stable_coordinates, db_pose):
        """
        Calculate the error between the current pose (stable_coordinates) and the stored database pose.
        For simplicity, let's use the sum of absolute angle differences as a proxy for pose similarity.
        """
        error = 0
        # Assume db_pose contains the ideal joint angles for comparison (e.g., db_pose["left_shoulder_angle"])
        for joint in stable_coordinates:
            # Compare angles (e.g., wrist, elbow, shoulder) with the ideal pose in the database
            error += abs(stable_coordinates[joint] - db_pose[joint])  # Simple error calculation

        return error

    def compare_with_db(pose_name, stable_coordinates, image_url):
        """
        Compare the detected pose with ideal angles from the DB and calculate errors.
        It classifies the pose, retrieves the ideal angles, and calculates errors for original and flipped poses.
        """
        # Step 1: Classify the view (Front, Side, etc.)
        classified_view = classify_view(stable_coordinates)

        # Step 2: Fetch ideal angles for the pose and classified view
        ideal_angles = YogaPoseIdealAngle.objects.filter(
            pose_name=pose_name, view=classified_view
        ).first()

        if not ideal_angles:
            print(f"Combination not available in DB: {pose_name} - {classified_view}")
            return JsonResponse({
                "error": f"Combination not available in DB: {pose_name} - {classified_view}"
            })

        # Step 3: Prepare the original and flipped angle mappings
        original_angles = {
            "Left_Elbow_Angle": ideal_angles.left_elbow_angle_mean,
            "Right_Elbow_Angle": ideal_angles.right_elbow_angle_mean,
            "Left_Shoulder_Angle": ideal_angles.left_shoulder_angle_mean,
            "Right_Shoulder_Angle": ideal_angles.right_shoulder_angle_mean,
            "Left_Knee_Angle": ideal_angles.left_knee_angle_mean,
            "Right_Knee_Angle": ideal_angles.right_knee_angle_mean,
            "Left_Hip_Angle": ideal_angles.left_hip_angle_mean,
            "Right_Hip_Angle": ideal_angles.right_hip_angle_mean,
            "Left_Ankle_Angle": ideal_angles.left_ankle_angle_mean,
            "Right_Ankle_Angle": ideal_angles.right_ankle_angle_mean
        }

        flipped_angles = {
            "Left_Elbow_Angle": ideal_angles.right_elbow_angle_mean,
            "Right_Elbow_Angle": ideal_angles.left_elbow_angle_mean,
            "Left_Shoulder_Angle": ideal_angles.right_shoulder_angle_mean,
            "Right_Shoulder_Angle": ideal_angles.left_shoulder_angle_mean,
            "Left_Knee_Angle": ideal_angles.right_knee_angle_mean,
            "Right_Knee_Angle": ideal_angles.left_knee_angle_mean,
            "Left_Hip_Angle": ideal_angles.right_hip_angle_mean,
            "Right_Hip_Angle": ideal_angles.left_hip_angle_mean,
            "Left_Ankle_Angle": ideal_angles.right_ankle_angle_mean,
            "Right_Ankle_Angle": ideal_angles.left_ankle_angle_mean
        }

        # Step 4: Calculate errors for both original and flipped poses
        actual_angles = {
            "Left_Elbow_Angle": stable_coordinates[13],  # Example of extracting angles from pose data
            "Right_Elbow_Angle": stable_coordinates[14],
            "Left_Shoulder_Angle": stable_coordinates[11],
            "Right_Shoulder_Angle": stable_coordinates[12],
            "Left_Knee_Angle": stable_coordinates[25],
            "Right_Knee_Angle": stable_coordinates[26],
            "Left_Hip_Angle": stable_coordinates[23],
            "Right_Hip_Angle": stable_coordinates[24],
            "Left_Ankle_Angle": stable_coordinates[27],
            "Right_Ankle_Angle": stable_coordinates[28]
        }

        original_errors = calculate_error(actual_angles, original_angles)
        flipped_errors = calculate_error(actual_angles, flipped_angles)

        # Step 5: Calculate the average error for original and flipped poses
        avg_error_original = np.mean([e['error'] for e in original_errors.values()])
        avg_error_flipped = np.mean([e['error'] for e in flipped_errors.values()])

        # Step 6: Choose the best match (either original or flipped) based on the minimum average error
        best_match = "Flipped Pose" if avg_error_flipped < avg_error_original else "Original Pose"
        best_errors = flipped_errors if avg_error_flipped < avg_error_original else original_errors
        avg_error = round(min(avg_error_original, avg_error_flipped), 2)

        # Step 7: Generate corrections based on error thresholds
        corrections = []
        for joint, error in best_errors.items():
            if error['error'] > 5:  # Threshold for corrections (e.g., 5°)
                direction = "Lift" if error['error'] > 0 else "Lower"
                correction = f"{direction} your {joint.replace('_', ' ').lower()} by {round(abs(error['error']), 1)}°"
                corrections.append(correction)

        if not corrections:
            corrections.append("Pose is nearly perfect!")

        # Step 8: Construct feedback
        feedback = {
            "pose_name": pose_name,
            "classified_view": classified_view,
            "best_match": best_match,
            "avg_error": avg_error,
            "corrections": corrections,
            "errors": best_errors,
            "image_url": image_url
        }

        return feedback

            
            
            
    def get_ideal_angles(self, pose_name: str) -> Dict[str, Dict[str, float]]:
        """Get ideal angles for a pose from predefined dictionary."""
        ideal_angles = {
            
        }
        
        return ideal_angles.get(pose_name, {})
    def print_stable_coordinates(self):
        """Print stable pose X, Y, Z coordinates, angles, and ideal angles."""
        if self.needs_printing and self.stable_coordinates:
            print("\n Stable Pose Coordinates and Angles:")
            print("=" * 60)

            # Print the coordinates
            for idx, (x, y, z, confidence) in enumerate(self.stable_coordinates):
                if idx in self.keypoints:
                    print(f"{self.keypoints[idx]}:")
                    print(f"  X: {x}")
                    print(f"  Y: {y}")
                    print(f"  Z: {z:.4f}")
                    print(f"  Confidence: {confidence:.4f}")
                    print("-" * 40)

            # Calculate and print angles
            angles = self.calculate_pose_angles()
            print("\n Calculated Angles:")
            for angle_name, value in angles.items():
                print(f"{angle_name}: {value:.2f}°")

            # Classify pose
            if self.stable_coordinates:
                input_data = []
                for coord in self.stable_coordinates:
                    input_data.extend(coord[:3])
                
                while len(input_data) < 71:
                    input_data.extend([0, 0, 0])
                
                input_data = np.array(input_data[:71])
                input_data = np.expand_dims(input_data, axis=0)
                
                print(f"\n Input data shape: {input_data.shape}")
                print(f"First few coordinates: {input_data[0][:10]}")  
                try:
                    prediction = self.model.predict(input_data)
                    print(f"\n Raw prediction: {prediction}")
                    predicted_class_idx = np.argmax(prediction)
                    predicted_class = self.pose_classes[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx]
                    print(f"Predicted class index: {predicted_class_idx}")
                    print(f"Predicted class: {predicted_class}")
                    confidence_threshold = 0.0
                    print("\n Pose Classification:")
                    
                    if confidence >= confidence_threshold:
                        if predicted_class != self.current_pose:
                            try:
                                self.current_pose = predicted_class
                                self.ideal_angles = self.get_ideal_angles(predicted_class)
                                print(f"Predicted Pose: {predicted_class}")
                                print(f"Confidence: {confidence:.2f}")
                                
                                # Print ideal angles if available
                                if self.ideal_angles:
                                    print("\n Ideal Angles for this pose:")
                                    for angle_name, stats in self.ideal_angles.items():
                                        print(f"{angle_name}:")
                                        print(f"  Ideal: {stats['mean']:.2f}°")
                                        print(f"  Range: {stats['min']:.2f}° to {stats['max']:.2f}°")
                                        print("-" * 40)
                                else:
                                    print("No ideal angles defined for this pose.")
                            except Exception as e:
                                print(f"Error fetching ideal angles: {str(e)}")
                                self.ideal_angles = None
                        else:
                            # For the same pose, just update the current angles
                            if self.ideal_angles:
                                print("\n Current Angles vs Ideal:")
                                for angle_name in self.ideal_angles:
                                    if angle_name in angles:
                                        current = angles[angle_name]
                                        ideal = self.ideal_angles[angle_name]
                                        print(f"{angle_name}:")
                                        print(f"  Current: {current:.2f}°")
                                        print(f"  Ideal: {ideal['mean']:.2f}°")
                                        print(f"  Range: {ideal['min']:.2f}° to {ideal['max']:.2f}°")
                                        print("-" * 40)
                    else:
                        print("Predicted Pose: Unknown Pose")
                        print(f"Confidence: {confidence:.2f}")
                except Exception as e:
                    print(f"Error in model prediction: {str(e)}")
                    print(f"Input data shape: {input_data.shape}")
                    print(f"First few coordinates: {input_data[0][:10]}")

            print("=" * 60)
            self.needs_printing = False

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Draw proper stick figure on the frame with ideal angles and guidance."""
        # Draw connections
        for p1, p2 in self.connections:
            if landmarks[p1] and landmarks[p2]:
                cv2.line(
                    frame,
                    (int(landmarks[p1][0]), int(landmarks[p1][1])),
                    (int(landmarks[p2][0]), int(landmarks[p2][1])),
                    (0, 255, 0),  # Green color for lines
                    3
                )

        # Draw circles at each keypoint
        for idx, landmark in enumerate(landmarks):
            if landmark:
                cv2.circle(
                    frame,
                    (int(landmark[0]), int(landmark[1])),
                    7,
                    (0, 0, 255),  # Red color for joints
                    -1
                )
                # Add label for each keypoint
                cv2.putText(
                    frame,
                    self.keypoints.get(idx, ""),
                    (int(landmark[0]) + 10, int(landmark[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),  # White text
                    1
                )

        # If we have ideal angles, display them on the frame
        if self.ideal_angles and self.current_pose:
            try:
                angles = self.calculate_pose_angles()
                y_position = 30  # Starting position for text
                
                # Display current pose
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
        }

        angles = {}

        if self.stable_coordinates:
            for angle_name, points in angle_definitions.items():
                if all(p < len(self.stable_coordinates) for p in points):
                    a = self.stable_coordinates[points[0]][:2]
                    b = self.stable_coordinates[points[1]][:2]
                    c = self.stable_coordinates[points[2]][:2]

                    # Calculate angle
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
