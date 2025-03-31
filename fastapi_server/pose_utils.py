from typing import List, Tuple
import numpy as np
import cv2
import time

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
    def __init__(self):
        """Initialize MediaPipe pose model."""
        mp_pose = _get_mp_pose()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6
        )

        # Keypoints dictionary
        self.keypoints = {
            0: "Nose",
            11: "Left Shoulder",
            12: "Right Shoulder",
            23: "Left Hip",
            24: "Right Hip",
            13: "Left Elbow",
            14: "Right Elbow",
            25: "Left Knee",
            26: "Right Knee",
            27: "Left Ankle",
            28: "Right Ankle",
            15: "Left Wrist",
            16: "Right Wrist"
        }

        # Stability keypoints
        self.stability_keypoints = list(self.keypoints.keys())

        # Stability settings
        self.previous_landmarks = None
        self.stable_time = 0
        self.stability_threshold = 0.1  # 1 second threshold for stability
        self.tolerance_range = 5  # ±5 pixels tolerance
        self.stable_coordinates = None
        self.needs_printing = False
        self.pause_stability = False
        self.pause_time = 0

        # FPS tracking
        self.last_time = time.time()
        self.fps = 30.0

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
            if time.time() - self.pause_time >= 1.0:  # Pause for 1 second
                self.pause_stability = False
                print("\nResuming stability checking...")
            return landmarks

        # Stability check
        if self.previous_landmarks is not None:
            stable_points = 0

            for idx in self.stability_keypoints:
                if landmarks[idx] and self.previous_landmarks[idx]:
                    x, y = landmarks[idx][:2]
                    prev_x, prev_y = self.previous_landmarks[idx][:2]

                    # Check stability
                    if (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
                            prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
                        stable_points += 1

            # Print stability status
            print(f"\nStable Points: {stable_points}/{len(self.stability_keypoints)}")

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

    def print_stable_coordinates(self):
        """Print stable pose X, Y, Z coordinates and angles."""
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

            print("=" * 60)
            self.needs_printing = False

    def calculate_pose_angles(self) -> dict:
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


    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Draw proper stick figure on the frame."""
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

        return frame
