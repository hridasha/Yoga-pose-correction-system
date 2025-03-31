from typing import List, Tuple
import math
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

        # Stick figure connections
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
            (24, 26), (26, 28)        # Right Hip → Knee → Ankle
        ]

        # Stability settings
        self.previous_landmarks = None
        self.stable_time = 0
        self.stability_threshold = 0.5  # 0.5 seconds
        self.tolerance_range = 5  # Tolerance range ±5 pixels
        self.last_printed_time = 0
        self.stable_coordinates = None
        self.needs_printing = False

        # Keypoints used for stability checks
        self.stability_keypoints = {
            0: "Nose",
            11: "Left Shoulder",
            12: "Right Shoulder",
            23: "Left Hip",
            24: "Right Hip",
            13: "Left Elbow",
            14: "Right Elbow",
            25: "Left Knee",
            26: "Right Knee"
        }

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
                if i in self.keypoints:  # Only include the 17 keypoints
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    confidence = landmark.visibility
                    landmarks.append((x, y, z, confidence))
                else:
                    landmarks.append(None)  # Placeholder for missing keypoints

        # Debug: Print FPS
        print(f"Current FPS: {self.fps:.2f}")

        # Check stability for specific key points
        if self.previous_landmarks is not None:
            stable = True
            unstable_points = []

            print(f"\nChecking stability for {len(self.stability_keypoints)} key points...")
            print(f"Stability points: {list(self.stability_keypoints.values())}")

            for i in self.stability_keypoints.keys():
                if landmarks[i] and self.previous_landmarks[i]:
                    x, y = landmarks[i][:2]
                    prev_x, prev_y = self.previous_landmarks[i][:2]

                    # Debug: Print current and previous coordinates
                    print(f"\nChecking {self.stability_keypoints[i]}:")
                    print(f"  Current: x={x}, y={y}")
                    print(f"  Previous: x={prev_x}, y={prev_y}")
                    print(f"  Distance: {abs(x-prev_x):.2f}, {abs(y-prev_y):.2f}")

                    # Check if X and Y are within the tolerance range
                    if not (prev_x - self.tolerance_range <= x <= prev_x + self.tolerance_range and
                            prev_y - self.tolerance_range <= y <= prev_y + self.tolerance_range):
                        unstable_points.append(self.stability_keypoints[i])
                        stable = False
                        print(f"  Status: UNSTABLE (Distance > {self.tolerance_range})")
                    else:
                        print(f"  Status: STABLE")

            if stable:
                self.stable_time += 1 / self.fps
                print(f"\nStability time: {self.stable_time:.2f} seconds")
                
                if self.stable_time >= self.stability_threshold:
                    current_time = time.time()
                    if current_time - self.last_printed_time >= 0.5:
                        print("\nStable pose detected!")
                        print(f"Stable for {self.stable_time:.2f} seconds")
                        self.stable_coordinates = landmarks
                        self.needs_printing = True
                        self.last_printed_time = current_time
            else:
                print(f"\nPose unstable - Unstable points: {unstable_points}")
                self.stable_time = 0
                self.needs_printing = False

        self.previous_landmarks = landmarks
        return landmarks

    def print_coordinates(self):
        """Print stable pose coordinates."""
        if self.needs_printing and self.stable_coordinates:
            print("\nStable Pose Coordinates:")
            print("=" * 50)
            print(f"Stability Check Points: {list(self.stability_keypoints.values())}")
            print("=" * 50)
            
            for idx, (x, y, z, confidence) in enumerate(self.stable_coordinates):
                if x is not None:  # Only print valid keypoints
                    print(f"{self.keypoints[idx]}:")
                    print(f"  X: {x}")
                    print(f"  Y: {y}")
                    print(f"  Z: {z:.4f}")
                    print(f"  Confidence: {confidence:.4f}")
            print("=" * 50)
            self.needs_printing = False



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
