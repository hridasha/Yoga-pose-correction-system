from typing import List, Tuple
import numpy as np
import cv2

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

    def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
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
                    landmarks.append((x, y, z))
                else:
                    landmarks.append(None)  # Placeholder for missing keypoints

        return landmarks

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
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
