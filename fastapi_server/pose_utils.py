from typing import List, Tuple
import numpy as np
import cv2

# Lazy import MediaPipe
_mp_pose = None

def _get_mp_pose():
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
    return _mp_pose

class PoseDetector:
    def __init__(self):
        mp_pose = _get_mp_pose()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Process frame and return pose landmarks."""
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)
        
        landmarks = []
        if results.pose_landmarks:
            # Extract landmarks and convert to pixel coordinates
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                z = landmark.z
                landmarks.append((x, y, z))
        
        return landmarks

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Draw pose landmarks on the frame."""
        # Draw lines connecting the keypoints
        connections = [
            (12, 14), (14, 16),  # Right arm
            (11, 13), (13, 15),  # Left arm
            (24, 26), (26, 28),  # Right leg
            (23, 25), (25, 27),  # Left leg
            (11, 12), (23, 24),  # Shoulders and hips
            (24, 23), (12, 11),  # Body lines
        ]

        for connection in connections:
            if (landmarks[connection[0]] is not None and 
                landmarks[connection[1]] is not None):
                cv2.line(
                    frame,
                    (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1])),
                    (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1])),
                    (0, 255, 0),  # Green color
                    2
                )

        # Draw circles for each keypoint
        for idx, landmark in enumerate(landmarks):
            if landmark is not None:
                cv2.circle(
                    frame,
                    (int(landmark[0]), int(landmark[1])),
                    5,
                    (0, 0, 255),  # Red color
                    -1
                )

        return frame