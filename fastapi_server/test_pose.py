import cv2
from pose_utils import PoseDetector
import time

def main():
    # Initialize pose detector
    detector = PoseDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera!")
        print("Please check:")
        print("1. Is your webcam connected?")
        print("2. Is another application using the camera?")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    
    while True:
        try:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
                
            # Process frame and print coordinates
            landmarks = detector.process_frame(frame)
            
            # Print coordinates
            print("\nPose Coordinates:")
            for idx, (x, y, z) in enumerate(landmarks):
                if x is not None:  # Only print valid keypoints
                    print(f"{detector.keypoints[idx]}:")
                    print(f"  X: {x}")
                    print(f"  Y: {y}")
                    print(f"  Z: {z:.4f}")
            
            # Draw landmarks on frame
            detector.draw_pose_landmarks(frame, landmarks)
            
            # Show the frame
            cv2.imshow('Pose Detection', frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Add a small delay to prevent CPU overload
            time.sleep(0.03)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()