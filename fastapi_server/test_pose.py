def calculate_error(self, actual, ideal):
        """Calculate the angle errors between actual and ideal angles."""
        errors = {}
        for angle in actual:
            detected_value = actual.get(angle, 0)
            ideal_value = ideal.get(angle, 0)
            error_value = abs(detected_value - ideal_value)

            errors[angle] = {
                "detected": round(detected_value, 2),
                "ideal": round(ideal_value, 2),
                "error": round(error_value, 2)
            }
        return errors

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
                            
                        print(f"\nView: {view_name} (Flipped={is_flipped})")
                        print(f"Total Error: {total_error:.2f}")
                        print("Angle Errors:")
                        for angle, error in errors.items():
                            print(f"{angle}: Detected={error['detected']}°, Ideal={error['ideal']}°, Error={error['error']}°")

                except Exception as e:
                    print(f"Error processing view {view_name}: {e}")
                    continue

            if best_angles:
                print(f"\nBest Matching View: {best_view}")
                print(f"Total Error: {min_error:.2f}")
                print("\nBest Angle Errors:")
                for angle, error in best_errors.items():
                    print(f"{angle}: Detected={error['detected']}°, Ideal={error['ideal']}°, Error={error['error']}°")
                
                return best_angles
            else:
                print(f"No ideal angles found for pose: {pose_name}")
                return {}

        except Exception as e:
            print(f"Error fetching ideal angles: {e}")
            return {}



def generate_feedback(errors):
    feedback = []
    
    for joint, data in errors.items():
        detected, ideal, error = data["detected"], data["ideal"], data["error"]
        
        if "Elbow" in joint:
            if detected < ideal:
                feedback.append(f"Try **straightening your {joint.lower().replace('_', ' ')}** more.")
            else:
                feedback.append(f"**Bend your {joint.lower().replace('_', ' ')} a little**.")
        
        elif "Shoulder" in joint:
            if detected < ideal:
                feedback.append(f"**Lift your {joint.lower().replace('_', ' ')} higher**.")
            else:
                feedback.append(f"**Lower your {joint.lower().replace('_', ' ')} slightly**.")
        
        elif "Hip" in joint:
            if detected < ideal:
                feedback.append(f"**Raise your {joint.lower().replace('_', ' ')} slightly**.")
            else:
                feedback.append(f"**Shift your {joint.lower().replace('_', ' ')} downward a bit**.")
        
        elif "Knee" in joint:
            if detected < ideal:
                feedback.append(f"**Straighten your {joint.lower().replace('_', ' ')} a little**.")
            else:
                feedback.append(f"**Bend your {joint.lower().replace('_', ' ')} slightly**.")
        
        elif "Ankle" in joint:
            if detected < ideal:
                feedback.append(f"**Tilt your {joint.lower().replace('_', ' ')} slightly forward**.")
            else:
                feedback.append(f"**Adjust your {joint.lower().replace('_', ' ')} a little**.")
    
    return "\n".join(feedback)


# import cv2
# import numpy as np
# import time
# import os
# import sys
# import django

# # Configure Django settings

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ypc.settings')
# django.setup()
# from pose_utils import PoseDetector
# from pose_selection.models import YogaPoseIdealAngle

# def analyze_video(video_path: str):
#     """
#     Analyze poses from a video file and display the results.
#     """
#     # Initialize pose detector
#     detector = PoseDetector()
    
#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     frame_count = 0
#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # Process frame
#         landmarks = detector.process_frame(frame)
        
#         # Draw landmarks
#         if landmarks:
#             frame = detector.draw_pose_landmarks(frame, landmarks)

#         # Print coordinates if stable
#         detector.print_stable_coordinates()

#         # Display frame
#         cv2.imshow('Pose Analysis', frame)
        
#         # Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     # Calculate and print statistics
#     end_time = time.time()
#     total_time = end_time - start_time
#     avg_fps = frame_count / total_time
    
#     print(f"\nAnalysis Complete!")
#     print(f"Total frames processed: {frame_count}")
#     print(f"Total time: {total_time:.2f} seconds")
#     print(f"Average FPS: {avg_fps:.2f}")

# video_path = r"D:\YogaPC\ypc\datasets\dhanurasana.mp4"
    
#     # Test with a sample video
# analyze_video(video_path)
