
    # async def process_correction(self, frame: np.ndarray, pose_name: str) -> Dict[int, Tuple[float, float, float, float]]:
    #     """Process frame and return pose landmarks."""
    #     try:
    #         if frame is None:
    #             print("\nERROR: Frame is None - Camera connection issue")
    #             return {}

    #         try:
    #             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         except Exception as e:
    #             print(f"\nERROR: Failed to convert frame to RGB: {str(e)}")
    #             return {}

    #         results = self.pose.process(rgb_frame)

    #         landmarks_dict = {}
    #         if results.pose_landmarks:
    #             for i, landmark in enumerate(results.pose_landmarks.landmark):
    #                 if i in self.keypoints:
    #                     x = int(landmark.x * frame.shape[1])
    #                     y = int(landmark.y * frame.shape[0])
    #                     z = landmark.z
    #                     confidence = landmark.visibility
    #                     landmarks_dict[i] = (x, y, z, confidence)

    #         if not landmarks_dict:
    #             print("\nNo keypoints detected")
    #             return {}  

    #         if self.high_fps:
    #             print(f"Current FPS: {self.fps:.2f}")
    #             print(f"Current Pose: {pose_name}")
            
    #         if self.pause_stability:
    #             if time.time() - self.pause_time >= 300:
    #                 self.pause_stability = False
    #                 print("\nResuming stability checking...")
    #             return landmarks_dict

    #         if self.previous_landmarks:
    #             stable_points = sum(
    #                 1 for idx in self.keypoints
    #                 if idx in landmarks_dict and idx in self.previous_landmarks
    #                 and abs(landmarks_dict[idx][0] - self.previous_landmarks[idx][0]) <= self.tolerance_range
    #                 and abs(landmarks_dict[idx][1] - self.previous_landmarks[idx][1]) <= self.tolerance_range
    #             )

    #             print(f"\nStable Points: {stable_points}/{len(self.keypoints)}")

    #             if stable_points >= 7:
    #                 self.stable_time += 1 / self.fps
    #                 if self.stable_time >= self.stability_threshold:
    #                     print("\nPose Stable!")
    #                     print(f"Stable for {self.stable_time:.2f} seconds")
    #                     self.stable_coordinates = landmarks_dict.copy()

    #                     print("-------------------------------------------------->Stable coordinates:",self.stable_coordinates)
                        
                        
    #                     self.pause_stability = True
    #                     self.pause_time = time.time()
                        
                        
    #                     if self.high_fps:
    #                         self.fps = 2
    #                         self.high_fps = False
    #                         print("\nSwitching to 2 FPS for continuous feedback")
                            
    #                     angles = self.calculate_pose_angles(landmarks_dict)
    #                     self.ideal_angles = await self.get_ideal_angles(pose_name, landmarks_dict)
    #                     errors = self.calculate_angle_errors(angles, self.ideal_angles)
                        
                        
    #                     current_time = time.time()
    #                     if current_time - self.last_feedback_time >= 5:
    #                         self.last_feedback_time = current_time
    #                         self.process_feedback_queue(errors)
                        
    #                     if self.high_fps:
    #                         print("\nCalculated Angles:")
    #                         for angle_name, angle_value in angles.items():
    #                             print(f"{angle_name}: {angle_value:.1f} degrees")
                            
    #                         if self.ideal_angles:
    #                             print("\nIdeal Angles:")
    #                             for angle_name, ideal in self.ideal_angles.items():
    #                                 print(f"{angle_name}: Target={ideal['target']:.1f}, Min={ideal['min']:.1f}, Max={ideal['max']:.1f}")
                            
    #                         if errors:
    #                             print("\nAngle Errors:")
    #                             for angle_name, error in errors.items():
    #                                 within_range = "✓" if error['within_range'] else "✗"
    #                                 print(f"{angle_name}: Error={error['error']:.1f}°, Actual={error['actual']:.1f}°, Target={error['target']:.1f}°, Range=[{error['min']:.1f}°-{error['max']:.1f}°] ({within_range})")
                        
    #                     self.print_stable_keypoints(landmarks_dict)

    #         self.previous_landmarks = landmarks_dict
    #         return landmarks_dict

    #     except Exception as e:
    #         print(f"\nERROR in process_correction: {str(e)}")
    #         return {}
    
    
    
    def process_feedback_queue(self, errors):
        """Process errors and provide feedback based on highest errors."""
        if not errors:
            return

        current_time = time.time()
        feedback_text = []
        
        for angle_name, error in errors.items():
            if not error['within_range']:
                # Get the angle data from fixed_ideal_angles
                angle_data = self.fixed_ideal_angles.get(angle_name)
                if angle_data:
                    # Use mean as target angle
                    target = angle_data.get('mean', angle_data.get('target', 0))
                    actual = error['actual']
                    
                    # Generate feedback text
                    feedback = self.generate_feedback_text(angle_name, actual, target)
                    if feedback:
                        feedback_text.append(feedback)
                        
                        # Track this error
                        if angle_name not in self.error_tracking:
                            self.error_tracking[angle_name] = {
                                'error_sum': error['error'],
                                'count': 1,
                                'last_error': error['error']
                            }
                        else:
                            self.error_tracking[angle_name]['error_sum'] += error['error']
                            self.error_tracking[angle_name]['count'] += 1
                            self.error_tracking[angle_name]['last_error'] = error['error']

        # Sort and limit feedback to top 3 errors
        sorted_feedback = sorted(
            [(k, v) for k, v in self.error_tracking.items()],
            key=lambda x: x[1]['error_sum'] / x[1]['count'],
            reverse=True
        )[:3]

        # Print feedback
        print("\nPose Correction Feedback:")
        for angle_name, error_stats in sorted_feedback:
            avg_error = error_stats['error_sum'] / error_stats['count']
            print(f"- {angle_name}: Average Error = {avg_error:.1f}°, Last Error = {error_stats['last_error']:.1f}°")

        # Print detailed feedback
        print("\nDetailed Corrections:")
        for feedback in feedback_text:
            print(f"- {feedback}")

        # Clear feedback queue if pose is correct
        if all(error['within_range'] for error in errors.values()):
            self.feedback_queue.clear()
            print("\nPose is correct! No further corrections needed.")
            self.error_tracking.clear()

    def generate_feedback_text(self, angle_name, actual, target):
        """Generate human-readable feedback text based on angle error."""
        if angle_name.endswith("Elbow_Angle"):
            if actual < target:
                return f"Extend your {angle_name.replace('_Angle', '').lower()} angle fully. error with {abs(actual - target):.2f}"
            else:
                return f"Bend your {angle_name.replace('_Angle', '').lower()} angle slightly. error with {abs(actual - target):.2f}"
        elif angle_name.endswith("Hip_Angle"):
            if actual < target:
                return f"Drop your {angle_name.replace('_Angle', '').lower()} down a little to balance. error with {abs(actual - target):.2f}"
            else:
                return f"Lift your {angle_name.replace('_Angle', '').lower()} up slightly to balance. error with {abs(actual - target):.2f}"
        elif angle_name.endswith("Knee_Angle"):
            if actual < target:
                return f"Bend your {angle_name.replace('_Angle', '').lower()} slightly for balance. error with {abs(actual - target):.2f}"
            else:
                return f"Straighten your {angle_name.replace('_Angle', '').lower()} slightly. error with {abs(actual - target):.2f}"
        elif angle_name.endswith("Ankle_Angle"):
            if actual < target:
                return f"Shift your weight on {angle_name.replace('_Angle', '').lower()} slightly forward onto your toes. error with {abs(actual - target):.2f}"
            else:
                return f"Shift your weight on {angle_name.replace('_Angle', '').lower()} slightly back onto your heels. error with {abs(actual - target):.2f}"
        return None