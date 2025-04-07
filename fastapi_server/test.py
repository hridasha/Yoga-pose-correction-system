 def calculate_pose_angles(self) -> Dict[str, float]:
        """Calculate angles between key body parts."""
        angles = {}
        
        # Calculate shoulder angles
        left_shoulder = self.get_2d_coords(11)
        right_shoulder = self.get_2d_coords(12)
        left_elbow = self.get_2d_coords(13)
        right_elbow = self.get_2d_coords(14)
        left_wrist = self.get_2d_coords(15)
        right_wrist = self.get_2d_coords(16)
        
        if all([left_shoulder, left_elbow, left_wrist]):
            try:
                left_shoulder_angle = self.calculate_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )
                angles['Left_Shoulder_Angle'] = left_shoulder_angle
            except:
                pass
                
        if all([right_shoulder, right_elbow, right_wrist]):
            try:
                right_shoulder_angle = self.calculate_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )
                angles['Right_Shoulder_Angle'] = right_shoulder_angle
            except:
                pass

        # Calculate hip angles
        left_hip = self.get_2d_coords(23)
        right_hip = self.get_2d_coords(24)
        left_knee = self.get_2d_coords(25)
        right_knee = self.get_2d_coords(26)
        
        if all([left_hip, left_knee, left_shoulder]):
            try:
                left_hip_angle = self.calculate_angle(
                    left_hip,
                    left_knee,
                    left_shoulder
                )
                angles['Left_Hip_Angle'] = left_hip_angle
            except:
                pass
                
        if all([right_hip, right_knee, right_shoulder]):
            try:
                right_hip_angle = self.calculate_angle(
                    right_hip,
                    right_knee,
                    right_shoulder
                )
                angles['Right_Hip_Angle'] = right_hip_angle
            except:
                pass

        # Calculate knee angles
        left_ankle = self.get_2d_coords(27)
        right_ankle = self.get_2d_coords(28)
        
        if all([left_hip, left_knee, left_ankle]):
            try:
                left_knee_angle = self.calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )
                angles['Left_Knee_Angle'] = left_knee_angle
            except:
                pass
                
        if all([right_hip, right_knee, right_ankle]):
            try:
                right_knee_angle = self.calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )
                angles['Right_Knee_Angle'] = right_knee_angle
            except:
                pass

        # Calculate ankle angles
        if all([left_knee, left_ankle, left_shoulder]):
            try:
                left_ankle_angle = self.calculate_angle(
                    left_knee,
                    left_ankle,
                    left_shoulder
                )
                angles['Left_Ankle_Angle'] = left_ankle_angle
            except:
                pass
                
        if all([right_knee, right_ankle, right_shoulder]):
            try:
                right_ankle_angle = self.calculate_angle(
                    right_knee,
                    right_ankle,
                    right_shoulder
                )
                angles['Right_Ankle_Angle'] = right_ankle_angle
            except:
                pass

        return angles