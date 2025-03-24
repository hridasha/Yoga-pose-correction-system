from django.db import models

class YogaPoseIdealAngle(models.Model):
    """
    Model for storing ideal angles of yoga poses, including original and flipped data.
    """
    
    # Pose and View
    pose_name = models.CharField(max_length=100, db_index=True) 
    view = models.CharField(max_length=100, db_index=True)      
    is_flipped = models.BooleanField(default=False)               

    # Left-side angles
    left_elbow_angle_mean = models.FloatField()
    left_elbow_angle_std = models.FloatField()
    left_shoulder_angle_mean = models.FloatField()
    left_shoulder_angle_std = models.FloatField()
    left_knee_angle_mean = models.FloatField()
    left_knee_angle_std = models.FloatField()
    left_hip_angle_mean = models.FloatField()
    left_hip_angle_std = models.FloatField()
    left_ankle_angle_mean = models.FloatField()
    left_ankle_angle_std = models.FloatField()

    # Right-side angles
    right_elbow_angle_mean = models.FloatField()
    right_elbow_angle_std = models.FloatField()
    right_shoulder_angle_mean = models.FloatField()
    right_shoulder_angle_std = models.FloatField()
    right_knee_angle_mean = models.FloatField()
    right_knee_angle_std = models.FloatField()
    right_hip_angle_mean = models.FloatField()
    right_hip_angle_std = models.FloatField()
    right_ankle_angle_mean = models.FloatField()
    right_ankle_angle_std = models.FloatField()

    # Timestamp fields
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set on creation
    updated_at = models.DateTimeField(auto_now=True)      # Automatically updated on save

    def __str__(self):
        """
        String representation for easy debugging.
        """
        flip_status = "Flipped" if self.is_flipped else "Original"
        return f"{self.pose_name} - {self.view} ({flip_status})"

    class Meta:
        unique_together = ('pose_name', 'view', 'is_flipped')  
        indexes = [
            models.Index(fields=['pose_name', 'view', 'is_flipped']),  
        ]


class YogaPoseDetails(models.Model):
    pose_name = models.CharField(max_length=100, unique=True)
    english_name = models.CharField(max_length=100)
    benefits = models.TextField()
    level = models.CharField(max_length=20)
    hold_duration = models.CharField(max_length=20)

    def __str__(self):
        return self.pose_name

class YogaPoseHold(models.Model):
    pose_name = models.CharField(max_length=100, unique=True)
    english_name = models.CharField(max_length=100)
    hold_duration = models.CharField(max_length=20)

    def __str__(self):
        return self.pose_name
