import os
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ypc.settings")
django.setup()

from pose_selection.models import YogaPoseIdealAngle

csv_file ="datasets\ideal_angles_mean_std_no_nan.csv"

df = pd.read_csv(csv_file)
for _, row in df.iterrows():
    
    YogaPoseIdealAngle.objects.create(
        pose_name=row['Folder Name'],    
        view=row['View'],
        is_flipped=False,  

        # Left-side angles
        left_elbow_angle_mean=row['Left_Elbow_Angle_mean'],
        left_elbow_angle_std=row['Left_Elbow_Angle_std'],
        left_shoulder_angle_mean=row['Left_Shoulder_Angle_mean'],
        left_shoulder_angle_std=row['Left_Shoulder_Angle_std'],
        left_knee_angle_mean=row['Left_Knee_Angle_mean'],
        left_knee_angle_std=row['Left_Knee_Angle_std'],
        left_hip_angle_mean=row['Left_Hip_Angle_mean'],
        left_hip_angle_std=row['Left_Hip_Angle_std'],
        left_ankle_angle_mean=row['Left_Ankle_Angle_mean'],
        left_ankle_angle_std=row['Left_Ankle_Angle_std'],

        # Right-side angles
        right_elbow_angle_mean=row['Right_Elbow_Angle_mean'],
        right_elbow_angle_std=row['Right_Elbow_Angle_std'],
        right_shoulder_angle_mean=row['Right_Shoulder_Angle_mean'],
        right_shoulder_angle_std=row['Right_Shoulder_Angle_std'],
        right_knee_angle_mean=row['Right_Knee_Angle_mean'],
        right_knee_angle_std=row['Right_Knee_Angle_std'],
        right_hip_angle_mean=row['Right_Hip_Angle_mean'],
        right_hip_angle_std=row['Right_Hip_Angle_std'],
        right_ankle_angle_mean=row['Right_Ankle_Angle_mean'],
        right_ankle_angle_std=row['Right_Ankle_Angle_std'],
    )

    YogaPoseIdealAngle.objects.create(
        pose_name=row['Folder Name'],    
        view=row['View'],
        is_flipped=True, 

        # Flipped: Left <-> Right
        left_elbow_angle_mean=row['Right_Elbow_Angle_mean'],
        left_elbow_angle_std=row['Right_Elbow_Angle_std'],
        left_shoulder_angle_mean=row['Right_Shoulder_Angle_mean'],
        left_shoulder_angle_std=row['Right_Shoulder_Angle_std'],
        left_knee_angle_mean=row['Right_Knee_Angle_mean'],
        left_knee_angle_std=row['Right_Knee_Angle_std'],
        left_hip_angle_mean=row['Right_Hip_Angle_mean'],
        left_hip_angle_std=row['Right_Hip_Angle_std'],
        left_ankle_angle_mean=row['Right_Ankle_Angle_mean'],
        left_ankle_angle_std=row['Right_Ankle_Angle_std'],

        # Flipped: Right <-> Left
        right_elbow_angle_mean=row['Left_Elbow_Angle_mean'],
        right_elbow_angle_std=row['Left_Elbow_Angle_std'],
        right_shoulder_angle_mean=row['Left_Shoulder_Angle_mean'],
        right_shoulder_angle_std=row['Left_Shoulder_Angle_std'],
        right_knee_angle_mean=row['Left_Knee_Angle_mean'],
        right_knee_angle_std=row['Left_Knee_Angle_std'],
        right_hip_angle_mean=row['Left_Hip_Angle_mean'],
        right_hip_angle_std=row['Left_Hip_Angle_std'],
        right_ankle_angle_mean=row['Left_Ankle_Angle_mean'],
        right_ankle_angle_std=row['Left_Ankle_Angle_std'],
    )

print(" CSV data (original + flipped) imported successfully!")
