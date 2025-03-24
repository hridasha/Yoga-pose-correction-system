import os
import django
import pandas as pd

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ypc.settings")
django.setup()

# Import the model
from pose_selection.models import YogaPoseIdealAngle

# CSV path
csv_file ="datasets\ideal_angles_mean_std_no_nan.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

# Iterate through the DataFrame and save both original and flipped data
for _, row in df.iterrows():
    
    # ✅ Import Original Pose
    YogaPoseIdealAngle.objects.create(
        pose_name=row['Folder Name'],     # Same name for original and flipped
        view=row['View'],
        is_flipped=False,  # Original data

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

    # ✅ Import Flipped Pose (same pose name)
    YogaPoseIdealAngle.objects.create(
        pose_name=row['Folder Name'],     # SAME POSE NAME
        view=row['View'],
        is_flipped=True,  # Flipped data

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

print("✅ CSV data (original + flipped) imported successfully!")
