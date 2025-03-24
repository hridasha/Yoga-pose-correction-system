import os
import pandas as pd
import django


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ypc.settings')
django.setup()

from pose_selection.models import YogaPoseDetails, YogaPoseHold
csv_file = "datasets\yoga_poses_final.csv"

df = pd.read_csv(csv_file)
for _,row in df.iterrows():

    YogaPoseDetails.objects.update_or_create(
        pose_name=row['Pose Name'],
        defaults={
            'english_name': row['English Name'],
            'benefits': row['Benefits'],
            'level': row['Level'],
            'hold_duration': row['Hold Duration']
            }
        )

    YogaPoseHold.objects.update_or_create(
        pose_name=row['Pose Name'],
        defaults={
            'english_name': row['English Name'],
            'hold_duration': row['Specific Hold (sec)']
        }
    )

print("Yoga poses imported successfully!")
