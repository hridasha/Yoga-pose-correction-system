from django.shortcuts import render, redirect, HttpResponse
from .models import YogaPoseDetails, YogaPoseIdealAngle
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import JsonResponse
import cv2
import os
import numpy as np
from django.urls import reverse
import subprocess
import requests
import signal
from django.contrib.auth.decorators import login_required
import time
from .fastapi_manager import start_fastapi_server, stop_fastapi_server

def live_stream(request):
    """ Django view to render live stream """

    try:
        # Check FastAPI server status
        response = requests.get("http://127.0.0.1:8001/status")
        fastapi_status = "Running" if response.status_code == 200 else "Not Running"
    except requests.ConnectionError:
        fastapi_status = "Not Running"

    return render(request, "pose_selection/live_stream.html", {"fastapi_status": fastapi_status})
def stop_stream(request):
    """ AJAX call to stop the FastAPI server """
    
    stop_fastapi_server()
    
    return JsonResponse({"status": "stopped"})

def home(request):
    difficulty = request.GET.get('level', 'Beginner')

    poses = YogaPoseDetails.objects.filter(level__iexact=difficulty).order_by('pose_name')
    for pose in poses:
        pose.benefits = pose.benefits.split(',')
    context = {
        'poses': poses,
        'selected_level': difficulty
    }

    return render(request, 'home.html', context)



def filter_poses(request):
    """Handle AJAX filtering of yoga poses by level."""
    
    level = request.GET.get('level', '')
    
    if level:
        poses = YogaPoseDetails.objects.filter(level=level)
    else:
        poses = YogaPoseDetails.objects.all()

    pose_data = [
        {
            "pose_name": pose.pose_name,
            "level": pose.level
        }
        for pose in poses
    ]

    return JsonResponse({"poses": pose_data})
def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def extract_features(landmarks, mp_pose):
    """Extract angles from landmarks."""
    angle_definitions = {
        "Left_Elbow_Angle": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "Left_Shoulder_Angle": ["LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"],
        "Left_Knee_Angle": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "Left_Hip_Angle": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
        "Right_Elbow_Angle": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "Right_Shoulder_Angle": ["RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"],
        "Right_Knee_Angle": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
        "Right_Hip_Angle": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"]
    }

    angles = {}

    for angle_name, points in angle_definitions.items():
        coords = [(landmarks[getattr(mp_pose.PoseLandmark, p)].x,
                   landmarks[getattr(mp_pose.PoseLandmark, p)].y) for p in points]

        if len(coords) == 3:
            angles[angle_name] = calculate_angle(*coords)

    return angles


def calculate_error(actual, ideal):
    """Calculate the angle errors."""
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

def upload_image(request, pose_name):
    """Handle image upload and redirect to analysis with proper parameters."""
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_name = f"{int(time.time())}-{image.name}"
        
        # Save the image
        with default_storage.open(f"uploads/{image_name}", 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Return the redirect URL as plain text
        return HttpResponse(
            reverse('analyze_pose', kwargs={'pose_name': pose_name}) + f'?image_name={image_name}'
        )
    
    # For GET request, show the upload form
    return render(request, 'pose_selection/upload_image.html', {
        'pose_name': pose_name
    })

def classify_view(row):
    """
    Further Enhanced View Classification with more sub-categories.
    """
    try:
        shoulder_angle = float(row['Left_Shoulder_Angle'])
        hip_angle = float(row['Left_Hip_Angle'])
        knee_angle = float(row['Left_Knee_Angle'])

        shoulder_depth_diff = abs(float(row['LEFT_SHOULDER_z']) - float(row['RIGHT_SHOULDER_z']))
        hip_depth_diff = abs(float(row['LEFT_HIP_z']) - float(row['RIGHT_HIP_z']))
        knee_depth_diff = abs(float(row['LEFT_KNEE_z']) - float(row['RIGHT_KNEE_z']))
        
        wrist_depth_diff = abs(float(row['LEFT_WRIST_z']) - float(row['RIGHT_WRIST_z']))
        elbow_depth_diff = abs(float(row['LEFT_ELBOW_z']) - float(row['RIGHT_ELBOW_z']))

        shoulder_height_diff = abs(float(row['LEFT_SHOULDER_y']) - float(row['RIGHT_SHOULDER_y']))
        hip_height_diff = abs(float(row['LEFT_HIP_y']) - float(row['RIGHT_HIP_y']))
        knee_height_diff = abs(float(row['LEFT_KNEE_y']) - float(row['RIGHT_KNEE_y']))

        shoulder_hip_dist = abs(float(row['LEFT_SHOULDER_x']) - float(row['LEFT_HIP_x']))
        knee_hip_dist = abs(float(row['LEFT_KNEE_x']) - float(row['LEFT_HIP_x']))

        pose = row.get("Pose", "").lower()

        if (shoulder_depth_diff < 0.1 and hip_depth_diff < 0.1) and \
           (shoulder_height_diff < 0.1 and hip_height_diff < 0.1):
            
            if knee_depth_diff < 0.1:
                return "Front View (Perfect)"
            
            elif knee_depth_diff < 0.2:
                return "Front View (Partial)"
            
            else:
                return "Front View (Mixed)"
        elif (shoulder_depth_diff > 0.5 and hip_depth_diff > 0.5) and \
             (shoulder_height_diff < 0.2 and hip_height_diff < 0.2):
            
            if knee_depth_diff > 0.5:
                return "Back View (Full)"
            
            elif knee_depth_diff > 0.3:
                return "Back View (Partial)"
            
            else:
                return "Back View (Mixed)"

        elif (shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3) and \
             (shoulder_height_diff < 0.1 and hip_height_diff < 0.1):
            
            # if shoulder_depth_diff > 0.6 and hip_depth_diff > 0.6:
            #     return "Side View (Perfect - Full Profile)"
            
            if shoulder_depth_diff > 0.4 and hip_depth_diff > 0.4:
                return "Side View (Perfect - Near Full)"
            
            elif shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3:
                return "Side View (Perfect - Partial)"
            
            else:
                return "Side View (Intermediate)"

        elif (shoulder_depth_diff > 0.2 and hip_depth_diff > 0.2) and \
             (shoulder_height_diff > 0.1 and hip_height_diff > 0.1):
            
            if shoulder_depth_diff > 0.4 and hip_depth_diff > 0.4:
                return "Oblique View (Strong)"
            
            elif shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3:
                return "Oblique View (Moderate)"
            
            else:
                return "Oblique View (Mild)"

        elif (wrist_depth_diff > 0.4 or elbow_depth_diff > 0.4):
            
            if wrist_depth_diff > 0.6 and elbow_depth_diff > 0.6:
                return "Arm-Specific (Extended Side View)"
            
            elif wrist_depth_diff > 0.4:
                return "Arm-Specific (Partial Extension)"
            
            else:
                return "Arm-Specific (Mixed)"
        else: 
            return "Rare or Mixed View"
        # elif (shoulder_depth_diff > 0.25 and hip_depth_diff > 0.25) and \
        #      (shoulder_height_diff > 0.15 and hip_height_diff > 0.15):
            
        #     if shoulder_depth_diff > 0.4 and hip_depth_diff > 0.4:
        #         return "Diagonal View (Strong)"
            
        #     elif shoulder_depth_diff > 0.3 and hip_depth_diff > 0.3:
        #         return "Diagonal View (Moderate)"
            
        #     else:
        #         return "Diagonal View (Mild)"

        # elif (shoulder_depth_diff > 0.2 and hip_depth_diff > 0.2) and \
        #      (shoulder_height_diff > 0.1 and hip_height_diff > 0.1):
            
        #     if wrist_depth_diff > 0.3 and elbow_depth_diff > 0.3:
        #         return "Hybrid View (Mixed)"
            
        #     else:
        #         return "Hybrid View (Partial)"

        # elif (shoulder_height_diff < 0.15 and hip_height_diff < 0.15) and \
        #      (shoulder_depth_diff > 0.5 and hip_depth_diff > 0.5):
            
        #     return "Low-Angle View"

        # elif (shoulder_height_diff > 0.5 and hip_height_diff > 0.5) and \
        #      (shoulder_depth_diff < 0.2 and hip_depth_diff < 0.2):
            
        #     return "High-Angle View"

        # else:
            # return "Rare or Mixed View"

    except Exception as e:
        return f"Unknown View: {str(e)}"

def yoga_options(request, pose_name):
    context = {
        'pose_name': pose_name
    }
    return render(request, 'pose_selection/yoga_options.html', context)


def live_correction(request, pose_name):
    try:
        pose = YogaPoseDetails.objects.get(pose_name=pose_name)
        context = {
            'pose_name': pose_name,
            'pose': pose
        }
        return render(request, 'pose_selection/live_correction.html', context)
    except YogaPoseDetails.DoesNotExist:
        return HttpResponse("Pose not found", status=404)

def yoga_poses(request):
    """Display unique yoga poses."""
    poses = YogaPoseDetails.objects.all()
    for pose in poses:
        pose.benefits = pose.benefits.split(',')
    context = {
        'poses': poses,
    }
    return render(request, 'pose_selection/yoga_poses.html', context)

def yoga_views(request, pose_name):
    """Display the views for a given pose with links to upload image."""
    views = YogaPoseIdealAngle.objects.filter(
        pose_name=pose_name,
        is_flipped=False
    ).values('view').distinct()

    # Create URLs for each view
    view_links = [
        {
            'view': view['view'],
            'upload_url': reverse('upload_image', kwargs={'pose_name': pose_name, 'view': view['view']})
        }
        for view in views
    ]

    context = {
        'pose_name': pose_name,
        'view_links': view_links  # Send view links to the template
    }
    return render(request, 'pose_selection/yoga_views.html', context)



def show_views(request, pose_name):
    # Query distinct views for the selected pose
    views = YogaPoseIdealAngle.objects.filter(
        pose_name=pose_name,
        is_flipped=False  # Exclude flipped poses
    ).values('view').distinct()

    context = {
        'pose_name': pose_name,
        'views': views
    }
    return render(request, 'pose_selection/show_views.html', context)




@login_required
def analyze_pose(request, pose_name):
    import mediapipe as mp
    print(f"=== Starting analyze_pose for {pose_name} ===")
    print(f"Request: {request.GET}")

    image_name = request.GET.get('image_name')
    if not image_name:
        print("Error: No image specified")
        return JsonResponse({"error": "No image specified."})

    try:
        image_path = default_storage.path(f"uploads/{image_name}")
        print(f"Attempting to read image from: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return JsonResponse({"error": f"Could not read image: {image_name}"})

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6
        ) as pose:

            print("Processing image with MediaPipe Pose...")
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                print("Error: No pose detected")
                return JsonResponse({"error": "No pose detected."})

            print("Pose detected, creating annotated image...")
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )

            annotated_image_name = f"annotated_{image_name}"
            annotated_image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', annotated_image_name)
            print(f"Saving annotated image to: {annotated_image_path}")
            os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
            cv2.imwrite(annotated_image_path, annotated_image)
            image_url = f"{settings.MEDIA_URL}uploads/{annotated_image_name}"

            print("Extracting landmarks and angles...")
            landmarks = results.pose_landmarks.landmark
            actual_angles = extract_features(landmarks, mp.solutions.pose)
            print(f"Actual angles extracted: {actual_angles}")

            row = {
                'Left_Shoulder_Angle': actual_angles.get('Left_Shoulder_Angle', 0),
                'Left_Hip_Angle': actual_angles.get('Left_Hip_Angle', 0),
                'Left_Knee_Angle': actual_angles.get('Left_Knee_Angle', 0),
                
                'LEFT_SHOULDER_z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z,
                'RIGHT_SHOULDER_z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z,
                
                'LEFT_HIP_z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].z,
                'RIGHT_HIP_z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].z,
                
                'LEFT_KNEE_z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].z,
                'RIGHT_KNEE_z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].z,
                
                'LEFT_WRIST_z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].z,
                'RIGHT_WRIST_z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].z,
                
                'LEFT_ELBOW_z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].z,
                'RIGHT_ELBOW_z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].z,

                'LEFT_SHOULDER_y': landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y,
                'RIGHT_SHOULDER_y': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y,
                
                'LEFT_HIP_y': landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
                'RIGHT_HIP_y': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
                
                'LEFT_KNEE_y': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
                'RIGHT_KNEE_y': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
                
                'LEFT_SHOULDER_x': landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                'LEFT_HIP_x': landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
                
                'LEFT_KNEE_x': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
            }

            print("Classifying view...")
            classified_view = classify_view(row)
            print(f"Classified view: {classified_view}")
            
            print(f"Looking for ideal angles for pose: {pose_name}")
            ideal_angles = YogaPoseIdealAngle.objects.filter(
                pose_name=pose_name
            ).first()

            if not ideal_angles:
                print(f"No ideal angles found for pose: {pose_name}, looking for default pose...")
                default_pose = YogaPoseIdealAngle.objects.filter(
                    pose_name='default'
                ).first()
                if default_pose:
                    print("Using default pose angles")
                    ideal_angles = default_pose
                else:
                    print("Error: No default pose found")
                    return JsonResponse({
                        "error": f"No ideal angles found for pose: {pose_name} and no default pose available",
                        "image_url": image_url
                    })

            print(f"Found ideal angles: {ideal_angles}")
            print("Calculating errors...")
            original_angles = {
                "Left_Elbow_Angle": ideal_angles.left_elbow_angle_mean,
                "Right_Elbow_Angle": ideal_angles.right_elbow_angle_mean,
                "Left_Shoulder_Angle": ideal_angles.left_shoulder_angle_mean,
                "Right_Shoulder_Angle": ideal_angles.right_shoulder_angle_mean,
                "Left_Knee_Angle": ideal_angles.left_knee_angle_mean,
                "Right_Knee_Angle": ideal_angles.right_knee_angle_mean,
                "Left_Hip_Angle": ideal_angles.left_hip_angle_mean,
                "Right_Hip_Angle": ideal_angles.right_hip_angle_mean,
                "Left_Ankle_Angle": ideal_angles.left_ankle_angle_mean,
                "Right_Ankle_Angle": ideal_angles.right_ankle_angle_mean
            }

            flipped_angles = {
                "Left_Elbow_Angle": ideal_angles.right_elbow_angle_mean,
                "Right_Elbow_Angle": ideal_angles.left_elbow_angle_mean,
                "Left_Shoulder_Angle": ideal_angles.right_shoulder_angle_mean,
                "Right_Shoulder_Angle": ideal_angles.left_shoulder_angle_mean,
                "Left_Knee_Angle": ideal_angles.right_knee_angle_mean,
                "Right_Knee_Angle": ideal_angles.left_knee_angle_mean,
                "Left_Hip_Angle": ideal_angles.right_hip_angle_mean,
                "Right_Hip_Angle": ideal_angles.left_hip_angle_mean,
                "Left_Ankle_Angle": ideal_angles.right_ankle_angle_mean,
                "Right_Ankle_Angle": ideal_angles.left_ankle_angle_mean
            }

            print("Calculating original errors...")
            original_errors = calculate_error(actual_angles, original_angles)
            print(f"Original errors: {original_errors}")
            print("Calculating flipped errors...")
            flipped_errors = calculate_error(actual_angles, flipped_angles)
            print(f"Flipped errors: {flipped_errors}")

            avg_error_original = np.mean([e['error'] for e in original_errors.values()])
            avg_error_flipped = np.mean([e['error'] for e in flipped_errors.values()])
            print(f"Average errors - Original: {avg_error_original}, Flipped: {avg_error_flipped}")

            best_match = "Flipped Pose" if avg_error_flipped < avg_error_original else "Original Pose"
            best_errors = flipped_errors if avg_error_flipped < avg_error_original else original_errors
            avg_error = round(min(avg_error_original, avg_error_flipped), 2)
            print(f"Best match: {best_match}, Average error: {avg_error}")

            corrections = []
            for joint, error in best_errors.items():
                if error['error'] > 5:
                    direction = "Lift" if error['error'] > 0 else "Lower"
                    correction = f"{direction} your {joint.replace('_', ' ').lower()} by {round(abs(error['error']), 1)}°"
                    corrections.append(correction)
            print(f"Corrections: {corrections}")

            if not corrections:
                print("No significant corrections needed")
                corrections.append("Pose is nearly perfect!")

            feedback = {
                "pose_name": pose_name,
                "view": classified_view,
                "best_match": best_match,
                "avg_error": avg_error,
                "corrections": corrections,
                "errors": best_errors,
                "image_url": image_url
            }
            print(f"Final feedback: {feedback}")

            return render(request, 'pose_selection/analysis.html', {'feedback': feedback})

    except Exception as e:
        print(f"Error in analyze_pose: {str(e)}")
        return JsonResponse({"error": f"An error occurred: {str(e)}"})

def realtime_pose_base(request):
    return render(request, 'pose_selection/realtime_pose.html')


def yoga_details(request, pose_name):
    pose = YogaPoseDetails.objects.get(pose_name=pose_name)
    pose.benefits = pose.benefits.split(',')
    
    # Get related poses of the same level
    related_poses = YogaPoseDetails.objects.filter(
        level=pose.level
    ).exclude(pose_name=pose_name).order_by('pose_name')[:4]  # Get 4 related poses
    
    context = {
        'pose': pose,
        'related_poses': related_poses
    }
    return render(request, 'pose_selection/yoga_details.html', context)

def yoga_poses(request):
    """Display unique yoga poses."""
    poses = YogaPoseDetails.objects.all()
    for pose in poses:
        pose.benefits = pose.benefits.split(',')
    context = {
        'poses': poses,
    }
    return render(request, 'pose_selection/yoga_poses.html', context)
