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
import tensorflow as tf
import pickle
import mediapipe as mp
from pathlib import Path

# MODEL_PATH = r"D:\YogaPC\ypc\datasets\final_student_model_35.keras"
# POSE_CLASSES_PATH = r"D:\YogaPC\ypc\datasets\pose_classes.pkl"

active_connections = set()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "datasets" / "final_student_model_35.keras")
POSE_CLASSES_PATH = str(PROJECT_ROOT / "datasets" / "pose_classes.pkl")
# MODEL_PATH = r"\datasets\final_student_model_35.keras"
# POSE_CLASSES_PATH = r"\datasets\pose_classes.pkl"


model = tf.keras.models.load_model(MODEL_PATH)
with open(POSE_CLASSES_PATH, 'rb') as f:
    pose_classes = pickle.load(f)
    # print(f"Pose classes loaded: {pose_classes}")
    
landmark_names = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

angle_definitions = {
    "Left_Elbow_Angle": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
    "Left_Shoulder_Angle": ["LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"],
    "Left_Knee_Angle": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
    "Left_Hip_Angle": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
    "Left_Ankle_Angle": ["LEFT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"],
    "Right_Elbow_Angle": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
    "Right_Shoulder_Angle": ["RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"],
    "Right_Knee_Angle": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    "Right_Hip_Angle": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
    "Right_Ankle_Angle": ["RIGHT_KNEE", "RIGHT_ANKLE", "LEFT_ANKLE"]
}

def live_stream(request):
    """ Django view to render live stream """

    try:
        response = requests.get("http://127.0.0.1:8001/status")
        fastapi_status = "Running" if response.status_code == 200 else "Not Running"
    except requests.ConnectionError:
        fastapi_status = "Not Running"

    return render(request, "pose_selection/live_stream.html", {"fastapi_status": fastapi_status})

def stop_stream(request):
    """ AJAX call to stop the FastAPI server """
    
    stop_fastapi_server()
    
    return JsonResponse({"status": "stopped"})


def start_stream(request):
    """ AJAX call to start the FastAPI server """
    
    start_fastapi_server()
    
    return JsonResponse({"status": "started"})


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
    """Handle image upload, process it, and return analysis results."""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save the uploaded image
            image = request.FILES['image']
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image.name)
            with open(image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Process the image using MediaPipe
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True) as pose:
                image = cv2.imread(image_path)
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    # Calculate angles
                    angles = extract_features(results.pose_landmarks.landmark, mp_pose)
                    
                    # Get ideal angles for the pose
                    try:
                        pose_obj = YogaPoseDetails.objects.get(pose_name=pose_name)
                        ideal_angles = {
                            angle: float(value) 
                            for angle, value in pose_obj.ideal_angles.items()
                        }
                    except YogaPoseDetails.DoesNotExist:
                        ideal_angles = {}

                    # Calculate errors
                    errors = calculate_error(angles, ideal_angles)
                    
                    # Get pose classification
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] 
                        for lm in results.pose_landmarks.landmark
                    ])
                    landmarks = landmarks.reshape(1, -1)
                    predicted_pose = pose_classes[np.argmax(model.predict(landmarks))]

                    # Get view classification
                    view = classify_view({
                        'LEFT_SHOULDER_z': landmarks[0][11*3+2],
                        'RIGHT_SHOULDER_z': landmarks[0][12*3+2],
                        'LEFT_HIP_z': landmarks[0][23*3+2],
                        'RIGHT_HIP_z': landmarks[0][24*3+2],
                        'LEFT_KNEE_z': landmarks[0][25*3+2],
                        'RIGHT_KNEE_z': landmarks[0][26*3+2],
                        'LEFT_WRIST_z': landmarks[0][15*3+2],
                        'RIGHT_WRIST_z': landmarks[0][16*3+2]
                    })

                    # Calculate average error
                    avg_error = sum(error['error'] for error in errors.values()) / len(errors)

                    # Generate corrections
                    corrections = []
                    for angle, error in errors.items():
                        if error['error'] > 10:  # Only show corrections for significant errors
                            if error['detected'] < error['ideal']:
                                corrections.append(f"Extend your {angle.lower()} fully")
                            else:
                                corrections.append(f"Relax your {angle.lower()} slightly")

                    # Return JSON response
                    return JsonResponse({
                        'success': True,
                        'annotated_image_url': f'/media/uploads/{image.name}',
                        'predicted_pose': predicted_pose,
                        'view': view,
                        'best_match': predicted_pose,
                        'avg_error': round(avg_error, 2),
                        'corrections': corrections,
                        'errors': errors
                    })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

    return JsonResponse({
        'success': False,
        'error': 'No image uploaded'
    }, status=400)


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

    # Create urls for each view
    view_links = [
        {
            'view': view['view'],
            'upload_url': reverse('upload_image', kwargs={'pose_name': pose_name, 'view': view['view']})
        }
        for view in views
    ]

    context = {
        'pose_name': pose_name,
        'view_links': view_links 
    }
    return render(request, 'pose_selection/yoga_views.html', context)



# def show_views(request, pose_name):
#     views = YogaPoseIdealAngle.objects.filter(
#         pose_name=pose_name,
#         is_flipped=False 
#     ).values('view').distinct()

#     context = {
#         'pose_name': pose_name,
#         'views': views
#     }
#     return render(request, 'pose_selection/show_views.html', context)




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


# def realtime_pose_base(request):
#     return render(request, 'pose_selection/realtime_pose.html')




def yoga_details(request, pose_name):
    pose = YogaPoseDetails.objects.get(pose_name=pose_name)
    pose.benefits = pose.benefits.split(',')
    #pose of same level
    related_poses = YogaPoseDetails.objects.filter(
        level=pose.level
    ).exclude(pose_name=pose_name).order_by('pose_name')[:4]  
    
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

@login_required
def upload_image_for_pose(request):
    """Handle image upload, analysis, and return results."""
    print("=== Starting upload_image_for_pose ===")
    print(f"Request method: {request.method}")
    print(f"Has image file: {bool(request.FILES.get('image'))}")
    
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            print("=== Image upload processing started ===")
            
            image = request.FILES['image']
            image_name = f"{int(time.time())}-{image.name}"
            
            with default_storage.open(f"uploads/{image_name}", 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            image_path = default_storage.path(f"uploads/{image_name}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image is None")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6
            ) as pose:
                results = pose.process(image_rgb)

                if not results.pose_landmarks:
                    return JsonResponse({
                        'error': 'No pose detected in the image.'
                    })

                # Create annotated image
                annotated_image = image.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

                # Save annotated image
                annotated_image_name = f"annotated_{image_name}"
                annotated_image_path = f"uploads/{annotated_image_name}"
                cv2.imwrite(default_storage.path(annotated_image_path), annotated_image)

                # Get pose prediction
                landmarks = results.pose_landmarks.landmark
                features = []
                angles = {}

                print("Extracting landmarks and calculating angles...")
                for landmark in landmark_names:
                    lm = getattr(mp.solutions.pose.PoseLandmark, landmark)
                    features.extend([landmarks[lm].x, landmarks[lm].y, landmarks[lm].z])

                for angle_name, points in angle_definitions.items():
                    coords = [(landmarks[getattr(mp.solutions.pose.PoseLandmark, p)].x,
                               landmarks[getattr(mp.solutions.pose.PoseLandmark, p)].y) for p in points]

                    if len(coords) == 3:
                        angle = calculate_angle(*coords)
                        angles[angle_name] = angle
                        angle_radians = np.deg2rad(angle)
                        features.append(np.sin(angle_radians))
                        features.append(np.cos(angle_radians))

                if len(features) == 71:
                    print("Features complete, making prediction...")
                    prediction = model.predict(np.array([features]))
                    predicted_label = np.argmax(prediction)
                    predicted_pose = pose_classes.get(predicted_label, "Unknown Pose")
                    print(f"Predicted pose: {predicted_pose}")

                    actual_angles = angles
                    print(f"Actual angles: {actual_angles}")

                    row = {
                        'Left_Shoulder_Angle': actual_angles.get('left_shoulder', 0),
                        'Left_Hip_Angle': actual_angles.get('left_hip', 0),
                        'Left_Knee_Angle': actual_angles.get('left_knee', 0),
                        
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

                    print(f"Looking for ideal angles for pose: {predicted_pose}")
                    ideal_angles = YogaPoseIdealAngle.objects.filter(
                        pose_name=predicted_pose.lower()
                    ).first()

                    if not ideal_angles:
                        print(f"No ideal angles found for pose: {predicted_pose}, looking for default pose...")
                        default_pose = YogaPoseIdealAngle.objects.filter(
                            pose_name='default'
                        ).first()
                        if default_pose:
                            print("Using default pose angles")
                            ideal_angles = default_pose
                        else:
                            print("Error: No default pose found")
                            return JsonResponse({
                                "error": f"No ideal angles found for pose: {predicted_pose} and no default pose available",
                                "image_url": f"{settings.MEDIA_URL}uploads/{image_name}",
                                "annotated_image_url": f"{settings.MEDIA_URL}uploads/{annotated_image_name}"
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

                    return JsonResponse({
                        'image_url': f"{settings.MEDIA_URL}uploads/{image_name}",
                        'annotated_image_url': f"{settings.MEDIA_URL}uploads/{annotated_image_name}",
                        'predicted_pose': predicted_pose,
                        'view': classified_view,
                        'best_match': best_match,
                        'avg_error': avg_error,
                        'corrections': corrections,
                        'errors': best_errors
                    })

        except Exception as e:
            print(f"=== Error occurred ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            
            return JsonResponse({
                'error': f'Error processing image: {str(e)}'
            })

    print("=== Returning upload form ===")
    return render(request, 'pose_selection/upload_image_pose.html')

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
