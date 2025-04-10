
@login_required
def analyze_pose_view(request):
    import mediapipe as mp
    print(f"=== Starting analyze_pose_view ===")
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
        annoted_image = image.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annoted_image,
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
        print("Landmarks extracted, calculating angles...")
        angles = calculate_angles(landmarks)
        print("Angles calculated, preparing response...")
        response_data = {
            "image_url": image_url,
            "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks],
            "angles": angles
        }
    return render(request, 'pose_selection/analyze_pose_view.html')