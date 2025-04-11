async def process_websocket(websocket: WebSocket, pose_name: str):
    frame_count = 0
    feedback_interval = 10
    cooldown_duration = 5
    last_feedback_time = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Camera opened successfully")
    corrector = pose_corrector

    previous_landmarks = None
    stable_time = 0
    stability_threshold = 0.5
    tolerance_range = 5
    ideal_angles_selected = False
    fixed_ideal_angles = None
    last_frame_for_feedback = None
    view_classified = False
    pause_stability = False
    pause_time = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            # Send frame over WebSocket
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            try:
                await websocket.send_bytes(frame_bytes)
                await asyncio.sleep(0.01)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break

            # Only process feedback frame every `feedback_interval` frames
            if frame_count % feedback_interval != 0:
                continue

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = corrector.pose.process(rgb_frame)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue

            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected")
                continue

            # Extract landmarks
            landmarks = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i in corrector.keypoints:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    visibility = landmark.visibility
                    landmarks[i] = (x, y, z, visibility)

            if not landmarks:
                continue

            # Draw keypoints
            for idx, (x, y, z, conf) in landmarks.items():
                if conf > 0.5:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for p1, p2 in corrector.connections:
                if p1 in landmarks and p2 in landmarks:
                    x1, y1, _, _ = landmarks[p1]
                    x2, y2, _, _ = landmarks[p2]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # === Pause stability mode ===
            current_time = time.time()
            if pause_stability:
                if current_time - pause_time >= 300:
                    pause_stability = False
                    logger.info("Resuming stability check")
                else:
                    if current_time - last_feedback_time >= cooldown_duration:
                        last_feedback_time = current_time
                        if last_frame_for_feedback and fixed_ideal_angles:
                            angles = corrector.calculate_pose_angles(last_frame_for_feedback)
                            errors = corrector.calculate_angle_errors(angles, fixed_ideal_angles)
                            corrector.process_feedback_queue(errors)
                    continue

            # === Stability Check ===
            if previous_landmarks:
                stable_points = sum(
                    1 for idx in corrector.keypoints
                    if idx in landmarks and idx in previous_landmarks
                    and abs(landmarks[idx][0] - previous_landmarks[idx][0]) <= tolerance_range
                    and abs(landmarks[idx][1] - previous_landmarks[idx][1]) <= tolerance_range
                )

                if stable_points >= 7:
                    stable_time += 1 / corrector.fps
                    if stable_time >= stability_threshold:
                        logger.info(f"Pose Stable! Stable for {stable_time:.2f} seconds")
                        stable_coordinates = landmarks.copy()
                        pause_stability = True
                        pause_time = current_time
                        last_frame_for_feedback = stable_coordinates.copy()

                        # classify view once
                        if not view_classified:
                            corrector.current_view = corrector.classify_view(stable_coordinates)
                            view_classified = True
                            logger.info(f"View classified as: {corrector.current_view}")

                        # select ideal angles once
                        if not ideal_angles_selected:
                            try:
                                ideal_data = await corrector.get_ideal_angles(pose_name, stable_coordinates)
                                for angle, val in ideal_data.items():
                                    if not all(k in val for k in ("mean", "min", "max")):
                                        logger.warning(f"Incomplete angle data for {angle}")
                                fixed_ideal_angles = ideal_data
                                ideal_angles_selected = True
                            except Exception as e:
                                logger.error(f"Failed to get ideal angles: {e}")

                        # feedback after pause
                        if last_frame_for_feedback and fixed_ideal_angles:
                            angles = corrector.calculate_pose_angles(last_frame_for_feedback)
                            errors = corrector.calculate_angle_errors(angles, fixed_ideal_angles)
                            corrector.process_feedback_queue(errors)

            previous_landmarks = landmarks.copy()

    except Exception as e:
        logger.error(f"Unhandled error in process_websocket: {e}")
    finally:
        cap.release()
