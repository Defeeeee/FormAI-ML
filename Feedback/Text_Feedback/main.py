"""
This script analyzes a video of a plank exercise using a pre-trained model,
provides performance feedback through the terminal, and includes additional insights.
"""

import cv2
import mediapipe as mp
import pandas as pd
import joblib
import time


def analyze_video(video_path):
    # Load the trained model
    pipe = joblib.load('../../Models/Core/Plank/model.pkl')

    # Mediapipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Video setup
    cap = cv2.VideoCapture(video_path)

    # Column names for the DataFrame
    columns = ['nose_x', 'nose_y', 'nose_z', 'nose_v', 'left_shoulder_x', 'left_shoulder_y',
               'left_shoulder_z', 'left_shoulder_v', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
               'right_shoulder_v', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z', 'left_elbow_v', 'right_elbow_x',
               'right_elbow_y', 'right_elbow_z', 'right_elbow_v', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
               'left_wrist_v', 'right_wrist_x', 'right_wrist_y', 'right_wrist_z', 'right_wrist_v', 'left_hip_x',
               'left_hip_y', 'left_hip_z', 'left_hip_v', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_v',
               'left_knee_x', 'left_knee_y', 'left_knee_z', 'left_knee_v', 'right_knee_x', 'right_knee_y',
               'right_knee_z',
               'right_knee_v', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_ankle_v', 'right_ankle_x',
               'right_ankle_y', 'right_ankle_z', 'right_ankle_v', 'left_heel_x', 'left_heel_y', 'left_heel_z',
               'left_heel_v',
               'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_heel_v', 'left_foot_index_x', 'left_foot_index_y',
               'left_foot_index_z', 'left_foot_index_v', 'right_foot_index_x', 'right_foot_index_y',
               'right_foot_index_z',
               'right_foot_index_v']

    # Initialize variables for tracking time and stage counts
    start_time = time.time()
    stage_times = {'Correct': 0, 'Low Back': 0, 'High Back': 0}
    current_stage = None
    stage_counts = {'Correct': 0, 'Low Back': 0, 'High Back': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = []

            for landmark_name in ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                                  'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                  'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
                landmark = landmarks[mp_pose.PoseLandmark[landmark_name]]
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            X = pd.DataFrame([row], columns=columns)
            prediction = pipe.predict(X)[0]

            label_map = {0: 'Correct', 1: 'Low Back', 2: 'High Back'}
            predicted_label = label_map[prediction]

            # Update stage time and count tracking
            if current_stage != predicted_label:
                if current_stage:
                    stage_times[current_stage] += time.time() - start_time
                current_stage = predicted_label
                start_time = time.time()
            stage_counts[predicted_label] += 1

    # Update time for the final stage
    if current_stage:
        stage_times[current_stage] += time.time() - start_time

    # Calculate percentages and other metrics
    total_time = sum(stage_times.values())
    total_frames = sum(stage_counts.values())

    percentages = {stage: (time / total_time) * 100 for stage, time in stage_times.items()}
    frame_counts = {stage: (count / total_frames) * 100 for stage, count in stage_counts.items()}

    # Generate additional insights
    additional_insights = ""
    if percentages['Correct'] > 80:
        additional_insights += "\nGreat job maintaining a correct plank position for most of the exercise!"
    if percentages['Low Back'] > 30:
        additional_insights += "\nFocus on keeping your back straight and avoid sagging in the middle."
    if percentages['High Back'] > 30:
        additional_insights += "\nTry to engage your core more and avoid raising your hips too high."

    # Return the results in a dictionary
    return {
        "percentages": percentages,
        "frame_counts": frame_counts,
        "additional_insights": additional_insights
    }
