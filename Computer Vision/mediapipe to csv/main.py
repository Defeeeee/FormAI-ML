# This file will focus on gathering the data for the plank exercise using the webcam using mediapipe and the OpenCV library.

# 1.0 Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# 2.0 Setting up the mediapipe pose module

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 3.0 Setting up the OpenCV window

images = [
    '/Users/defeee/Downloads/Screenshot 2024-09-08 at 14.47.07.png',
    '/Users/defeee/Downloads/Screenshot 2024-09-08 at 14.46.55.png',
    '/Users/defeee/Downloads/Screenshot 2024-09-08 at 14.46.42.png',
]

# 4.0 Extracting the pose landmarks from the images

for image in images:
    cap = cv2.VideoCapture(image)
    cap.set(3, 640)
    cap.set(4, 480)

    # Create a list to store the landmark data
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find pose landmarks
        results = pose.process(image)

        # Check if any landmarks are detected
        if results.pose_landmarks:
            # Extract the landmark data you need
            landmarks = results.pose_landmarks.landmark
            row = ['C'] # Start with the label

            for landmark_name in ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                                  'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                  'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
                landmark = landmarks[mp_pose.PoseLandmark[landmark_name]]
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Append the row to the data list
            data.append(row)

            # (Optional) Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the image back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Mediapipe Pose', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Create a DataFrame and save to CSV
    columns = ['label', 'nose_x', 'nose_y', 'nose_z', 'nose_v', 'left_shoulder_x', 'left_shoulder_y',
               'left_shoulder_z', 'left_shoulder_v', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
               'right_shoulder_v', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z', 'left_elbow_v', 'right_elbow_x',
               'right_elbow_y', 'right_elbow_z', 'right_elbow_v', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
               'left_wrist_v', 'right_wrist_x', 'right_wrist_y', 'right_wrist_z', 'right_wrist_v', 'left_hip_x',
               'left_hip_y', 'left_hip_z', 'left_hip_v', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_v',
               'left_knee_x', 'left_knee_y', 'left_knee_z', 'left_knee_v', 'right_knee_x', 'right_knee_y', 'right_knee_z',
               'right_knee_v', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_ankle_v', 'right_ankle_x',
               'right_ankle_y', 'right_ankle_z', 'right_ankle_v', 'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_heel_v',
               'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_heel_v', 'left_foot_index_x', 'left_foot_index_y',
               'left_foot_index_z', 'left_foot_index_v', 'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z',
               'right_foot_index_v']

    df = pd.DataFrame(data, columns=columns)

    # add to existing csv file
    if os.path.exists('pose_data.csv'):
        df.to_csv('pose_data.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('pose_data.csv', index=False)
