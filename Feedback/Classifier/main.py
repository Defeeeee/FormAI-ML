"""
This is the main file for the classifier. It will be used to run the classifier
from the API on a video file or a live feed.
Will return the results of the classification.
"""

import os
import sys

import cv2
import joblib
import pandas as pd

import mediapipe as mp

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classify(path):
    """
    This function will classify the exercise in the video at the given path as either a 'squat' or a 'plank'.
    :param path: The path to the video file.
    :return: The predicted exercise label ('squat' or 'plank').
    """

    model = joblib.load(os.path.join(root, 'Models/Core/Classifier/classifier_improved.pkl'))

    # open mediapipe and load the pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # open the video file
    cap = cv2.VideoCapture(path)

    # Print available landmark names for reference
    print("Available landmark names:", [landmark.name for landmark in mp_pose.PoseLandmark])

    # initialize the columns for the dataframe (matching the training data)
    columns = ['RIGHT_HIP_v',
               'LEFT_ANKLE_y',
               'LEFT_ANKLE_v',
               'NOSE_y',
               'LEFT_SHOULDER_v',
               'LEFT_HIP_x',
               'LEFT_SHOULDER_y',
               'RIGHT_SHOULDER_z',
               'RIGHT_SHOULDER_x',
               'LEFT_KNEE_x',
               'RIGHT_SHOULDER_v',
               'NOSE_z',
               'RIGHT_ANKLE_v',
               'LEFT_HIP_y',
               'NOSE_v',
               'RIGHT_HIP_x',
               'LEFT_SHOULDER_z',
               'LEFT_KNEE_y',
               'RIGHT_KNEE_y',
               'RIGHT_HIP_y',
               'RIGHT_SHOULDER_y',
               'LEFT_KNEE_v',
               'RIGHT_KNEE_v',
               'LEFT_ANKLE_x',
               'RIGHT_KNEE_z',
               'LEFT_HIP_z',
               'LEFT_SHOULDER_x',
               'LEFT_ANKLE_z',
               'RIGHT_HIP_z',
               'RIGHT_ANKLE_y',
               'RIGHT_ANKLE_x',
               'NOSE_x',
               'RIGHT_KNEE_x',
               'LEFT_HIP_v',
               'RIGHT_ANKLE_z',
               'LEFT_KNEE_z']

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = []

            # Extract landmarks based on your 'columns' list
            for landmark_name in columns:
                # Split landmark name and coordinate
                name, coord = landmark_name.rsplit('_', 1)
                landmark = landmarks[mp_pose.PoseLandmark[name]]  # Use uppercase name

                # Append the appropriate coordinate value
                if coord == 'x':
                    row.append(landmark.x)
                elif coord == 'y':
                    row.append(landmark.y)
                elif coord == 'z':
                    row.append(landmark.z)
                elif coord == 'v':
                    row.append(landmark.visibility)

            X = pd.DataFrame([row], columns=columns)
            prediction = model.predict(X)[0]
            predictions.append(prediction)

    # Determine the most frequent prediction
    most_frequent_prediction = max(set(predictions), key=predictions.count)

    # Map the prediction to the exercise label
    label_map = {0: 'plank', 1: 'squat'}  # Adjust if your model uses different labels
    predicted_exercise = label_map[most_frequent_prediction]

    return predicted_exercise

if __name__ == '__main__':
    print(classify('https://s36370.pcdn.co/wp-content/uploads/2016/07/BW-Squat-Finish-Side-View.jpg'))