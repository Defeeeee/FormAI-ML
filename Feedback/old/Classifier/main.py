"""
This is the main file for the classifier. It will be used to run the classifier
from the API on a video file or a live feed.
Will return the results of the classification.
"""

import os
import sys
from time import sleep

import cv2
import pandas as pd
import torch
import torch.nn as nn

import mediapipe as mp

# Define the model architecture
class ExerciseClassifier(nn.Module):
    def __init__(self, input_size):
        super(ExerciseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Match the size from the error message
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)         # Match the size from the error message
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)           # Add the missing fc3 layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)                     # Pass through fc3
        return out

# Load your PyTorch model

# Create a new model instance
input_size = 36
model = ExerciseClassifier(input_size)

# Load the state dictionary into the model instance
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model.load_state_dict(torch.load(os.path.join(root, 'Models/Core/Classifier/exercise_classifier_full.pth')))

model.eval()

def classify(path):
    """
    This function will classify the exercise in the video at the given path as either a 'squat' or a 'plank'.
    :param path: The path to the video file.
    :return: The predicted exercise label ('squat' or 'plank').
    """

    # Open MediaPipe and load the pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(path)

    # Print available landmark names for reference
    print("Available landmark names:", [landmark.name for landmark in mp_pose.PoseLandmark])

    # Initialize the columns for the dataframe (matching the training data)
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
                landmark = landmarks[mp_pose.PoseLandmark[name]]

                # Append the appropriate coordinate value
                if coord == 'x':
                    row.append(landmark.x)
                elif coord == 'y':
                    row.append(landmark.y)
                elif coord == 'z':
                    row.append(landmark.z)
                elif coord == 'v':
                    row.append(landmark.visibility)

            # Convert the row to a PyTorch tensor
            X = torch.tensor([row], dtype=torch.float32)

            # Make a prediction using your PyTorch model
            with torch.no_grad():
                output = model(X)
                _, predicted_class = torch.max(output, 1)
                prediction = predicted_class.item()

            predictions.append(prediction)

            # Visualize landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            cv2.imshow('MediaPipe Pose', image)
            while not cv2.waitKey(1) & 0xFF == ord('q'):
                sleep(0.1)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine the most frequent prediction
    most_frequent_prediction = max(set(predictions), key=predictions.count)

    # Map the prediction to the exercise label
    label_map = {0: 'plank', 1: 'squat'}
    predicted_exercise = label_map[most_frequent_prediction]

    return predicted_exercise

if __name__ == '__main__':
    print(classify('https://media.post.rvohealth.io/wp-content/uploads/sites/2/2019/05/ForearmPlank.png'))