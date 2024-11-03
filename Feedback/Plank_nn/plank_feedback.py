import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import torch.nn.functional as F

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # with 3 hidden layers with 128 neurons each and ReLU activation and a final output layer with 3 neurons and softmax activation
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def load_model(model_path, model_class=Net):
    model = model_class()
    loaded_data = torch.load(model_path)
    if isinstance(loaded_data, dict):  # Check if it's already a state_dict
        model.load_state_dict(loaded_data)
    else:  # If it's the whole model, extract the state_dict
        model.load_state_dict(loaded_data.state_dict())
    model.eval()
    return model


def extract_landmarks_from_frame(frame, pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return [
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y
        ]
    return None


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    predictions = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extract_landmarks_from_frame(frame, pose)
            if landmarks:
                input_tensor = torch.tensor([landmarks], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, prediction = torch.max(output, 1)
                    predictions.append(prediction.item())

    cap.release()
    return predictions


def aggregate_predictions(predictions):
    counter = Counter(predictions)
    most_common_prediction = counter.most_common(1)[0][0]
    return most_common_prediction


def analyze_plank_video(video_path):
    model_path = os.path.join(root, 'Models/Core/Plank/model.pth')
    model = load_model(model_path)
    predictions = process_video(video_path, model)
    result = aggregate_predictions(predictions)

    if result == 0:
        return {
            'correcto': True,
            'issue': None
        }
    else:
        issues = ["high back", "low back"]
        return {
            'correcto': False,
            'issue': issues[result - 1]
        }
