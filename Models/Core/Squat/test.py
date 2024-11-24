import cv2
import mediapipe as mp
import numpy as np
from torch import nn
import torch

mp_pose = mp.solutions.pose

class SquatCNN(nn.Module):
    def __init__(self, input_size):
        super(SquatCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_size // 4), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

model = SquatCNN(input_size=5)  # Create an instance of the SquatCNN model

# Load the trained squat classification model
# Replace 'path/to/your/model.pth' with the actual path to your saved model
model_path = '/Users/defeee/Documents/GitHub/FormAI-ML/Models/Core/Squat/best_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()


def calculate_angle(a, b, c):
    """Calculates the angle between three points in 2D space."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_squat_features(image_path):
    """Extracts squat features from an image using Mediapipe."""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image = cv2.imread(image_path)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None  # Pose not detected

        landmarks = results.pose_landmarks.landmark
        features = {
            'left_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
            'right_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]),
            'left_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                               landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                               landmarks[mp_pose.PoseLandmark.LEFT_KNEE]),
            'right_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]),
            'back_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
        }
        return features

def predict_squat(image_path):
    """Predicts the squat class for an image."""
    features = extract_squat_features(image_path)
    if features:
        input_tensor = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            print(output)
            probability = torch.sigmoid(output).item()
            predicted_class = "Good Squat" if probability > 0.5 else "Bad Squat"
        return predicted_class, probability
    else:
        return "Error: Could not extract features.", None


image_path = '/Users/defeee/Downloads/images (6).jpeg'  # Replace with the actual path to your image
predicted_class, probability = predict_squat(image_path)
print(f"Predicted class: {predicted_class}")
if probability is not None:
    print(f"Probability: {probability:.4f}")