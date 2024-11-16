import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import requests

mp_pose = mp.solutions.pose

class PlankCNN(nn.Module):
    def __init__(self):
        super(PlankCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16, 7)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_back_curvature(landmarks):
  shoulder_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                       landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
  hip_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
  curvature = abs(shoulder_hip_angle - hip_knee_angle)
  return curvature

def calculate_head_alignment(landmarks):
  shoulder_ear_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                       landmarks[mp_pose.PoseLandmark.LEFT_EAR],
                                       landmarks[mp_pose.PoseLandmark.LEFT_HIP])
  alignment = abs(180 - shoulder_ear_angle)
  return alignment

def calculate_arm_placement(landmarks):
  shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
  elbow_x = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x
  distance = abs(shoulder_x - elbow_x)
  return distance

def calculate_foot_placement(landmarks):
  left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x
  right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x
  distance = abs(left_ankle_x - right_ankle_x)
  return distance

def extract_plank_features(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        features = {
            'shoulder_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE]),
            'hip_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
            'shoulder_elbow_wrist_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]),
            'back_curvature': calculate_back_curvature(landmarks),
            'head_alignment': calculate_head_alignment(landmarks),
            'arm_placement': calculate_arm_placement(landmarks),
            'foot_placement': calculate_foot_placement(landmarks)
        }
        return features

def download_image(image_url):
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def predict_plank_anomaly(image_url, model_path='plank_model.pth', threshold=0.1):
    model = PlankCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = download_image(image_url)
    features = extract_plank_features(image)
    if features:
        input_tensor = torch.tensor(list(features.values()), dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            reconstructed_features = model(input_tensor)
        mse = nn.MSELoss()(input_tensor, reconstructed_features).item()
        print(f"Reconstruction MSE: {mse}")
        if mse > threshold:
            print("Incorrect plank detected!")
        else:
            print("Plank looks good!")
    else:
        print("Unable to analyze the image.")

if __name__ == '__main__':
    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTf52v1-JBv4vOsP2qmU29YHAZ5fgFpLmbSbA&s"
    predict_plank_anomaly(image_url)
