import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import requests
import json
import os

mp_pose = mp.solutions.pose

# project root folder
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def extract_key_frames(video_url, num_keyframes=20):
    """
    Extracts keyframes from a video URL.

    Args:
        video_url: URL of the video.
        num_keyframes: Number of keyframes to extract.

    Returns:
        A list of keyframes as OpenCV images.
    """
    try:
        # Download the video
        video_path = download_video(video_url)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1,
 num_keyframes, dtype=int)
        keyframes = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                keyframes.append(frame)
        cap.release()
        os.remove(video_path)  # Remove the downloaded video file
        return keyframes
    except Exception as e:
        print(f"Error extracting keyframes from {video_url}: {e}")
        return []

def download_video(video_url):
    """Downloads a video from a URL and saves it as a temporary file."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        temp_file_name = "temp_video.mp4"  # You can customize the file name
        with open(temp_file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_file_name
    except Exception as e:
        print(f"Error downloading video from {video_url}: {e}")
        return None

def predict_plank(video_url, model_path=ROOT + "/Models/Core/Plank/plank_model.pth", threshold=0.1, num_keyframes=5):
    model = PlankCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    keyframes = extract_key_frames(video_url, num_keyframes)

    plank_results = []
    mses = []
    for frame in keyframes:
        features = extract_plank_features(frame)
        if features:
            input_tensor = torch.tensor(list(features.values()), dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)
            with torch.no_grad():
                reconstructed_features = model(input_tensor)

            # Calculate MSE for each feature individually and then average
            mse = np.mean([nn.MSELoss()(input_tensor[:, i], reconstructed_features[:, i]).item()
                           for i in range(input_tensor.shape[1])])

            mses.append(mse)
            plank_results.append(mse <= threshold)

    # Determine majority vote
    is_plank = plank_results.count(True) > len(plank_results) // 2

    result = {
        "success": True,
        "correct": is_plank,
        "message": "Workout looks good!" if is_plank else "Incorrect plank detected!",
        "mse": np.mean(mses) if mses else None,
        "keyframes_results": [bool(x) for x in plank_results]
    }

    return result

if __name__ == "__main__":
    image_url = "https://i.ytimg.com/vi/zygY7fXFOz4/sddefault.jpg"
    prediction_result = predict_plank(image_url)
    print(prediction_result)