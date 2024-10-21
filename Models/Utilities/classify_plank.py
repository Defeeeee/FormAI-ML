import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np
import requests
import os

mp_pose = mp.solutions.pose

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def classify_plank(image_url, model_path=os.path.join(root, 'Models/Core/Plank/model.pth')):
    """
    Classifies a plank pose from an image URL.

    Args:
      image_url: The URL of the image containing the plank pose.
      model_path: The path to the trained PyTorch model file.
                   Defaults to 'model.pth' in the current directory.

    Returns:
      A string indicating the classification of the plank pose:
        - "high back"
        - "low back"
        - "correct"
        - An error message if the image cannot be loaded, landmarks cannot be
          detected, or an error occurs during processing.
    """

    # Load the trained model
    try:
        model = torch.load(model_path)
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        return "Error: Model file not found."
    except Exception as e:
        return f"Error loading model: {e}"

    # Load the image from the URL
    try:
        response = requests.get(image_url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        return f"Error loading image from URL: {e}"

    # Extract landmarks using MediaPipe
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = [[landmark.x, landmark.y, landmark.visibility]
                               for landmark in landmarks]
        except:
            return "Error: No landmarks detected."

    # Preprocess landmarks
    selected_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    try:
        preprocessed_landmarks = []
        for i in selected_indices:
            landmark = frame_landmarks[i]
            preprocessed_landmarks.extend([landmark[0], landmark[1]])

        # Convert landmarks to PyTorch tensor
        input_tensor = torch.FloatTensor([preprocessed_landmarks])

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output.data, 1)
            prediction_class = prediction.item()

        # Map prediction to class labels
        if prediction_class == 2:
            return "high back"
        elif prediction_class == 1:
            return "low back"
        elif prediction_class == 0:
            return "correct"
        else:
            return "Error: Invalid prediction class."
    except Exception as e:
        return f"Error processing landmarks: {e}"


if __name__ == "__main__":
    image_url = 'https://static-ssl.businessinsider.com/image/5a675c71a24444ff3b8b5097-2000/max%20lowery%20-%20tom%20joy%20%200r1a8427%20.jp2'
    result = classify_plank(image_url)
    print(result)  # Output the classification result