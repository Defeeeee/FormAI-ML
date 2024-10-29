import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np
import os
import io

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

def classify_plank_live(model_path=os.path.join(root, 'Models/Core/Plank/model.pth')):
    """
    Classifies plank poses from a live camera feed and displays the
    predictions with confidence on the video stream.

    Args:
      model_path: The path to the trained PyTorch model file.
                   Defaults to 'model.pth' in the current directory.
    """

    # Load the trained model
    try:
        model = Net()
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer))
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print("Error: Model file not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the camera
    cap = cv2.VideoCapture(2)  # Use default camera (index 0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                frame_landmarks = [[landmark.x, landmark.y, landmark.visibility]
                                   for landmark in landmarks]
            except:
                continue  # Skip frames without landmarks

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
                    probabilities = F.softmax(output, dim=1)  # Get probabilities
                    confidence, prediction = torch.max(probabilities, 1)
                    prediction_class = prediction.item()
                    confidence_value = confidence.item() * 100

                # Map prediction to class labels
                class_labels = ["correct", "low back", "high back"]
                predicted_label = class_labels[prediction_class]

                # Draw landmarks and prediction on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image,
                            f"{predicted_label} ({confidence_value:.2f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)

            except Exception as e:
                print(f"Error processing landmarks: {e}")

            cv2.imshow('Live Plank Classification', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_plank_live()
