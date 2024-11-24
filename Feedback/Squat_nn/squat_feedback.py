import cv2
import mediapipe as mp
import numpy as np
import requests
from tensorflow import keras
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load your trained Keras model
model = keras.models.load_model(ROOT + '/Models/Core/Squat/squat_model_tf2.h5')  # Replace with your model path

# Initialize Mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

def get_pose_angles(frame):
    """
    Detects pose landmarks in a frame using Mediapipe and calculates angles.

    Args:
      frame: An image frame.

    Returns:
      A dictionary of angles.
    """
    # Convert BGR frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Calculate angles
            angles = {
                'left_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
                'right_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
                'left_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
                'right_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
                'back_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
            }
            return angles
        else:
            return None

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

def analyze_squat_video(video_url):
    """
    Analyzes a squat video, extracts keyframes, predicts overall quality,
    and returns a detailed result dictionary.

    Args:
      video_path: Path to the squat video file.

    Returns:
      A dictionary containing analysis results.
    """

    video_path = download_video(video_url)

    cap = cv2.VideoCapture(video_path)
    keyframes = []
    all_predictions = []
    keyframes_results = []  # Store individual keyframe predictions
    confidences = []  # Store prediction confidences

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        angles = get_pose_angles(frame)
        if angles:
            # Prepare features for the model
            features = np.array([
                angles['left_knee_angle'],
                angles['right_knee_angle'],
                angles['left_hip_angle'],
                angles['right_hip_angle'],
                angles['back_angle']
            ]).reshape(1, -1)

            # Reshape for the CNN model
            features = features.reshape(1, 5, 1)

            # Make prediction
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]  # Confidence of the predicted class
            all_predictions.append(predicted_class)
            confidences.append(confidence)

            keyframes_results.append(not(bool(predicted_class)))  # Store True/False for good/bad

            if predicted_class == 0:
                keyframes.append(frame)

    cap.release()

    # Determine overall squat quality
    if len(all_predictions) > 0:
        majority_prediction = max(set(all_predictions), key=all_predictions.count)
        overall_quality = "good" if majority_prediction == 0 else "bad"
        average_confidence = np.mean(confidences)  # Calculate average confidence
    else:
        overall_quality = "unknown"
        average_confidence = 0

    # Construct the result dictionary
    result = {
        "success": True,
        "correct": overall_quality == "good",
        "message": "Workout looks good!" if overall_quality == "good" else "Incorrect squat detected!",
        "prediction_confidence": average_confidence,
        "keyframes_results": keyframes_results
    }

    return result
