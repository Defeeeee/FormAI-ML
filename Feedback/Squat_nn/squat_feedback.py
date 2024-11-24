import cv2
import mediapipe as mp
import numpy as np
import requests
from tensorflow import keras
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load your trained Keras model
model = keras.models.load_model(ROOT + '/Models/Core/Squat/squat_model_tf2.h5')

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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        print(f"Error downloading video from {video_url}: {e}")
        return None

def analyze_squat_video(video_url):
    """
    Analyzes a squat video, extracts keyframes (1 in 10 frames),
    predicts overall quality, and returns a detailed result dictionary.
    """

    # Comment out the download logic if you're providing the video_path directly
    video_path = download_video(video_url)
    if video_path is None:
        return {"success": False, "message": "Error downloading video."}

    cap = cv2.VideoCapture(video_path)
    keyframes = []
    all_predictions = []
    keyframes_results = []
    confidences = []

    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process only every 10th frame

        if frame_count % 3 == 0 or frame_count < 3:
            angles = get_pose_angles(frame)
            if angles:
                if angles['left_knee_angle'] > 140.0 or angles['right_knee_angle'] > 140.0:
                    continue

                # Prepare features for the model
                features = np.array([
                    angles['left_knee_angle'],
                    angles['right_knee_angle'],
                    angles['left_hip_angle'],
                    angles['right_hip_angle'],
                    angles['back_angle']
                ]).reshape(1, -1)

                # Reshape for the CNN model (if your model requires it)
                features = features.reshape(1, 5, 1)

                # Make prediction
                prediction = model.predict(features)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                all_predictions.append(predicted_class)
                confidences.append(confidence)

                keyframes_results.append(not(bool(predicted_class)))

                # Add the current frame as a keyframe
                keyframes.append(frame)

    cap.release()
    os.remove(video_path)  # Comment this out if you don't want to delete the video file

    # Determine overall squat quality
    if len(all_predictions) > 0:
        # if 10% are good, then it is good
        majority_prediction = max(set(all_predictions), key=all_predictions.count)
        print(all_predictions)
        overall_quality = "good" if majority_prediction == 0 else "bad"
        average_confidence = sum(confidences) / len(confidences)
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

if __name__ == "__main__":
    video_url = "https://s36370.pcdn.co/wp-content/uploads/2016/07/BW-Squat-Finish-Side-View.jpg"
    result = analyze_squat_video(video_url)
    print(result)