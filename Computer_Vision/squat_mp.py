import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load your trained Keras model
model = keras.models.load_model('/Users/defeee/Documents/GitHub/FormAI-ML/Models/Core/Squat/squat_model_tf2.h5')  # Replace with your model path

# Initialize Mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Feature scaler (use the same scaler used during training)
scaler = MinMaxScaler()  # You might need to load the scaler if you saved it separately

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

def calculate_angle(a, b, c):
    """Calculates the angle between three points in 3D space."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_squat_video(video_path):
    """
    Analyzes a squat video, extracts keyframes, and predicts overall quality.

    Args:
      video_path: Path to the squat video file.

    Returns:
      A list of keyframe images and the overall prediction ("good" or "bad").
    """

    cap = cv2.VideoCapture(video_path)
    keyframes = []
    all_predictions = []

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

            # Normalize features
            features = scaler.transform(features)

            # Reshape for the CNN model
            features = features.reshape(1, 5, 1)

            # Make prediction
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction)
            all_predictions.append(predicted_class)

            # Simple keyframe extraction (replace with more robust logic if needed)
            if predicted_class == 1:  # Assuming 1 is the "good" class
                keyframes.append(frame)

    cap.release()

    # Determine overall squat quality
    if len(all_predictions) > 0:
        majority_prediction = max(set(all_predictions), key=all_predictions.count)
        overall_quality = "good" if majority_prediction == 1 else "bad"
    else:
        overall_quality = "unknown"

    return keyframes, overall_quality

# Example usage:
video_path = 'path/to/your/squat_video.mp4'  # Replace with your video path
keyframes, overall_quality = analyze_squat_video(video_path)

# Display keyframes
for i, keyframe in enumerate(keyframes):
    cv2.imshow(f'Keyframe {i+1}', keyframe)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Overall Squat Quality:", overall_quality)