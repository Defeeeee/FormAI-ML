import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('squat_model_tf.h5')

mp_pose = mp.solutions.pose


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
        input_tensor = np.array(list(features.values())).reshape(1, -1)  # Reshape for TensorFlow
        output = model.predict(input_tensor)[0][0]  # Get the prediction
        probability = output  # For TensorFlow, the output is already a probability
        predicted_class = "Good Squat" if probability > 0.5 else "Bad Squat"
        return predicted_class, probability
    else:
        return "Error: Could not extract features.", None


# Example usage
image_path = '/Users/defeee/Downloads/images (7).jpeg'
predicted_class, probability = predict_squat(image_path)
print(f"Predicted class: {predicted_class}")
if probability is not None:
    print(f"Probability: {probability:.4f}")
