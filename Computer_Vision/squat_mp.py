import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def extract_squat_features(image_path):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image = cv2.imread(image_path)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark

        # Feature engineering (Squat-specific features)
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
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),  # Adjust landmarks as needed
            # Add more features as needed (e.g., knee valgus, ankle dorsiflexion)
        }
        return features


def process_images_from_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    features_list = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        features = extract_squat_features(image_path)
        if features:
            features['filename'] = row['filename']  # Keep the filename
            features_list.append(features)

    features_df = pd.DataFrame(features_list)
    final_df = pd.merge(df, features_df, on='filename', how='left')
    final_df.to_csv('squat_features_with_labels_TEST.csv', index=False)
    print("Features saved to squat_features_with_labels.csv")


if __name__ == "__main__":
    csv_path = '/Users/defeee/Downloads/Squat_Classification.v4-test.multiclass/test/_classes.csv'
    image_folder = '/Users/defeee/Downloads/Squat_Classification.v4-test.multiclass/test'
    process_images_from_csv(csv_path, image_folder)