import warnings

# Suppress Protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

def extract_features_from_video(video_url):
    """Retrieves video from URL and extracts pose features using MediaPipe."""
    cap = cv2.VideoCapture(video_url)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    all_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            landmarks = landmarks.flatten()
            all_features.append(landmarks)

    cap.release()
    return np.array(all_features)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import warnings

# Suppress Protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# 1. Data Preparation

training_video_url = '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (1).mp4'
pose_features = extract_features_from_video(training_video_url)

if pose_features.size == 0:
    print("Error: No frames were processed from the training video. Check the file path.")
    exit()

hip_midpoint = (pose_features[:, 23*3:23*3+2] + pose_features[:, 24*3:24*3+2]) / 2

hip_angles = []
left_knee_angles = []
trunk_angles = []
for frame_counter in range(len(pose_features)):
    hip = hip_midpoint[frame_counter]
    left_knee = pose_features[frame_counter, 25*3:25*3+2]
    left_ankle = pose_features[frame_counter, 27*3:27*3+2]
    left_shoulder = pose_features[frame_counter, 11*3:11*3+2]
    right_shoulder = pose_features[frame_counter, 12*3:12*3+2]
    left_hip = pose_features[frame_counter, 23*3:23*3+2]
    right_hip = pose_features[frame_counter, 24*3:24*3+2]

    hip_angle = calculate_angle(left_knee, hip, left_ankle)
    knee_angle = calculate_angle(hip, left_knee, left_ankle)
    trunk_angle = calculate_angle(left_hip, (left_shoulder + right_shoulder) / 2, right_hip)

    hip_angles.append(hip_angle)
    left_knee_angles.append(knee_angle)
    trunk_angles.append(trunk_angle)

# Create DataFrame with column names
data = {
    'hip_angle': hip_angles,
    'knee_angle': left_knee_angles,
    'trunk_angle': trunk_angles,
}
df = pd.DataFrame(data)

# (Optional) Data Augmentation
# ... (Implement data augmentation techniques here if needed)

# 2. Model Training

X = df[['hip_angle', 'knee_angle', 'trunk_angle']]
y = df.index

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', max_iter=500)
model.fit(X_train, y_train)

# 3. Real-time Pose Estimation and Feedback

cap = cv2.VideoCapture(training_video_url)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Adjusted thresholds (experiment with these values)
hip_angle_threshold = 15
knee_angle_threshold = 15
trunk_angle_threshold = 7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Extract landmarks and calculate angles
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        landmarks = landmarks.flatten()

        hip = (landmarks[23*3:23*3+2] + landmarks[24*3:24*3+2]) / 2
        left_knee = landmarks[25*3:25*3+2]
        left_ankle = landmarks[27*3:27*3+2]
        left_shoulder = landmarks[11*3:11*3+2]
        right_shoulder = landmarks[12*3:12*3+2]
        left_hip = landmarks[23*3:23*3+2]
        right_hip = landmarks[24*3:24*3+2]

        hip_angle = calculate_angle(left_knee, hip, left_ankle)
        knee_angle = calculate_angle(hip, left_knee, left_ankle)
        trunk_angle = calculate_angle(left_hip, (left_shoulder + right_shoulder) / 2, right_hip)

        # Print extracted angles for debugging
        print(f"Hip Angle: {hip_angle}, Knee Angle: {knee_angle}, Trunk Angle: {trunk_angle}")

        # Make predictions
        features_df = pd.DataFrame({'hip_angle': [hip_angle], 'knee_angle': [knee_angle], 'trunk_angle': [trunk_angle]})
        predicted_frame = model.predict(scaler.transform(features_df))[0]

        # Find the closest frame in the training data
        closest_frame = np.argmin(np.abs(y_train - predicted_frame))

        # Provide feedback based on the closest frame's angles
        ideal_hip_angle = X_train[closest_frame][0]
        ideal_knee_angle = X_train[closest_frame][1]
        ideal_trunk_angle = X_train[closest_frame][2]

        feedback = ""
        if abs(hip_angle - ideal_hip_angle) > hip_angle_threshold:
            feedback += "Adjust your hip angle. "
        if abs(knee_angle - ideal_knee_angle) > knee_angle_threshold:
            feedback += "Adjust your knee angle. "
        if abs(trunk_angle - ideal_trunk_angle) > trunk_angle_threshold:
            feedback += "Keep your back straight. "
        if not feedback:
            feedback = "Good form!"

        # Display feedback on the frame
        cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()