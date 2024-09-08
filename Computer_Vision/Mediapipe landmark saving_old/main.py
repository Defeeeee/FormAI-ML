import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(all_features)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Function to extract data from a single frame
def extract_data_from_frame(landmarks):
    # Calculate angles
    left_knee_angle = calculate_angle(landmarks[LEFT_HIP * 3:LEFT_HIP * 3 + 3],
                                      landmarks[LEFT_KNEE * 3:LEFT_KNEE * 3 + 3],
                                      landmarks[LEFT_ANKLE * 3:LEFT_ANKLE * 3 + 3])
    right_knee_angle = calculate_angle(landmarks[RIGHT_HIP * 3:RIGHT_HIP * 3 + 3],
                                       landmarks[RIGHT_KNEE * 3:RIGHT_KNEE * 3 + 3],
                                       landmarks[RIGHT_ANKLE * 3:RIGHT_ANKLE * 3 + 3])

    left_hip_angle = calculate_angle(landmarks[LEFT_SHOULDER * 3:LEFT_SHOULDER * 3 + 3],
                                     landmarks[LEFT_HIP * 3:LEFT_HIP * 3 + 3],
                                     landmarks[LEFT_KNEE * 3:LEFT_KNEE * 3 + 3])
    right_hip_angle = calculate_angle(landmarks[RIGHT_SHOULDER * 3:RIGHT_SHOULDER * 3 + 3],
                                      landmarks[RIGHT_HIP * 3:RIGHT_HIP * 3 + 3],
                                      landmarks[RIGHT_KNEE * 3:RIGHT_KNEE * 3 + 3])

    # Calculate relative positions
    left_knee_position = landmarks[LEFT_KNEE * 3] - landmarks[LEFT_ANKLE * 3]
    right_knee_position = landmarks[RIGHT_KNEE * 3] - landmarks[RIGHT_ANKLE * 3]

    # Assuming frame rate is 30 FPS, calculate time in seconds
    time = len(data) / 30

    return [time, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
            left_knee_position, right_knee_position]

# Example Usage
video_url = '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (1).mp4'
pose_features = extract_features_from_video(video_url)

if pose_features.size == 0:
    print("Error: No frames were processed. Check camera access and functionality.")
    exit()

# Extract data for all frames
data = []
for frame_landmarks in pose_features:
    data.append(extract_data_from_frame(frame_landmarks))

# Create DataFrame
df = pd.DataFrame(data, columns=["Time", "Left Knee Angle", "Right Knee Angle",
                                 "Left Hip Angle", "Right Hip Angle",
                                 "Left Knee Position", "Right Knee Position"])

# Identify transitions between up and down phases
df['Knee Angle Change'] = df['Left Knee Angle'].diff()
df['Phase'] = 'Transition'
df.loc[df['Knee Angle Change'] < 0, 'Phase'] = 'Down'
df.loc[df['Knee Angle Change'] > 0, 'Phase'] = 'Up'

# Save to CSV
csv_file_path = 'data.csv'
df.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")
