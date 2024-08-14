import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def extract_features_from_video(video_url):
    """Retrieves video from URL and extracts pose features using MediaPipe.

        Args:
            video_url: The URL of the video to process.

        Returns:
            A NumPy array containing the extracted pose features for each frame.
        """
    cap = cv2.VideoCapture(video_url)  # Open video from URL

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    all_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # MediaPipe Pose processing
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # Extract and normalize pose landmarks (33 keypoints)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            landmarks = landmarks.flatten()  # Flatten for easier storage

            # Further feature engineering (optional)
            # E.g., calculate joint angles, distances, etc.

            all_features.append(landmarks)

        # Draw landmarks and display AFTER processing (inside the loop)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the processed frame
        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release video resources after the loop
    cap.release()
    cv2.destroyAllWindows()  # Close the display window
    return np.array(all_features)  # Convert to NumPy array


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Example Usage (make sure to replace '1' with your actual video URL or file path)
video_url = '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (1).mp4'  # Replace with your video
pose_features = extract_features_from_video(video_url)

if pose_features.size == 0:
    print("Error: No frames were processed. Check camera access and functionality.")
    exit()

# Assuming pose_features is a NumPy array with shape (num_frames, 99) from your code

# MediaPipe landmark indices for key points (refer to MediaPipe documentation for full list)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_EAR = 7
RIGHT_EAR = 8


# Function to extract data for a single frame
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

    # You can similarly calculate ankle and trunk angles using appropriate landmarks

    # Calculate relative positions (horizontal distance between knee and ankle)
    left_knee_position = landmarks[LEFT_KNEE * 3] - landmarks[LEFT_ANKLE * 3]
    right_knee_position = landmarks[RIGHT_KNEE * 3] - landmarks[RIGHT_ANKLE * 3]

    # Assuming frame rate is 30 FPS, calculate time in seconds
    time = len(data) / 30

    return [time, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
            left_knee_position, right_knee_position]  # Add more data as needed


# Extract data for all frames
data = []
for frame_landmarks in pose_features:
    data.append(extract_data_from_frame(frame_landmarks))

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=["Time", "Left Knee Angle", "Right Knee Angle",
                                 "Left Hip Angle", "Right Hip Angle",
                                 "Left Knee Position", "Right Knee Position"])  # Add more columns as needed

df.to_csv("squat_data.csv", index=False)
