

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

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

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

# Example Usage (make sure to replace '1' with your actual video URL or file path)
video_url = '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (1).mp4'  # Replace with your video
pose_features = extract_features_from_video(video_url)


if pose_features.size == 0:
    print("Error: No frames were processed. Check camera access and functionality.")
    exit()

# Calculate hip midpoint
hip_midpoint = (pose_features[:, 23*3:23*3+2] + pose_features[:, 24*3:24*3+2]) / 2

# Assuming left leg is more visible in the side view
hip_angles = []
left_knee_angles = []
for frame_counter in range(len(pose_features)):
    hip = hip_midpoint[frame_counter]
    left_knee = pose_features[frame_counter, 25*3:25*3+2]
    left_ankle = pose_features[frame_counter, 27*3:27*3+2]

    hip_angle = calculate_angle(left_knee, hip, left_ankle)
    knee_angle = calculate_angle(hip, left_knee, left_ankle)

    hip_angles.append(hip_angle)
    left_knee_angles.append(knee_angle)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hip_angles, marker='o', linestyle='-')
plt.title('Hip Angle During Squat (Side View)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(left_knee_angles, marker='o', linestyle='-')
plt.title('Left Knee Angle During Squat (Side View)')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.grid(True)

plt.tight_layout()
plt.show()

