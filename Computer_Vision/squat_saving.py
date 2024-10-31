import csv
import math

import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture the video
cap = cv2.VideoCapture(
    '/Users/defeee/Downloads/stock-footage-online-workout-service-professional-trainer-explaining-exercise-virtual-video-tutorial-for (2).mp4')

# Open CSV file
with open('squat_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write CSV header
    writer.writerow(['nose_x', 'nose_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
                     'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y',
                     'right_knee_x', 'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y'])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Pose
        results = pose.process(image_rgb)

        # Draw the landmarks on the image
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get the relevant landmarks
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Write data to CSV
            writer.writerow([nose.x, nose.y, left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y,
                             left_hip.x, left_hip.y, right_hip.x, right_hip.y, left_knee.x, left_knee.y,
                             right_knee.x, right_knee.y, left_ankle.x, left_ankle.y, right_ankle.x, right_ankle.y])

        # Show the image
        cv2.imshow("Squat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
