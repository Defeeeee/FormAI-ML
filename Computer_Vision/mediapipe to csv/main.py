import os
import cv2
import pandas as pd
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# CSV file path
csv_file_path = 'exercise_data.csv'

# Check if the CSV file exists, if not, create it with headers
if not os.path.exists(csv_file_path):
    columns = ['nose_x', 'nose_y', 'nose_z', 'nose_v', 'left_shoulder_x',
               'left_shoulder_y', 'left_shoulder_z', 'left_shoulder_v',
               'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
               'right_shoulder_v', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
               'left_elbow_v', 'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
               'right_elbow_v', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
               'left_wrist_v', 'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
               'right_wrist_v', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_hip_v',
               'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_v',
               'left_knee_x', 'left_knee_y', 'left_knee_z', 'left_knee_v',
               'right_knee_x', 'right_knee_y', 'right_knee_z', 'right_knee_v',
               'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_ankle_v',
               'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'right_ankle_v',
               'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_heel_v',
               'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_heel_v',
               'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z',
               'left_foot_index_v', 'right_foot_index_x', 'right_foot_index_y',
               'right_foot_index_z', 'right_foot_index_v', 'label']
    pd.DataFrame(columns=columns).to_csv(csv_file_path, index=False)

# Directory containing images
image_dir = '/Users/defeee/Downloads/archive/plank'

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Visualize landmarks on the image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # Display the image and wait for a key press
            cv2.imshow(f'Label for {filename}', annotated_image)
            print("Press 'h' for High back, 'l' for Low back, 'c' for Correct, or 'q' to quit")

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
                elif key in [ord('h'), ord('l'), ord('c')]:
                    label = chr(key).upper()
                    break
                else:
                    print("Invalid key. Please press 'h', 'l', 'c', or 'q'")

            cv2.destroyAllWindows()

            # Extract landmarks and format data
            landmarks = results.pose_landmarks.landmark
            row = []
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            row.append(label)

            # Append row to the CSV file
            pd.DataFrame([row], columns=columns).to_csv(csv_file_path, mode='a', header=False, index=False)

            print(f'Saved landmarks for {image_path} with label {label} to CSV')