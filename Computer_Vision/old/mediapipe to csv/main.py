import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# CSV file path
csv_file_path = 'exercise_data.csv'

# Columns for the DataFrame (adjusted for 33 landmarks)
columns = [
    'NOSE_x', 'NOSE_y', 'NOSE_z', 'NOSE_v',
    'LEFT_EYE_INNER_x', 'LEFT_EYE_INNER_y', 'LEFT_EYE_INNER_z', 'LEFT_EYE_INNER_v',
    'LEFT_EYE_x', 'LEFT_EYE_y', 'LEFT_EYE_z', 'LEFT_EYE_v',
    'LEFT_EYE_OUTER_x', 'LEFT_EYE_OUTER_y', 'LEFT_EYE_OUTER_z', 'LEFT_EYE_OUTER_v',
    'RIGHT_EYE_INNER_x', 'RIGHT_EYE_INNER_y', 'RIGHT_EYE_INNER_z', 'RIGHT_EYE_INNER_v',
    'RIGHT_EYE_x', 'RIGHT_EYE_y', 'RIGHT_EYE_z', 'RIGHT_EYE_v',
    'RIGHT_EYE_OUTER_x', 'RIGHT_EYE_OUTER_y', 'RIGHT_EYE_OUTER_z', 'RIGHT_EYE_OUTER_v',
    'LEFT_EAR_x', 'LEFT_EAR_y', 'LEFT_EAR_z', 'LEFT_EAR_v',
    'RIGHT_EAR_x', 'RIGHT_EAR_y', 'RIGHT_EAR_z', 'RIGHT_EAR_v',
    'MOUTH_LEFT_x', 'MOUTH_LEFT_y', 'MOUTH_LEFT_z', 'MOUTH_LEFT_v',
    'MOUTH_RIGHT_x', 'MOUTH_RIGHT_y', 'MOUTH_RIGHT_z', 'MOUTH_RIGHT_v',
    'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z', 'LEFT_SHOULDER_v',
    'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_z', 'RIGHT_SHOULDER_v',
    'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_z', 'LEFT_ELBOW_v',
    'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'RIGHT_ELBOW_z', 'RIGHT_ELBOW_v',
    'LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_z', 'LEFT_WRIST_v',
    'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'RIGHT_WRIST_z', 'RIGHT_WRIST_v',
    'LEFT_PINKY_x', 'LEFT_PINKY_y', 'LEFT_PINKY_z', 'LEFT_PINKY_v',
    'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'RIGHT_PINKY_z', 'RIGHT_PINKY_v',
    'LEFT_INDEX_x', 'LEFT_INDEX_y', 'LEFT_INDEX_z', 'LEFT_INDEX_v',
    'RIGHT_INDEX_x', 'RIGHT_INDEX_y', 'RIGHT_INDEX_z', 'RIGHT_INDEX_v',
    'LEFT_THUMB_x', 'LEFT_THUMB_y', 'LEFT_THUMB_z', 'LEFT_THUMB_v',
    'RIGHT_THUMB_x', 'RIGHT_THUMB_y', 'RIGHT_THUMB_z', 'RIGHT_THUMB_v',
    'LEFT_HIP_x', 'LEFT_HIP_y', 'LEFT_HIP_z', 'LEFT_HIP_v',
    'RIGHT_HIP_x', 'RIGHT_HIP_y', 'RIGHT_HIP_z', 'RIGHT_HIP_v',
    'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_z', 'LEFT_KNEE_v',
    'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'RIGHT_KNEE_z', 'RIGHT_KNEE_v',
    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z', 'LEFT_ANKLE_v',
    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z', 'RIGHT_ANKLE_v',
    'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_z', 'LEFT_HEEL_v',
    'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_z', 'RIGHT_HEEL_v',
    'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_z', 'LEFT_FOOT_INDEX_v',
    'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_z', 'RIGHT_FOOT_INDEX_v'
]


def save_pose_data(image_paths):
    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Load the existing CSV file
        df = pd.read_csv(csv_file_path)
    else:
        # Create a new DataFrame
        df = pd.DataFrame(columns=columns)

    # Initialize the label column
    df['label'] = ''

    # Iterate through the image paths
    for image_path in image_paths:
        # Try to download the image from the URL
        try:
            resp = urllib.request.urlopen(image_path)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except urllib.error.URLError as e:
            print(f"Error downloading image from {image_path}: {e}")
            continue  # Skip to the next image if there's an error

        # Check if the image was loaded successfully
        if image is None:
            print(f"Error loading image from {image_path}")
            continue

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = pose.process(image_rgb)

        # Check if the pose was detected
        if results.pose_landmarks is not None:
            # Initialize the row
            row = []

            # Iterate through the landmarks
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark coordinates to the row
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
                row.append(landmark.visibility)

            # Check if we have the expected number of landmark values
            expected_num_values = len(columns)
            if len(row) != expected_num_values:
                # If not, try to copy values from the visible side (for plank images only)
                if len(row) == expected_num_values // 2:
                    # Logic to mirror the landmarks assuming symmetry in plank pose
                    # This will depend on how you define 'left' and 'right' landmarks in your `columns`
                    # You'll likely need to adjust the indexing and mirroring logic accordingly
                    mirrored_row = []
                    for i in range(0, len(row), 4):  # Iterate over each landmark (x, y, z, v)
                        x, y, z, v = row[i:i + 4]
                        if columns[i].startswith('LEFT'):
                            mirrored_x = 1 - x  # Mirror x-coordinate
                            mirrored_row.extend([mirrored_x, y, z, v])
                        else:
                            mirrored_row.extend([x, y, z, v])
                row = mirrored_row + row  # Combine mirrored and original landmarks
            else:
                print(
                    f"Skipping image {image_path}: Unexpected number of landmark values ({len(row)} instead of {expected_num_values})")
                continue

        # Append the row to the DataFrame
        df.loc[len(df)] = row

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Annotated Image', annotated_image)

    # wait for cv2 to get a key press
    while True:
        key = cv2.waitKey(0)
        if key == ord('c'):
            df.loc[df.index[-1], 'label'] = 'C'
            break
        elif key == ord('h'):
            df.loc[df.index[-1], 'label'] = 'H'
            break
        elif key == ord('l'):
            df.loc[df.index[-1], 'label'] = 'L'
            break
        elif key == ord('q'):
            break
        else:
            print('Invalid key. Press c for correct, h for high back, l for low back, or q to quit')

    else:
        print('Pose not detected')

# Save the DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False)

# Close the OpenCV windows
    cv2.destroyAllWindows()

    print(f'Pose data saved to {csv_file_path}')

image_paths = ['https://hips.hearstapps.com/hmg-prod/images/hdm119918mh15842-1545237096.png',
               'https://www.shape.com/thmb/T2GyvzFah3XYR8_L8W16ANWBTXs=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/low-plank-hold-b8a63da1ef844f00b6f6a21141ba1d87.jpg',
               'https://media.self.com/photos/615378c1150f8742ac896749/16:9/w_4991,h_2807,c_limit/Forearm%20Plank%20-%20Delise_001.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKschFmUKMdDGShQIwpQMcM_WeGHbKEZoUPQ&s',
               'https://t4.ftcdn.net/jpg/08/41/38/01/360_F_841380116_U81vXZJRojFn2BuAav7MLUafRgMLLjym.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnQCwpGPHT7oShh1A5rx2FXOHVlfq4DrXjEA&s',
               'https://embed-ssl.wistia.com/deliveries/edd92d0077218d633afa90bbe3d8652ce9e0b400.webp?image_crop_resized=1280x720',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQaNXmp3hOmZenCHAfc8VKs9bzJ63X2p9ZdWL9ZjYueVDOT880satIuLq0OyvRjnotAykE&usqp=CAU',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYOr6zfQgHEaY_rse8U-ICPMiU4yWvwNhS9Vn7Uy0WJYcHjvUZTwTlbC-OKykKHWZbP08&usqp=CAU',
               'https://cdn.shopify.com/s/files/1/0264/4885/5122/t/32/assets/pf-e1bfb663--Screenshot-20210527-at-091452min.png?v=1622712364',
               'https://www.shape.com/thmb/T2GyvzFah3XYR8_L8W16ANWBTXs=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/low-plank-hold-b8a63da1ef844f00b6f6a21141ba1d87.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLdNeW_0tzVU2bcRCpWQCGpAu5RxIK3Uw-Vw&s',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT__rzeah4I--qBl6HuqWTkUH0aaRCwkb3d6A&s',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNiS0Kkbztj6uW92ymJFogPUr8WvttF1YSjw&s',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMAwJcqIfmCGNcZnGxzXBwr_a6OpD05-a_ymqXjGXn4yXutRGQRMQEk0Cuh63TlBchVgs&usqp=CAU',
               'https://blog.nasm.org/hs-fs/hubfs/standard-plank.jpg?width=1000&name=standard-plank.jpg',
               'https://media.post.rvohealth.io/wp-content/uploads/sites/2/2019/05/Plank-1.png',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMvlmSgWN5PqkZozzFIm0aV4LqlCRwToUKag&s',
               'https://www.peacefuldumpling.com/wp-content/uploads/2019/05/modified_plank_stepouts.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStgb3c_6I-Eydl4oie5C35PdMoVqaUu_guNw&s',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5WSj0Lmhs9E3WEJuGHv4dDdOXnop9cFRhKw&s',
               'https://i.ytimg.com/vi/ql8qf61rCDo/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCwpjwk1fDPzwGTyH9-8-rtpBDMyQ',
               'https://i.ytimg.com/vi/ZoZV-0fiX5U/maxresdefault.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7TC9QxI4_dsQK1qpDo8VHlsI2rCHlgi3GXQ&s',
               'https://www.stretching-exercises-guide.com/images/repeated_extension_in_lying.jpg',
               'https://www.wikihow.com/images/thumb/6/68/Do-a-Lower-Back-Stretch-Safely-Step-17-preview.jpg/550px-nowatermark-Do-a-Lower-Back-Stretch-Safely-Step-17-preview.jpg',
               'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6d6nMYnFENB7YT8fsG9tU4Gjm0fVv5B7qrg&s'
               ]

# Save the pose data
save_pose_data(image_paths)
