import csv
import cv2
import mediapipe as mp
import numpy as np
import requests

mp_pose = mp.solutions.pose

def load_video_from_url(url):
    """
    Loads a video from a URL using requests.

    Args:
        url: The URL of the video.

    Returns:
        The loaded video, or None if loading fails.
    """
    try:
        response = requests.get(url, stream=True).raw
        video = np.asarray(bytearray(response.read()), dtype="uint8")
        return video
    except Exception as e:
        print(f"Error loading video from URL: {e}")
        return None

def extract_landmarks_from_frame(frame):
    """
    Extracts landmarks from a video frame using MediaPipe Pose.

    Args:
        frame: The video frame to process.

    Returns:
        A list of landmark sequences, where each sequence corresponds to a frame
        and contains the x, y coordinates of visible landmarks.
    """
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        # Recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Make detection
        results = pose.process(frame)

        # Recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []
            for landmark in landmarks:
                frame_landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return [frame_landmarks]  # Returning as a list of list to maintain consistency
        except:
            pass

    return []  # Return empty list if no landmarks are detected

def preprocess_landmarks(landmarks_list, exercise_type="squat"):
    """
    Preprocesses the extracted landmarks with more robust occlusion handling.

    Args:
        landmarks_list: List of landmark sequences.
        exercise_type: Type of exercise ("plank", "squat", etc.).

    Returns:
        Preprocessed landmarks.
    """
    if exercise_type == "squat":
        selected_indices = [
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        ]

    normalized_landmarks = []
    for frame_landmarks in landmarks_list:
        temp = []
        for i in selected_indices:
            landmark = frame_landmarks[i]

            # 1. Visibility Threshold (as before):
            if landmark[2] >= 0.5:
                temp.append([landmark[0], landmark[1]])
            else:
                # 2. Simple Interpolation:
                if i > 0 and not np.isnan(temp[-1][0]):  # If previous landmark is visible
                    prev_landmark = frame_landmarks[i - 1]
                    temp.append([prev_landmark[0], prev_landmark[1]])
                else:
                    # 3. Symmetric Landmark Estimation:
                    if 'LEFT' in mp_pose.PoseLandmark(i).name:
                        symmetric_index = mp_pose.PoseLandmark(i + 1).value
                    else:
                        symmetric_index = mp_pose.PoseLandmark(i - 1).value
                    symmetric_landmark = frame_landmarks[symmetric_index]
                    if symmetric_landmark[2] >= 0.5:  # If the symmetric landmark is visible
                        # Estimate based on symmetry
                        temp.append([1 - symmetric_landmark[0], symmetric_landmark[1]])
                    else:
                        # 4. Default value
                        temp.append([0.5, 0.5])

        normalized_landmarks.append(temp)

    return normalized_landmarks

def label_squat(frame, landmarks):
    """
    Displays the frame with landmarks and prompts the user to label the squat.

    Args:
        frame: The frame to display.
        landmarks: The landmarks detected in the frame.

    Returns:
        The label (0, 1, or 2) assigned by the user.
    """
    # Draw landmarks on the frame
    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame and wait for user input
    cv2.imshow('Label Squat', frame)
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == ord('0'):
            label = 0  # Correct
            break
        elif key == ord('1'):
            label = 1  # Low squat
            break
        elif key == ord('2'):
            label = 2  # High squat
            break
        elif key == 27:  # Esc key
            label = -1  # Exit labeling (optional)
            break

    cv2.destroyAllWindows()
    return label

def process_video(video_url):
    """
    Processes a video, extracts landmarks, labels squats, and saves data to CSV.

    Args:
        video_url: The URL of the video.
    """
    data = []  # List to store video data and labels

    cap = cv2.VideoCapture(video_url)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract and preprocess landmarks
            try:
                landmarks_list = extract_landmarks_from_frame(frame)
                if landmarks_list:
                    preprocessed_landmarks = preprocess_landmarks(landmarks_list, exercise_type="squat")[0]

                    # Label the squat
                    label = label_squat(frame.copy(), pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks)
                    if label == -1:  # Exit if Esc key is pressed
                        break

                    # Store the data
                    data.append(preprocessed_landmarks + [label])
                else:
                    print(f"No landmarks detected for frame")

            except Exception as e:
                print(f"Error processing landmarks for frame: {e}")

    cap.release()

    # Save data to CSV
    with open('squat_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y', 'label'])

        for row in data:
            flattened_row = []  # Flatten the list of lists
            for landmark in row[:-1]:  # Iterate through landmarks (exclude the label)
                flattened_row.extend(landmark)  # Add the x, y coordinates to the flattened list
            flattened_row.append(row[-1])  # Add the label
            writer.writerow(flattened_row)  # Write the flattened row to the CSV

if __name__ == "__main__":
    video_url = 'https://example.com/path/to/your/video.mp4'
    process_video(video_url)
