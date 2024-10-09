import cv2
import mediapipe as mp
import numpy as np
import csv
import requests

mp_pose = mp.solutions.pose


def load_image_from_url(url):
    """
    Loads an image from a URL using requests.

    Args:
        url: The URL of the image.

    Returns:
        The loaded image, or None if loading fails.
    """
    try:
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None


def extract_landmarks(image):
    """
    Extracts landmarks from an image using MediaPipe Pose.

    Args:
        image: The image to process.

    Returns:
        A list of landmark sequences, where each sequence corresponds to a frame
        and contains the x, y coordinates of visible landmarks.
    """

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []
            for landmark in landmarks:
                frame_landmarks.append(
                    [landmark.x, landmark.y, landmark.visibility])
            return [
                frame_landmarks
            ]  # Returning as a list of list to maintain consistency
        except:
            pass

    return []  # Return empty list if no landmarks are detected


def preprocess_landmarks(landmarks_list, exercise_type="plank"):
    """
    Preprocesses the extracted landmarks with more robust occlusion handling.

    Args:
        landmarks_list: List of landmark sequences.
        exercise_type: Type of exercise ("plank", "squat", etc.).

    Returns:
        Preprocessed landmarks.
    """

    if exercise_type == "plank":
        selected_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
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
                if i > 0 and not np.isnan(
                        temp[-1][0]):  # If previous landmark is visible
                    prev_landmark = frame_landmarks[i - 1]
                    temp.append([prev_landmark[0], prev_landmark[1]])
                else:
                    # 3. Symmetric Landmark Estimation:
                    if 'LEFT' in mp_pose.PoseLandmark(i).name:
                        symmetric_index = mp_pose.PoseLandmark(i + 1).value
                    else:
                        symmetric_index = mp_pose.PoseLandmark(i - 1).value
                    symmetric_landmark = frame_landmarks[symmetric_index]
                    if symmetric_landmark[
                            2] >= 0.5:  # If the symmetric landmark is visible
                        # Estimate based on symmetry
                        temp.append([1 - symmetric_landmark[0],
                                     symmetric_landmark[1]])
                    else:
                        # 4. Default value
                        temp.append([0.5, 0.5])

        normalized_landmarks.append(temp)

    return normalized_landmarks


def label_plank(image, landmarks):
    """
    Displays the image with landmarks and prompts the user to label the plank.

    Args:
        image: The image to display.
        landmarks: The landmarks detected in the image.

    Returns:
        The label (0, 1, or 2) assigned by the user.
    """

    # Draw landmarks on the image
    mp.solutions.drawing_utils.draw_landmarks(image, landmarks,
                                              mp_pose.POSE_CONNECTIONS)

    # Display the image and wait for user input
    cv2.imshow('Label Plank', image)
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == ord('0'):
            label = 0  # Correct
            break
        elif key == ord('1'):
            label = 1  # Low back
            break
        elif key == ord('2'):
            label = 2  # High back
            break
        elif key == 27:  # Esc key
            label = -1  # Exit labeling (optional)
            break

    cv2.destroyAllWindows()
    return label


def process_images(image_links):
    """
    Processes a list of image links, extracts landmarks, labels planks, and saves data to CSV.

    Args:
        image_links: A list of URLs or file paths to images.
    """

    data = []  # List to store image data and labels

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        for image_link in image_links:
            # Load the image using the new function
            try:
                image = load_image_from_url(image_link)
                if image is None:
                    print(f"Failed to load image: {image_link}")
                    continue

                # Extract and preprocess landmarks
                try:
                    # Recolor image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False

                    # Make detection (using the 'pose' object)
                    results = pose.process(image_rgb)

                    # Recolor back to BGR
                    image_rgb.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    landmarks_list = extract_landmarks(image.copy())
                    if landmarks_list:
                        preprocessed_landmarks = preprocess_landmarks(
                            landmarks_list, exercise_type="plank")[0]

                        # Label the plank (pass the 'results' object here)
                        label = label_plank(image.copy(),
                                            results.pose_landmarks)
                        if label == -1:  # Exit if Esc key is pressed
                            break

                        # Store the data (without image link)
                        data.append(preprocessed_landmarks + [label])
                    else:
                        print(f"No landmarks detected for {image_link}")

                except Exception as e:
                    print(f"Error processing landmarks for {image_link}: {e}")

            except Exception as e:
                print(f"Error loading or processing image {image_link}: {e}")

    # Save data to CSV (without image link column)
    with open('plank_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['left_shoulder_x', 'left_shoulder_y',
                         'right_shoulder_x', 'right_shoulder_y', 'left_hip_x',
                         'left_hip_y', 'right_hip_x', 'right_hip_y',
                         'left_knee_x', 'left_knee_y', 'right_knee_x',
                         'right_knee_y', 'left_ankle_x', 'left_ankle_y',
                         'right_ankle_x', 'right_ankle_y', 'label'])

        for row in data:
            flattened_row = []  # Flatten the list of lists
            for landmark in row[:-1]:  # Iterate through landmarks (exclude the label)
                flattened_row.extend(landmark)  # Add the x, y coordinates to the flattened list
            flattened_row.append(row[-1])  # Add the label
            writer.writerow(flattened_row)  # Write the flattened row to the CSV


if __name__ == "__main__":
    image_links = [
        "https://i.ytimg.com/vi/pSHjTRCQxIw/maxresdefault.jpg",
        # ... add more image links here ...
    ]
    process_images(image_links)