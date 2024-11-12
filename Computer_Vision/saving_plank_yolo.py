import cv2
import torch
import numpy as np
import requests

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

def extract_landmarks(image, model):
    """
    Extracts landmarks from an image using YOLO.

    Args:
        image: The image to process.
        model: The YOLO model.

    Returns:
        A list of landmark sequences, where each sequence corresponds to a frame
        and contains the x, y coordinates of visible landmarks.
    """
    results = model(image)
    landmarks = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        landmarks.append([x1.item(), y1.item(), x2.item(), y2.item()])
    return landmarks

def preprocess_landmarks(landmarks_list, exercise_type="plank"):
    """
    Preprocesses the extracted landmarks.

    Args:
        landmarks_list: List of landmark sequences.
        exercise_type: Type of exercise ("plank", "squat", etc.).

    Returns:
        Preprocessed landmarks.
    """
    if exercise_type == "plank":
        selected_indices = [0, 1, 2, 3]

    normalized_landmarks = []
    for frame_landmarks in landmarks_list:
        temp = []
        for i in selected_indices:
            landmark = frame_landmarks[i]
            temp.append([landmark[0], landmark[1]])
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
    for landmark in landmarks:
        x1, y1, x2, y2 = landmark
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow('Label Plank', image)
    while True:
        key = cv2.waitKey(0)
        if key == ord('0'):
            label = 0
            break
        elif key == ord('1'):
            label = 1
            break
        elif key == ord('2'):
            label = 2
            break
        elif key == 27:
            label = -1
            break

    cv2.destroyAllWindows()
    return label

def process_images(image_links, model):
    """
    Processes a list of image links, extracts landmarks, labels planks, and saves data to CSV.

    Args:
        image_links: A list of URLs or file paths to images.
        model: The YOLO model.
    """
    data = []

    for image_link in image_links:
        image = load_image_from_url(image_link)
        if image is None:
            print(f"Failed to load image: {image_link}")
            continue

        landmarks_list = extract_landmarks(image, model)
        if landmarks_list:
            preprocessed_landmarks = preprocess_landmarks(landmarks_list, exercise_type="plank")[0]
            label = label_plank(image.copy(), landmarks_list)
            if label == -1:
                break
            data.append(preprocessed_landmarks + [label])
        else:
            print(f"No landmarks detected for {image_link}")

    with open('plank_data_yolo.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x1', 'y1', 'x2', 'y2', 'label'])
        for row in data:
            flattened_row = []
            for landmark in row[:-1]:
                flattened_row.extend(landmark)
            flattened_row.append(row[-1])
            writer.writerow(flattened_row)

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    image_links = [
        "https://storage.googleapis.com/kagglesdsdata/datasets/920599/1559111/DATASET/TRAIN/plank/00000345.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com/20241103/auto/storage/goog4_request&X-Goog-Date=20241103T001121Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=633e31a23dfb8d883d46348eb1fe4996c6f1e04a67d5701020545ede14cfa52630420c12d66be8129918d98dbfa407f9e2813f5cf501eaa42ea0c83dcccc7af1d8f636b9edae4d3eb73fdbb6aa984ea58cde3c797cf4cbe8098ee4303a338cab56a7f003614caa2051d547c4f90f8d8f90530ec67b3083f4084dc57138575e464e25e5921f24d1f6e7f78ac781c84dd26fd62c3096283397aa1d30aa8e586dbb45718638dbc11f97f49909758190829a24fc09972a1485af2fc4cfa6e8dc4ea047e44fe86a95ca8a60aa0e1542c6ae74f9ac2542863d29f79640d39b9ce2004119bee28a96864cad446d17fd0d0c0841cb723ca7b72550e13dadbfd9595e67ff"
    ]
    process_images(image_links, model)
