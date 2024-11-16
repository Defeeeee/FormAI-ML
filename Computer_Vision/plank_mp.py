import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests

mp_pose = mp.solutions.pose


def extract_plank_features(image_url):
    try:
        image = download_image(image_url)
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks.landmark
            features = {
                'shoulder_hip_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE]),
                'hip_knee_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
                'shoulder_elbow_wrist_angle': calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST]),
                'back_curvature': calculate_back_curvature(landmarks),
                'head_alignment': calculate_head_alignment(landmarks),
                'arm_placement': calculate_arm_placement(landmarks),
                'foot_placement': calculate_foot_placement(landmarks)
            }
            return features
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_back_curvature(landmarks):
    shoulder_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
    hip_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
    curvature = abs(shoulder_hip_angle - hip_knee_angle)
    return curvature


def calculate_head_alignment(landmarks):
    shoulder_ear_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                         landmarks[mp_pose.PoseLandmark.LEFT_EAR],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP])
    alignment = abs(180 - shoulder_ear_angle)
    return alignment


def calculate_arm_placement(landmarks):
    shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
    elbow_x = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x
    distance = abs(shoulder_x - elbow_x)
    return distance


def calculate_foot_placement(landmarks):
    left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x
    right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x
    distance = abs(left_ankle_x - right_ankle_x)
    return distance


def download_image(image_url):
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def process_images(image_urls):
    all_features = []
    for image_url in image_urls:
        features = extract_plank_features(image_url)
        if features:
            all_features.append(features)
    df = pd.DataFrame(all_features)
    df.to_csv('plank_features.csv', index=False)
    print("Features saved to plank_features.csv")


if __name__ == "__main__":
    image_urls = [
        "https://media.self.com/photos/615378c1150f8742ac896749/4:3/w_2560%2Cc_limit/Forearm%2520Plank%2520-%2520Delise_001.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSI7pYEF9068MzJrkILCvRKOXtn4phJw-4a6w&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJpHouae9ZO99SwRSybUhpXOAb72VEKtObcg&s",
        "https://media.post.rvohealth.io/wp-content/uploads/sites/2/2019/05/ForearmPlank.png",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8DvIOt5iwkhcdLjNOkRJy4SNcQvJOAjXE1Q&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRW05eOsVyV14RsvkV0UadQhnyWu1gUwvBRJg&s"
        "https://i.guim.co.uk/img/media/7ce6b499ccf7634bc6820032f0ca2e19420ae88b/0_0_5000_3001/master/5000.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=9edf0d2f5b912c37e8f4a2931c46fe3a",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrvyDXhiAucamQ_DBe85F7pGiV9z8TVVOq1Q&s",
        "https://i.pinimg.com/736x/6c/9f/d3/6c9fd3dd4a6e9f55eb7b806fa150717a.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLO7z3h_9vdG1JegUAjUYyYfGorY-bRASVmQ&s",
        "https://www.shutterstock.com/image-photo/fit-young-woman-doing-plank-260nw-5447578.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6wWfERd9KyguG6u8_qRqsK2O1TfHqbxxXVQ&s",
        "https://media.istockphoto.com/id/531520915/photo/correct-plank-position-and-fitness.jpg?s=170667a&w=0&k=20&c=-EjVCBzII0Ao1g08s5RGTWdmp6Xwbor7Ak3E4-b4NPU=",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPBOoRk8hpqwbB48YmN1VKo5rOgtbJiHUSwQ&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQh54FZrjZBVDCqkfK7ccE0EPSJ6wCp3YDboA&s",
        "https://hips.hearstapps.com/hmg-prod/images/mid-adult-man-doing-plank-exercise-royalty-free-image-1578935805.jpg?resize=980:*",
        "https://miro.medium.com/v2/resize:fit:960/1*g7nzOoM4PhyNjZ-JHuTzKg.jpeg",
        "https://images.squarespace-cdn.com/content/v1/5da66994195c19564c58f4e7/1588633950558-MFTBFTGJVPLXRVBGMDEG/plankCorrect.png",
        "https://img.livestrong.com/375/clsd/getty/c48901cbcf0845a4887a11688bcb165e.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3eA_H3W-Apffm5OcXFq2V4P6krdx7pi2xIw&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFAL7gNNCBi9pWhCHp1D3Jm8Hnk6wKHd1RGA&s",
        "https://hips.hearstapps.com/vidthumb/brightcove/578940e9e694aa370d883627/thumb_1468612843.png",
        "https://i.ytimg.com/vi/mH5Sfb_KTGg/sddefault.jpg",
        "https://www.shutterstock.com/image-vector/man-elbow-plank-gradient-icon-260nw-1746491333.jpg",
        "https://i.ytimg.com/vi/zygY7fXFOz4/sddefault.jpg",
    ]
    process_images(image_urls)
