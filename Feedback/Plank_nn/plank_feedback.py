import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# 16 features to be extracted from the pose landmarks

# 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x',
#        'right_shoulder_y', 'left_hip_x', 'left_hip_y', 'right_hip_x',
#        'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x',
#        'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x',
#        'right_ankle_y'

# if one side is not visible, we can use the other side to estimate the position, or use the previous frame's position

# def extract_features(results):
#     features = []
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         features = [
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x,
#             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x,
#             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y
#         ]
#
#     for frame_landmarks in features:
#         temp = []
#         for i in selected_indices:
#             landmark = frame_landmarks[i]
#
#             # 1. Visibility Threshold (as before):
#             if landmark[2] >= 0.5:
#                 temp.append([landmark[0], landmark[1]])
#             else:
#                 # 2. Simple Interpolation:
#                 if i > 0 and not np.isnan(
#                         temp[-1][0]):  # If previous landmark is visible
#                     prev_landmark = frame_landmarks[i - 1]
#                     temp.append([prev_landmark[0], prev_landmark[1]])
#                 else:
#                     # 3. Symmetric Landmark Estimation:
#                     if 'LEFT' in mp_pose.PoseLandmark(i).name:
#                         symmetric_index = mp_pose.PoseLandmark(i + 1).value
#                     else:
#                         symmetric_index = mp_pose.PoseLandmark(i - 1).value
#                     symmetric_landmark = frame_landmarks[symmetric_index]
#                     if symmetric_landmark[
#                         2] >= 0.5:  # If the symmetric landmark is visible
#                         # Estimate based on symmetry
#                         temp.append([1 - symmetric_landmark[0],
#                                      symmetric_landmark[1]])
#                     else:
#                         # 4. Default value
#                         temp.append([0.5, 0.5])
#
#         normalized_landmarks.append(temp)
