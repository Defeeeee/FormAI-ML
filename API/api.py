import os
import subprocess
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Feedback.Live_feedback.main import analyze_live_feed
# from Feedback.Text_Feedback.main import analyze_video

from Feedback.Plank_nn.plank_feedback import predict_plank
# from Feedback.Squat_nn.squat_feedback import analyze_squat_video as analyze_squat_video_feedback

app = FastAPI()

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def check_resource(url: str):
    command = f"curl -o /dev/null --silent -Iw '%{{http_code}}' {url}"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        status_code = result.stdout.strip()  # Extract the status code
        return 1 if status_code == '200' else 0
    except Exception as e:
        return {"error": str(e)}


@app.get('/')
def check():
    return 1


@app.get('/analyze/plank')
def analyze_plank_video(video_url: str | None = None):
    if video_url is None:
        return {'error': 'Please provide a video path'}

    if check_resource(video_url) == 0:
        return {'error': 'Invalid video path'}

    # Call the video analysis function
    try:
        analysis = predict_plank(video_url)
        return analysis
    except Exception as e:
        return {'error': str(e)}


@app.get('/analyze/squat')
def analyze_squat_video(video_url: str | None = None):
    if video_url is None:
        return {'error': 'Please provide a video path'}

    if check_resource(video_url) == 0:
        return {'error': 'Invalid video path'}

    # Call the video analysis function
    try:
        # analysis = analyze_squat_video_feedback(video_url)
        # return analysis
        return {'error': 'Not implemented'}
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
