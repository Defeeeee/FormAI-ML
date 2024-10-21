import os
import subprocess
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Feedback.Live_feedback.main import analyze_live_feed
# from Feedback.Text_Feedback.main import analyze_video

from Models.Utilities.classify_plank import classify_plank

app = FastAPI()


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


@app.get('/analyze_plank')
def analyze_plank_video(video_path: str | None = None):
    if video_path is None:
        return {'error': 'Please provide a video path'}

    if check_resource(video_path) == 0:
        return {'error': 'Invalid video path'}

    # Call the video analysis function
    try:
        analysis = classify_plank(video_path)
    except Exception as e:
        return {'error': str(e)}

    return analysis


@app.get('/analyze_plank/live')
def analyze_plank_live_feed():
    return {'error': 'WIP: Not implemented yet'}
    # return StreamingResponse(analyze_live_feed(), media_type="multipart/x-mixed-replace; boundary=frame")


# @app.get('/classify')
# def classifiere(path: str | None = None):
#     return classify(path)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
