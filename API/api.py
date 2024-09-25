import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Feedback.Live_feedback.main import analyze_live_feed
from Feedback.Text_Feedback.main import analyze_video

app = FastAPI()


@app.get('/')
def check():
    return 1


@app.get('/analyze_plank')
def analyze_plank_video(video_path: str | None = None):
    print(video_path)
    if video_path is None:
        video_path = '/Users/defeee/Downloads/Screenshot 2024-09-08 at 12.14.38.png'

    # Call the video analysis function
    try:
        analysis = analyze_video(video_path, 'plank')
    except Exception as e:
        return {'error': str(e)}

    return analysis


@app.get('/analyze_plank/live')
def analyze_plank_live_feed():
    return {'error': 'WIP: Not implemented yet'}
    # return StreamingResponse(analyze_live_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('/classify')
def classify():
    return {'error': 'WIP: Not implemented yet'}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
