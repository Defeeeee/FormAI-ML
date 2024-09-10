import os
import sys

import uvicorn
from fastapi import FastAPI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        analysis = analyze_video(video_path)
    except Exception as e:
        return {'error': str(e)}

    return analysis


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
