import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Text_Feedback.main import analyze_video

app = FastAPI()

load_dotenv()

# Your secret token, replace with your actual secret
SECRET_TOKEN = os.getenv('TOKEN')


@app.get('/')
def check():
    return 1


@app.get('/analyze_plank')
def analyze_plank_video(video_path: str | None = None):
    print(video_path)
    if video_path is None:
        video_path = '/Users/defeee/Downloads/Screenshot 2024-09-08 at 12.14.38.png'

    # Call the video analysis function
    results = analyze_video(video_path)

    return results  # Return the analysis results


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
