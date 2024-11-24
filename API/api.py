from fastapi import FastAPI
import os
import subprocess
import sys

from Feedback.Squat_nn.squat_feedback import analyze_squat_video
from Feedback.Plank_nn.plank_feedback import predict_plank

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI()

def check_resource(url: str):
    print(f"Checking resource: {url}")
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
def predict_squat(video_url: str | None = None):
    if video_url is None:
        return {'error': 'Please provide a video path'}

    # Call the video analysis function
    try:
        analysis = analyze_squat_video(video_url)
        return analysis
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
