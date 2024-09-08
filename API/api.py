import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

# Import the video analysis function from your analysis script
from path/to/your/video_analysis_script import analyze_video  # Adjust the path accordingly

app = FastAPI()

load_dotenv()

# Your secret token, replace with your actual secret
SECRET_TOKEN = os.getenv('TOKEN')


@app.get('/')
def check(token: str | None = None):
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token was not provided",
        )
    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return 1


@app.get('/analyze_plank')
def analyze_plank_video(token: str | None = None, video_path: str | None = None):
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token was not provided",
        )
    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video path was not provided",
        )

    # Call the video analysis function
    results = analyze_video(video_path)

    return results  # Return the analysis results


if __name__ == '__main__':
    uvicorn.run(app)