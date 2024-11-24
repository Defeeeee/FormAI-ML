from fastapi import FastAPI
import os
import subprocess
import sys
from pydantic import BaseModel
import dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Feedback.Squat_nn.squat_feedback import analyze_squat_video
from Feedback.Plank_nn.plank_feedback import predict_plank

# Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())

app = FastAPI()

import os
import aiohttp  # For making asynchronous HTTP requests

class ExerciseData(BaseModel):
    exercise_type: str
    ai_prediction: str
    mse: float | None = None
    confidence: float | None = None
    keyframes_results: list | None = None
    video_url: str | None = None

async def call_gemini_api(prompt: str):
    """Calls the Gemini API with the given prompt."""
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API Key: {api_key}")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    request_body = {
      "contents": [
        {
          "parts": [
            {
              "text": prompt  # Use the provided prompt
            }
          ]
        }
      ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=request_body) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error calling Gemini API: {response.status}")
                return None

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

from fastapi import FastAPI
from fastapi.responses import JSONResponse  # For explicit JSON responses

app = FastAPI()

@app.post('/gemini')
async def gemini(data: ExerciseData):
    exercise_type = data.exercise_type
    ai_prediction = data.ai_prediction
    mse = data.mse
    confidence = data.confidence
    keyframes_results = data.keyframes_results
    video_url = data.video_url

    prompt = f"""
        You are a helpful AI assistant that provides feedback on fitness exercises. 
        Analyze the following exercise video and provide feedback on the user's form:
        The analysis following has been made by a simple convolutional neural network trained on several examples of a well done excersise
        Using mediapipe to detect the key points of the body and then calculate the angles between them

        Exercise Type: {exercise_type}
        AI Prediction: {ai_prediction}
        {"MSE" if mse is not None else "Confidence"}: {mse if mse is not None else confidence}
        {"Keyframes Results" if keyframes_results is not None else ""}: {keyframes_results if keyframes_results is not None else ""}
        {"Video URL" if video_url is not None else ""}: {video_url if video_url is not None else ""}

        Provide specific, BRIEF and constructive feedback on the user's exercise form, 
        highlighting any mistakes and suggesting improvements. 
        respond in a short paragraph.
        """

    # Call the Gemini API
    gemini_response = await call_gemini_api(prompt)

    if gemini_response:
        # Process the response and extract the feedback
        feedback = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

        result = {
            "success": True,
            "feedback": feedback
        }
        return JSONResponse(content=result)
    else:
        return {"error": "Failed to get feedback from Gemini API"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
