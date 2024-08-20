import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

# Your secret token, replace with your actual secret
SECRET_TOKEN = os.getenv('TOKEN')


@app.get('/')
def check(token: str | None = None):
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token not provided",
        )
    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return 1


if __name__ == '__main__':
    uvicorn.run(app)
