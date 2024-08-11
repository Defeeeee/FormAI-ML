import uvicorn
from fastapi import FastAPI

app = FastAPI()


# Placeholder route, future routes will be added here
@app.get('/')
def check():
    return 1


if __name__ == '__main__':
    uvicorn.run(app)
