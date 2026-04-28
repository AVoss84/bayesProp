import uvicorn
from fastapi import FastAPI
from my_package.config import global_config as glob

app = FastAPI(
    title="My App", description="This is a simple FastAPI app", version="0.0.1"
)


@app.get("/")
def health_check():
    status = f"Hi there, your service is up! version = {app.version}"
    return status


if __name__ == "__main__":
    uvicorn.run(app, host=str(glob.UC_APP_CONNECTION), port=int(glob.UC_PORT))
