from openenv.core.env_server import create_fastapi_app
from models import SchedulerAction, SchedulerObservation
from server.environment import LLMSchedulerEnvironment

app = create_fastapi_app(LLMSchedulerEnvironment, SchedulerAction, SchedulerObservation)
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the OpenEnv LLM Scheduler!",
        "status": "Green and Running 🚀",
        "ready_for_judges": True
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
