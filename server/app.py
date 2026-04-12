from openenv.core.env_server import create_fastapi_app
from models import SchedulerAction, SchedulerObservation
from server.environment import LLMSchedulerEnvironment, _GLOBAL_STATE
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

app = create_fastapi_app(LLMSchedulerEnvironment, SchedulerAction, SchedulerObservation)

app.router.routes = [
    r for r in app.router.routes
    if not (isinstance(r, APIRoute) and r.path == '/state')
]

@app.get("/state")
def get_state():
    from server.environment import _GLOBAL_STATE as s
    score = max(0.01, min(0.99, float(s.final_score)))
    return JSONResponse({
        "episode_id": getattr(s, 'episode_id', None),
        "step_count": s.step_count,
        "task_level": s.task_level,
        "completed_requests": s.completed_requests,
        "dropped_requests": s.dropped_requests,
        "sla_violations": s.sla_violations,
        "final_score": score
    })

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
