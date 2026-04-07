from openenv.core.env_server import create_fastapi_app
from models import SchedulerAction, SchedulerObservation
from server.environment import LLMSchedulerEnvironment

app = create_fastapi_app(LLMSchedulerEnvironment, SchedulerAction, SchedulerObservation)