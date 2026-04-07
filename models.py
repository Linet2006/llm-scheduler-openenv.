from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class SchedulerAction(Action):
    request_id: int
    gpu_id: int

class RequestData(Observation):
    request_id: int
    tier: str
    token_size: float
    deadline: int
    wait_time: int

class GPUData(Observation):
    gpu_id: int
    status: str
    remaining_tokens: float
    current_request_tier: Optional[str] = None

class SchedulerObservation(Observation):
    step: int
    max_steps: int
    queue: List[RequestData]
    gpus: List[GPUData]
    feedback_message: str
    
class SchedulerState(State):
    task_level: str
    completed_requests: int = 0
    dropped_requests: int = 0
    sla_violations: int = 0
    final_score: float = 0.0 # 0.0 to 1.0 grader score