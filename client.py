from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SchedulerAction, SchedulerObservation, SchedulerState, RequestData, GPUData

class SchedulerEnvClient(EnvClient[SchedulerAction, SchedulerObservation, SchedulerState]):
    def _step_payload(self, action: SchedulerAction) -> dict:
        return {"request_id": action.request_id, "gpu_id": action.gpu_id}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        queue = [RequestData(**q) for q in obs_data.get("queue", [])]
        gpus = [GPUData(**g) for g in obs_data.get("gpus", [])]

        observation = SchedulerObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 10),
            queue=queue,
            gpus=gpus,
            feedback_message=obs_data.get("feedback_message", "")
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SchedulerState:
        return SchedulerState(**payload)