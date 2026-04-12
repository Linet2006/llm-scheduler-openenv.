import random
import uuid
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SchedulerAction, SchedulerObservation, SchedulerState, RequestData, GPUData

# Module-level shared state — survives across instances
_GLOBAL_STATE = SchedulerState(task_level="easy")

class LLMSchedulerEnvironment:
    def __init__(self):
        global _GLOBAL_STATE
        self._state = _GLOBAL_STATE
        self.queue: Dict[int, RequestData] = {}
        self.gpus: Dict[int, GPUData] = {}
        self.max_steps = 10
        self.req_counter = 0
        self.gpu_speed = 25.0

    def reset(self, task_level: str = "easy") -> SchedulerObservation:
        global _GLOBAL_STATE
        self._state = SchedulerState(task_level=task_level, episode_id=str(uuid.uuid4()), step_count=0)
        _GLOBAL_STATE = self._state

        if task_level == "easy":
            self.max_steps, num_gpus, initial_reqs = 5, 1, 2
        elif task_level == "medium":
            self.max_steps, num_gpus, initial_reqs = 10, 2, 4
        else:
            self.max_steps, num_gpus, initial_reqs = 15, 3, 6

        self.gpus = {i: GPUData(gpu_id=i, status="IDLE", remaining_tokens=0.0) for i in range(num_gpus)}
        self.queue = {}
        self.req_counter = 0
        for _ in range(initial_reqs):
            self._spawn_request()

        return self._get_obs(done=False, reward=0.05, msg=f"Reset to {task_level} mode.")

    def _spawn_request(self):
        tier = "PAID" if random.random() > 0.5 else "FREE"
        req = RequestData(
            request_id=self.req_counter, tier=tier,
            token_size=round(random.uniform(20.0, 80.0), 1),
            deadline=random.randint(3, 8), wait_time=0
        )
        self.queue[self.req_counter] = req
        self.req_counter += 1

    def step(self, action: SchedulerAction) -> SchedulerObservation:
        global _GLOBAL_STATE
        self._state.step_count += 1
        msg = "Agent chose to wait."
        step_reward = 0.05

        if action.request_id != -1 and action.gpu_id != -1:
            if action.request_id not in self.queue:
                step_reward = 0.02
                msg = f"Error: Req {action.request_id} missing."
            elif action.gpu_id not in self.gpus:
                step_reward = 0.02
                msg = f"Error: GPU {action.gpu_id} missing."
            elif self.gpus[action.gpu_id].status == "BUSY":
                step_reward = 0.02
                msg = f"Error: GPU {action.gpu_id} BUSY."
            else:
                req = self.queue.pop(action.request_id)
                gpu = self.gpus[action.gpu_id]
                gpu.status = "BUSY"
                gpu.remaining_tokens = req.token_size
                gpu.current_request_tier = req.tier
                step_reward = 0.80 if req.tier == "PAID" else 0.60
                msg = f"Success: Dispatched Req {action.request_id} to GPU {action.gpu_id}."
        else:
            idle_gpus = [g for g in self.gpus.values() if g.status == "IDLE"]
            if self.queue and idle_gpus:
                step_reward = 0.03

        for gpu in self.gpus.values():
            if gpu.status == "BUSY":
                gpu.remaining_tokens -= self.gpu_speed
                if gpu.remaining_tokens <= 0:
                    gpu.status = "IDLE"
                    gpu.remaining_tokens = 0.0
                    gpu.current_request_tier = None
                    self._state.completed_requests += 1

        to_drop = []
        for req_id, req in self.queue.items():
            req.deadline -= 1
            if req.deadline < 0:
                to_drop.append(req_id)
        for req_id in to_drop:
            del self.queue[req_id]
            self._state.dropped_requests += 1
            self._state.sla_violations += 1

        if self._state.task_level == "hard" and random.random() < 0.4 and len(self.queue) < 8:
            self._spawn_request()

        done = (
            self._state.step_count >= self.max_steps or
            (len(self.queue) == 0 and all(g.status == "IDLE" for g in self.gpus.values()))
        )

        if done:
            total = self._state.completed_requests + self._state.dropped_requests
            if total == 0:
                raw_score = 0.50
            else:
                score = self._state.completed_requests / total
                penalty = (self._state.sla_violations / total) * 0.3
                raw_score = score - penalty
            self._state.final_score = max(0.01, min(0.99, float(raw_score)))
            step_reward = self._state.final_score

        step_reward = max(0.01, min(0.99, round(float(step_reward), 2)))
        _GLOBAL_STATE = self._state
        return self._get_obs(done=done, reward=step_reward, msg=msg)

    def _get_obs(self, done: bool, reward: float, msg: str) -> SchedulerObservation:
        return SchedulerObservation(
            done=done, reward=reward, step=self._state.step_count, max_steps=self.max_steps,
            queue=list(self.queue.values()), gpus=list(self.gpus.values()), feedback_message=msg
        )

    @property
    def state(self) -> SchedulerState:
        global _GLOBAL_STATE
        _GLOBAL_STATE.final_score = max(0.01, min(0.99, _GLOBAL_STATE.final_score))
        return _GLOBAL_STATE

    def close(self):
        pass

    async def reset_async(self, task_level: str = "easy") -> SchedulerObservation:
        return self.reset(task_level=task_level)

    async def step_async(self, action: SchedulerAction) -> SchedulerObservation:
        return self.step(action)

    async def get_state_async(self) -> SchedulerState:
        global _GLOBAL_STATE
        _GLOBAL_STATE.final_score = max(0.01, min(0.99, _GLOBAL_STATE.final_score))
        return _GLOBAL_STATE

    async def close_async(self):
        self.close()
