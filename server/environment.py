import random
import uuid
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SchedulerAction, SchedulerObservation, SchedulerState, RequestData, GPUData

class LLMSchedulerEnvironment:
    def __init__(self):
        self._state = SchedulerState(task_level="easy")
        self.queue: Dict[int, RequestData] = {}
        self.gpus: Dict[int, GPUData] = {}
        self.max_steps = 10
        self.req_counter = 0
        self.gpu_speed = 25.0

    def reset(self, task_level: str = "easy") -> SchedulerObservation:
        self._state = SchedulerState(task_level=task_level, episode_id=str(uuid.uuid4()), step_count=0)
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
            
        return self._get_obs(done=False, reward=0.0, msg=f"Reset to {task_level} mode.")

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
        self._state.step_count += 1
        msg = "Agent chose to wait."
        
        # SPARSE REWARD ENFORCEMENT: Intermediate steps must be exactly 0.0
        step_reward = 0.0  

        if action.request_id != -1 and action.gpu_id != -1:
            if action.request_id in self.queue and action.gpu_id in self.gpus and self.gpus[action.gpu_id].status == "IDLE":
                req = self.queue.pop(action.request_id)
                gpu = self.gpus[action.gpu_id]
                gpu.status, gpu.remaining_tokens, gpu.current_request_tier = "BUSY", req.token_size, req.tier
                msg = f"Success: Dispatched Req {action.request_id} to GPU {action.gpu_id}."
            else:
                msg = "Error: Invalid action or busy GPU."

        for gpu in self.gpus.values():
            if gpu.status == "BUSY":
                gpu.remaining_tokens -= self.gpu_speed
                if gpu.remaining_tokens <= 0:
                    gpu.status, gpu.remaining_tokens, gpu.current_request_tier = "IDLE", 0.0, None
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
                
            # CLAMP: Ensure the final score is safely bounded
            self._state.final_score = max(0.01, min(0.99, float(raw_score)))
            step_reward = self._state.final_score

        return self._get_obs(done=done, reward=step_reward, msg=msg)

    def _get_obs(self, done: bool, reward: float, msg: str) -> SchedulerObservation:
        return SchedulerObservation(
            done=done, reward=reward, step=self._state.step_count, max_steps=self.max_steps,
            queue=list(self.queue.values()), gpus=list(self.gpus.values()), feedback_message=msg
        )

    @property
    def state(self) -> SchedulerState:
        self._state.final_score = max(0.01, min(0.99, self._state.final_score))
        return self._state

    def close(self):
        pass

    async def reset_async(self, task_level: str = "easy") -> SchedulerObservation:
        return self.reset(task_level=task_level)

    async def step_async(self, action: SchedulerAction) -> SchedulerObservation:
        return self.step(action)

    async def get_state_async(self) -> SchedulerState:
        self._state.final_score = max(0.01, min(0.99, self._state.final_score))
        return self.state

    async def close_async(self):
        self.close()
