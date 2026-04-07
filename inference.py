import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from client import SchedulerEnvClient, SchedulerAction

# --- FINAL DEPLOYMENT CONFIGURATION ---
# Use os.getenv so the token is never hardcoded in the file!
API_KEY = os.getenv("HF_TOKEN") or "your_hf_token_here"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")

# --- STRICT LOGGING FORMATS (DO NOT CHANGE) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = """You are an AI load balancer. Dispatch ONE request from the 'queue' to an 'IDLE' GPU.
Reply ONLY with JSON format: {"request_id": int, "gpu_id": int}. If queue is empty or no GPUs are IDLE, return {"request_id": -1, "gpu_id": -1}."""

def get_model_message(client: OpenAI, obs_json: str) -> tuple[SchedulerAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs_json}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return SchedulerAction(request_id=data.get("request_id", -1), gpu_id=data.get("gpu_id", -1)), text
    except Exception as exc:
        print(f"[DEBUG] LLM Error: {exc}")
        return SchedulerAction(request_id=-1, gpu_id=-1), '{"request_id": -1, "gpu_id": -1}'

async def run_task(task_id: str, env_client: SchedulerEnvClient, openai_client: OpenAI):
    log_start(task=task_id, env="llm_scheduler", model=MODEL_NAME)
    level = task_id.split("-")[-1]
    result = await env_client.reset(task_level=level)
    
    rewards, steps_taken = [], 0
    
    for step in range(1, 20): # Failsafe limit
        if result.done: break
        
        obs_json = result.observation.model_dump_json()
        action_obj, raw_action_str = get_model_message(openai_client, obs_json)
        
        result = await env_client.step(action_obj)
        reward, done = result.reward or 0.0, result.done
        rewards.append(reward)
        steps_taken = step
        
        flat_action = raw_action_str.replace("\n", "").replace("\r", "")
        log_step(step=step, action=flat_action, reward=reward, done=done, error=None)

    final_state = await env_client.state()
    score = final_state.final_score
    log_end(success=score >= 0.5, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SchedulerEnvClient(base_url=ENV_URL)
    await env.connect()
    
    try:
        for task in ["scheduler-easy", "scheduler-medium", "scheduler-hard"]:
            await run_task(task, env, openai_client)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())