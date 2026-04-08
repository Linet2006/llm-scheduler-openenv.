import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI
from client import SchedulerEnvClient, SchedulerAction

# --- STRICT PROXY CONFIGURATION (PHASE 2 FIX) ---
# We MUST prioritize 'API_KEY' as the judge injects this name specifically.
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://router.huggingface.co/v1"

# Use model name from judge or default to Qwen
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")

# --- STRICT PHASE 2 LOGGING FORMAT ---
def log_start(task_name: str):
    # Format: [START] task=NAME
    print(f"[START] task={task_name}", flush=True)

def log_step(step: int, reward: float):
    # Format: [STEP] step=1 reward=0.5
    print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

def log_end(task_name: str, score: float, steps: int):
    # Format: [END] task=NAME score=0.95 steps=1
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)

SYSTEM_PROMPT = """You are an AI load balancer. Dispatch ONE request from the 'queue' to an 'IDLE' GPU.
Reply ONLY with JSON format: {"request_id": int, "gpu_id": int}. If queue is empty or no GPUs are IDLE, return {"request_id": -1, "gpu_id": -1}."""

def get_model_message(client: OpenAI, obs_json: str) -> tuple[SchedulerAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs_json}],
            temperature=0.1, 
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return SchedulerAction(request_id=data.get("request_id", -1), gpu_id=data.get("gpu_id", -1)), text
    except Exception:
        return SchedulerAction(request_id=-1, gpu_id=-1), '{"request_id": -1, "gpu_id": -1}'

async def run_task(task_id: str, env_client: SchedulerEnvClient, openai_client: OpenAI):
    # 1. LOG START
    log_start(task_id)
    
    level = task_id.split("-")[-1]
    result = await env_client.reset(task_level=level)
    
    steps_count = 0
    for step in range(1, 21): 
        if result.done:
            break
        
        steps_count = step
        obs_json = result.observation.model_dump_json()
        action_obj, _ = get_model_message(openai_client, obs_json)
        
        result = await env_client.step(action_obj)
        
        # 2. LOG STEP (Required for parsing rewards)
        log_step(step, result.reward)
        
    # 3. LOG END (Required for final score)
    final_state = await env_client.state()
    log_end(task_id, final_state.final_score, steps_count)

async def main() -> None:
    # Initialize with the prioritized proxy variables
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
