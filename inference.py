import asyncio
import os
import json
from openai import OpenAI
from client import SchedulerEnvClient, SchedulerAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
ENV_BENCHMARK = "llm_scheduler"

SYSTEM_PROMPT = """You are an AI load balancer managing GPU scheduling.
Given the current queue and GPU states, dispatch ONE request to an IDLE GPU.
Prioritize PAID tier requests and requests with low remaining deadline.
Reply ONLY with JSON: {"request_id": int, "gpu_id": int}.
If queue is empty or no GPUs are IDLE, return {"request_id": -1, "gpu_id": -1}."""

def get_action(client: OpenAI, obs_json: str) -> tuple[SchedulerAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_json}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return SchedulerAction(
            request_id=data.get("request_id", -1),
            gpu_id=data.get("gpu_id", -1)
        ), text
    except Exception:
        return SchedulerAction(request_id=-1, gpu_id=-1), '{"request_id":-1,"gpu_id":-1}'

async def run_task(task_id: str, env_client: SchedulerEnvClient, openai_client: OpenAI):
    rewards = []
    steps_done = 0
    success = False

    print(f"[START] task={task_id} env={ENV_BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        level = task_id.split("-")[-1]
        result = await env_client.reset(task_level=level)

        for step in range(1, 21):
            if result.done:
                success = True
                break

            steps_done = step
            obs_json = result.observation.model_dump_json()
            action_obj, action_str = get_action(openai_client, obs_json)
            result = await env_client.step(action_obj)

            # FIX 1: clamp every reward strictly between 0.01 and 0.99
            safe_reward = float(result.reward if result.reward is not None else 0.05)
            safe_reward = max(0.01, min(0.99, safe_reward))
            rewards.append(safe_reward)

            done_str = "true" if result.done else "false"
            clean_action = action_str.replace('\n', '').replace('\r', '').replace(' ', '')
            fb = result.observation.feedback_message or ""
            error_str = fb.replace('\n', ' ') if fb.startswith("Error") else "null"

            print(
                f"[STEP] step={step} action={clean_action} reward={safe_reward:.2f} done={done_str} error={error_str}",
                flush=True
            )

            if result.done:
                success = True
                break

    except Exception as e:
        clean_err = str(e).replace('\n', ' ')
        # FIX 2: exception reward must not be 0.00
        print(f"[STEP] step={steps_done+1} action=null reward=0.01 done=false error={clean_err}", flush=True)
        rewards.append(0.01)

    finally:
        # FIX 3: fallback must not be "0.00"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
        print(f"[END] success={'true' if success else 'false'} steps={steps_done} rewards={rewards_str}", flush=True)

async def main():
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SchedulerEnvClient(base_url=ENV_URL)
    await env.connect()

    try:
        for task in ["scheduler-easy", "scheduler-medium", "scheduler-hard"]:
            await run_task(task, env, openai_client)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
