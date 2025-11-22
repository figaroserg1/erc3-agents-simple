import json
import subprocess
import time
from typing import Annotated, List, Union, Literal, Optional
from annotated_types import MaxLen, MinLen
from pydantic import BaseModel, Field
from erc3 import store, ApiException, TaskInfo, ERC3
from openai import OpenAI

client = OpenAI()


class TaskProfile(BaseModel):
    goal_description: str
    required_items: List[str]
    coupons_mentioned: List[str]
    budget_limit: Optional[float]
    optimise_for: Literal["min_total_price", "max_discount", "exact_request_only"]
    allow_extra_items: bool


class ReportTaskCompletion(BaseModel):
    tool: Literal["report_completion"]
    completed_steps_laconic: List[str]
    code: Literal["completed", "failed"]


class ExecutePythonSnippet(BaseModel):
    tool: Literal["run_python_snippet"]
    code: Annotated[str, MaxLen(2000)] = Field(
        ..., description="Python code to execute for on-the-fly calculations"
    )


class NextStep(BaseModel):
    current_state: Annotated[str, MaxLen(400)]
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(
        ..., description="micro-plan for the remaining steps"
    )
    task_completed: bool
    function: Union[
        ReportTaskCompletion,
        ExecutePythonSnippet,
        store.Req_ListProducts,
        store.Req_ViewBasket,
        store.Req_ApplyCoupon,
        store.Req_RemoveCoupon,
        store.Req_AddProductToBasket,
        store.Req_RemoveItemFromBasket,
        store.Req_CheckoutBasket,
    ] = Field(..., description="execute first remaining step")


system_prompt = """
You are a two-phase shopping assistant for the OnlineStore benchmark.

You must finish a task only when:
- The basket contains exactly the requested items and quantities.
- Any budget constraints are met.
- All provided coupon codes have been evaluated and the best one is applied.
- CheckoutBasket has been executed and the basket is confirmed via ViewBasket.

Rules:
- Do not invent products, prices, or coupons. Trust only store API responses.
- Use ListProducts to exhaust pagination when searching for "all" items or the cheapest option.
- Use ViewBasket to compare totals and discounts after basket or coupon changes.
- ApplyCoupon / RemoveCoupon as needed to test options; only one coupon is active at a time.
- Call CheckoutBasket only after verifying the basket and discounts.
- If a task is impossible (missing items, insufficient budget, etc.), report failure explicitly.
- You can use run_python_snippet to execute short Python calculations (e.g., optimize bundles/discounts).
"""

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"


def parse_task_profile(model: str, task_text: str) -> TaskProfile:
    profile_prompt = """
You extract a structured TaskProfile from the user's shopping request.
Only capture facts that are explicitly stated. Do not invent items or coupon codes.
Keep coupons in upper case. Set budget_limit to null when not specified.
"""

    completion = client.beta.chat.completions.parse(
        model=model,
        response_format=TaskProfile,
        messages=[
            {"role": "system", "content": profile_prompt},
            {"role": "user", "content": task_text},
        ],
        max_completion_tokens=2000,
    )
    return completion.choices[0].message.parsed


def truncate_content(content: str, limit: int = 600) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "... (truncated)"


def run_python_snippet(code: str, timeout: float = 5.0) -> str:
    """Execute ad-hoc Python code and return combined stdout/stderr."""

    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as e:  # pragma: no cover - defensive guard
        return f"Execution failed: {e}"

    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout.strip())
    if result.stderr:
        output_parts.append(f"stderr:\n{result.stderr.strip()}")

    return "\n".join(output_parts) if output_parts else "(no output)"


def run_agent(model: str, api: ERC3, task: TaskInfo):

    store_api = api.get_store_client(task)

    task_profile = parse_task_profile(model, task.task_text)
    task_profile_json = task_profile.model_dump_json()

    # log will contain conversation context for the agent within task
    log = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Structured task profile (JSON): {task_profile_json}"},
        {"role": "user", "content": task.task_text},
    ]

    tool_message_indexes: List[int] = []
    recent_actions: List[str] = []

    # let's limit number of reasoning steps by 30, just to be safe
    for i in range(30):
        step = f"step_{i + 1}"
        print(f"Next {step}... ", end="")

        started = time.time()

        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=10000,
        )

        api.log_llm(
            task_id=task.task_id,
            model="openai/" + model,  # log in OpenRouter format
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        job = completion.choices[0].message.parsed

        action_name = job.function.__class__.__name__
        recent_actions.append(action_name)
        if len(recent_actions) > 8:
            recent_actions.pop(0)

        if len(recent_actions) >= 3 and all(a == action_name for a in recent_actions[-3:]):
            print(f"{CLI_RED}ERR: detected repeated action loop with {action_name}{CLI_CLR}")
            break

        if len(recent_actions) >= 6 and recent_actions[-6:-3] == recent_actions[-3:]:
            print(f"{CLI_RED}ERR: detected oscillation pattern {recent_actions[-3:]}{CLI_CLR}")
            break

        # if SGR wants to finish, then quit loop
        if isinstance(job.function, ReportTaskCompletion):
            print(f"[blue]agent {job.function.code}[/blue]. Summary:")
            for s in job.function.completed_steps_laconic:
                print(f"- {s}")
            break

        # print next sep for debugging
        print(job.plan_remaining_steps_brief[0], f"\n  {job.function}")

        # Let's add tool request to conversation history as if OpenAI asked for it.
        # a shorter way would be to just append `job.model_dump_json()` entirely
        log.append(
            {
                "role": "assistant",
                "content": job.plan_remaining_steps_brief[0],
                "tool_calls": [
                    {
                        "type": "function",
                        "id": step,
                        "function": {
                            "name": job.function.__class__.__name__,
                            "arguments": job.function.model_dump_json(),
                        },
                    }
                ],
            }
        )

        # now execute the tool by dispatching command to our handler
        try:
            if isinstance(job.function, ExecutePythonSnippet):
                txt = run_python_snippet(job.function.code)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
            else:
                result = store_api.dispatch(job.function)
                txt_dict = result.model_dump(exclude_none=True, exclude_unset=True)
                txt = json.dumps(txt_dict, ensure_ascii=False)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ApiException as e:
            txt = e.detail
            # print to console as ascii red
            print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

        # and now we add results back to the convesation history, so that agent
        # we'll be able to act on the results in the next reasoning step.
        log.append({"role": "tool", "content": txt, "tool_call_id": step})
        tool_message_indexes.append(len(log) - 1)

        # keep only the two most recent raw tool responses, compress older ones
        for msg_idx in tool_message_indexes[:-2]:
            log[msg_idx]["content"] = truncate_content(log[msg_idx]["content"])
