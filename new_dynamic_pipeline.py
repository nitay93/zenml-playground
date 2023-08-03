from typing import Any, List, Callable

from zenml import step, pipeline, get_step_context
from zenml.client import Client


def collect_step_outputs(id_list: List[str] = None):
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)
    return {step_name: step.outputs["output"].load() for step_name, step in run.steps.items() if step_name in id_list}


@step
def get_string() -> str:
    return "test"


@step
def return_same(i: int) -> int:
    return i


@step
def print_string(s: str, steps_to_collect: List[str]) -> None:
    step_outputs = collect_step_outputs(steps_to_collect)
    print(s + str(step_outputs))


@pipeline(enable_cache=False)
def my_dynamic_pipeline():
    step_ids = []
    for i in range(5):
        step_id = f"return_same_{i+1}" #"return_same" if i == 0 else f"return_same_{i+1}"
        return_same(i=i, id=step_id)
        step_ids.append(step_id)

    print_string("All: ", steps_to_collect=step_ids, after=step_ids)


my_dynamic_pipeline()