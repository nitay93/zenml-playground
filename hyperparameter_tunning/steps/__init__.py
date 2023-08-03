from typing import List

from zenml import get_step_context
from zenml.client import Client


def collect_step_outputs(id_list: List[str] = None):
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)
    return {step_name: step.outputs["output"].load() for step_name, step in run.steps.items() if step_name in id_list}