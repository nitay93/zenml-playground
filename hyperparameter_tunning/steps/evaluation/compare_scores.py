from typing import List

from zenml import step, get_step_context
from zenml.client import Client


def collect_step_outputs(id_list: List[str] = None):
    """
    Collect the step outputs that correspond to the id_list
    """
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)
    return {step_name: step.outputs["output"].load() for step_name, step in run.steps.items() if step_name in id_list}


@step
def compare_score(evaluation_steps: List[str], is_max=True) -> dict:
    """Compare scores over multiple evaluation outputs."""
    evaluation_results = collect_step_outputs(evaluation_steps)

    optimize = max if is_max else min
    optimal_result = optimize(evaluation_results.values(), key=lambda x: x.score)
    print(
        f"optimal evaluation at {optimal_result.model_parameters}. score = "
        f"{optimal_result.score*100:.2f}%"
    )
    return optimal_result.model_parameters
