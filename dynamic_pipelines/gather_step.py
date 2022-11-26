import inspect
from typing import List, Dict, Callable, Type

from zenml.environment import Environment
from zenml.post_execution import get_run
from zenml.steps import BaseStep, step


def gather(step_name_prefix: str) -> List[Dict]:
    run_name = Environment().step_environment.pipeline_run_id
    run = get_run(run_name)
    steps = [
        step for step in run.steps
        if step.name.startswith(step_name_prefix)
    ]

    return [
        {k: v.read() for k, v in step.outputs.items()} for step in steps
    ]


class GatherSteps:
    def __init__(self, func: Callable):
        self.func = func
        annotations = {**inspect.getfullargspec(inspect.unwrap(func)).annotations}
        self.return_type = annotations.pop('return', None)
        self.param_type = annotations.pop('params', None)

    def gather_steps_like(self, prefix: str) -> Type[BaseStep]:
        param_type = self.param_type
        return_type = self.return_type

        def step_function(params: param_type) -> return_type:
            outputs = gather(prefix)
            return self.func(params, outputs)

        step_function.__name__ = f"{self.func.__name__}_gather_steps_like_prefix_{prefix}"

        return step(step_function)


def gather_step(func: Callable) -> GatherSteps:
    return GatherSteps(func)
