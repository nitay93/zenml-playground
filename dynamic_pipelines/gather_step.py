from typing import List, TypeVar

from zenml.environment import Environment
from zenml.post_execution import get_run
from zenml.steps import Output, BaseParameters

GP = TypeVar("GP", bound="GatherParameters")


class GatherStepsParameters(BaseParameters):
    output_steps_prefix: str = None
    output_steps_names: List[str] = None


class OutputParameters(BaseParameters):
    @staticmethod
    def _extract(gather_steps_params: GatherStepsParameters) -> List[dict]:
        run_name = Environment().step_environment.pipeline_run_id
        run = get_run(run_name)
        prefix_steps = [
            s for s in run.steps
            if s.name.startswith(gather_steps_params.output_steps_prefix)
        ] if gather_steps_params.output_steps_prefix is not None else []

        named_steps = [
            s for s in run.steps
            if s.name in gather_steps_params.output_steps_names
        ] if gather_steps_params.output_steps_names is not None else []

        return [
            {k: v.read() for k, v in s.outputs.items()} for s in prefix_steps + named_steps
        ]

    @classmethod
    def extract(cls: GP, gather_steps_params: GatherStepsParameters) -> List[GP]:
        return [cls(**o) for o in cls._extract(gather_steps_params)]

    @classmethod
    def as_output(cls) -> Output:
        return Output(**{field_name: field.type_ for field_name, field in cls.__fields__.items()})
