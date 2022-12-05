import functools
import uuid
from abc import abstractmethod
from functools import partial
from typing import Any, List, Type, Union, Tuple, Iterable

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep, BaseParameters

from dynamic_pipelines.gather_step import GatherSteps


class DynamicPipeline(BasePipeline):
    def __init__(self, *steps: BaseStep, **kwargs: Any):
        if type(self).STEP_SPEC != {}:
            raise RuntimeError(f"A dynamic pipeline {self.__class__.__name__} was already initialized. Consider using "
                               f"PipelineFactory to generate a new pipeline based on a pipeline template.")
        type(self).STEP_SPEC = {s.name: type(s) for s in steps}
        super().__init__(*steps, **kwargs)

    def new_step(self, step: Type[BaseStep], step_id: Any = None, param: BaseParameters = None) -> BaseStep:
        named_step = self.named_step(step, step_id)
        return named_step() if param is None else named_step(param)

    @classmethod
    def named_step(cls, step: Type[BaseStep], step_id: Any = None) -> functools.partial:
        name = uuid.uuid1().hex if step_id is None else step_id
        return partial(step, name=cls.get_step_name(step, name))

    @staticmethod
    def get_prefix(step: Type[BaseStep]) -> str:
        return step.__name__

    @classmethod
    def get_step_name(cls, step: Type[BaseStep], step_id: Any = None) -> str:
        return cls.get_prefix(step) if step_id is None else f"{cls.get_prefix(step)}_{step_id}"

    def define_gather_step(self, gather_step: GatherSteps, by_type: Type[BaseStep] = None,
                           by_names: List[str] = None):
        return gather_step.gather_steps_like(prefix=None if by_type is None else self.get_prefix(by_type),
                                             step_names=by_names)

    def get_step(self, step: Type[BaseStep], step_id: Any = None) -> BaseStep:
        return self.steps[self.get_step_name(step, step_id)]

    def get_steps(self, ids: Iterable[Any], *steps_types: Type[BaseStep]) -> List[Tuple[BaseStep]]:
        return [tuple([self.get_step(t, i) for t in steps_types]) for i in ids]

    @classmethod
    def as_template_of(cls, pipeline_name: str, **kwargs):
        return type(pipeline_name, (cls, ), kwargs)

