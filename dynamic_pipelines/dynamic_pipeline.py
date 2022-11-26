from abc import abstractmethod
from functools import partial
from typing import Any, List, Type, Union, Tuple, Iterable

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep


class DynamicPipeline(BasePipeline):
    def __init__(self, **kwargs: Any):
        # steps = self.initialize_steps()
        if type(self).STEP_SPEC != {}:
            raise RuntimeError(f"A dynamic pipeline {self.__class__.__name__} was already initialized. Consider using "
                               f"PipelineFactory to generate a new pipeline based on a pipeline template.")
        type(self).STEP_SPEC = {s.name: type(s) for s in self.dynamic_steps}
        super().__init__(*self.dynamic_steps, **kwargs)

    @property
    @abstractmethod
    def dynamic_steps(self) -> List[BaseStep]:
        return list(self.initialize_steps())

    @abstractmethod
    def initialize_steps(self) -> Iterable[BaseStep]:
        pass

    @classmethod
    def create_step(cls, step: Type[BaseStep], step_id: Any = None):
        return partial(step, name=cls.get_step_name(step, step_id))

    @staticmethod
    def get_step_name(step: Type[BaseStep], step_id: Any = None):
        return step.__name__ if step_id is None else f"{step.__name__}_{step_id}"

    def get_step(self, step: Type[BaseStep], step_id: Any = None) -> BaseStep:
        return self.steps[self.get_step_name(step, step_id)]

    def get_steps(self, ids: Iterable[Any], *steps_types: Type[BaseStep]) -> List[Tuple[BaseStep]]:
        return [tuple([self.get_step(t, i) for t in steps_types]) for i in ids]

    @classmethod
    def as_template_of(cls, pipeline_name: str, **kwargs):
        return type(pipeline_name, (cls, ), kwargs)

