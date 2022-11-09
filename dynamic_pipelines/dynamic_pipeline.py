from abc import abstractmethod
from functools import partial
from typing import Any, List, Type

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep


class DynamicPipeline(BasePipeline):
    def __init__(self, **kwargs: Any):
        if type(self).STEP_SPEC != {}:
            raise RuntimeError(f"A dynamic pipeline {self.__class__.__name__} was already initialized. Consider using "
                               f"PipelineFactory to generate a new pipeline based on a pipeline template.")
        type(self).STEP_SPEC = {s.name: type(s) for s in self.dynamic_steps}
        super().__init__(*self.dynamic_steps, **kwargs)

    @property
    @abstractmethod
    def dynamic_steps(self, *args, **kwargs) -> List[BaseStep]:
        pass

    @classmethod
    def create_indexed_step(cls, step: Type[BaseStep], step_id):
        return partial(step, name=cls.get_step_name(step, step_id))

    @staticmethod
    def get_step_name(step: Type[BaseStep], index: int = None):
        return step.__name__ if index is None else f"{step.__name__}_{index}"

    def get_step(self, step: Type[BaseStep], index: int = None):
        return self.steps[self.get_step_name(step, index)]


# Since the STEP_SPEC is a class field, each pipeline containing different number of steps needs its own class
# this factory method define a new pipeline class that inherits all its functionality from the parent pipeline_template
def PipelineFactory(name, pipeline_template: Type[DynamicPipeline]):
    return type(name, (pipeline_template,), {})
