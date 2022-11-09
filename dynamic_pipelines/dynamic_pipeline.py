from abc import abstractmethod
from functools import partial
from typing import Any, List, Type

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep


class DynamicPipeline(BasePipeline):
    def __init__(self, **kwargs: Any):
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
    def get_step_name(step: Type[BaseStep], index: int = 0):
        return f"{step.__name__}_{index}"

    @classmethod
    def get_step(cls, step: Type[BaseStep], i: int = None, **kwargs):
        return kwargs[step.__name__] if i is None else kwargs[cls.get_step_name(step, i)]


def PipelineFactory(name, pipeline_template: Type[DynamicPipeline]):
    return type(name, (pipeline_template,), {})
