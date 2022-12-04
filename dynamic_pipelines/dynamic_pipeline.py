from abc import abstractmethod
from functools import partial
from typing import Any, List, Type, Union, Tuple, Iterable

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep, BaseParameters


class DynamicPipeline(BasePipeline):
    def __init__(self, **kwargs: Any):
        self.steps_dict = {}
        self.steps_list = []
        self.init_steps()
        if type(self).STEP_SPEC != {}:
            raise RuntimeError(f"A dynamic pipeline {self.__class__.__name__} was already initialized. Consider using "
                               f"PipelineFactory to generate a new pipeline based on a pipeline template.")
        type(self).STEP_SPEC = {s.name: type(s) for s in self.steps_list}
        super().__init__(*self.steps_list, **kwargs)

    def init_step(self, step: Type[BaseStep], param: BaseParameters = None, name: Any = None):
        if step.__name__ not in self.steps_dict:
            self.steps_dict[step.__name__] = {}

        if name is None:
            index = 0 if self.steps_dict[step.__name__] == {} \
                else max([k for k in self.steps_dict[step.__name__].keys() if isinstance(k, int)]) + 1
            self._add_new_step(step, param, index)
        elif name not in self.steps_dict[step.__name__]:
            self._add_new_step(step, param, name)
        else:
            raise ValueError(f"step with {name} already exists")

    def _add_new_step(self, step: Type[BaseStep], param: BaseParameters = None, name: Any = None):
        step_name = self.get_step_name(step, name)
        self.steps_dict[step.__name__][name] = step(name=step_name) if param is None else step(param, name=step_name)
        self.steps_list.append(self.steps_dict[step.__name__][name])

    @abstractmethod
    def init_steps(self) -> None:
        pass

    # @classmethod
    # def create_step(cls, step: Type[BaseStep], step_id: Any = None):
    #     return partial(step, name=cls.get_step_name(step, step_id))

    @staticmethod
    def get_prefix(step: Type[BaseStep]):
        return step.__name__

    @classmethod
    def get_step_name(cls, step: Type[BaseStep], step_id: Any = 0):
        return cls.get_prefix(step) if step_id is None else f"{cls.get_prefix(step)}_{step_id}"

    def get_step(self, step: Type[BaseStep], step_id: Any = None) -> BaseStep:
        return self.steps[self.get_step_name(step, step_id)]

    def get_steps(self, ids: Iterable[Any], *steps_types: Type[BaseStep]) -> List[Tuple[BaseStep]]:
        return [tuple([self.get_step(t, i) for t in steps_types]) for i in ids]

    @classmethod
    def as_template_of(cls, pipeline_name: str, **kwargs):
        return type(pipeline_name, (cls, ), kwargs)

