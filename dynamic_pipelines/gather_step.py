import inspect
from typing import List, Dict, Callable, Type, Union, Optional, Iterable, TypeVar, Any

from pydantic import BaseModel
from zenml.environment import Environment
from zenml.post_execution import get_run
from zenml.steps import BaseStep, step, Output, BaseParameters


F = TypeVar("F", bound=Callable[..., Any])


class GatherParameters(BaseModel):
    @classmethod
    def as_output(cls) -> Output:
        return Output(**{field_name: field.type_ for field_name, field in cls.__fields__.items()})


def gather_by(step_name_prefix: str = None, step_names: List[str] = None) -> List[Dict]:
    run_name = Environment().step_environment.pipeline_run_id
    run = get_run(run_name)
    prefix_steps = [
        s for s in run.steps
        if s.name.startswith(step_name_prefix)
    ] if step_name_prefix is not None else []

    named_steps = [
        s for s in run.steps
        if s.name in step_names
    ] if step_names is not None else []

    return [
        {k: v.read() for k, v in s.outputs.items()} for s in prefix_steps + named_steps
    ]


class GatherSteps:
    def __init__(self, func: F, gather_outputs_of_type: Type[GatherParameters] = None):
        self.func = func
        annotations = {**inspect.getfullargspec(inspect.unwrap(func)).annotations}
        self._return_type = annotations.pop('return', None)
        self._gather_parameters_type = gather_outputs_of_type

        self._parameters_type = None
        for param_name, param_type in annotations.copy().items():
            if inspect.isclass(param_type) and issubclass(param_type, BaseParameters):
                if self._parameters_type is not None:
                    raise ValueError("TODO")
                self._parameters_type = annotations.pop(param_name)

        if len(annotations) > 1:
            raise ValueError("TODO")

        for param_type in annotations.values():
            if param_type != List[gather_outputs_of_type] and param_type != Iterable[gather_outputs_of_type]:
                raise ValueError("TODO")

    def gather_steps_like(self, prefix: str = None, step_names: List[str] = None) -> Type[BaseStep]:
        param_type = self._parameters_type
        return_type = self._return_type

        def step_function(params: param_type) -> return_type:
            outputs = gather_by(step_name_prefix=prefix, step_names=step_names)
            if self._gather_parameters_type is not None:
                outputs = [self._gather_parameters_type(**o) for o in outputs]
            return self.func(params, outputs)

        step_function.__name__ = self.func.__name__

        return step(step_function)


def gather_step(
        _func: Optional[F] = None,
        *,
        gather_outputs_of_type: Optional[Type[GatherParameters]] = None
) -> Union[GatherSteps, Callable[[F], GatherSteps]]:
    def inner_decorator(func) -> GatherSteps:
        return GatherSteps(func, gather_outputs_of_type)

    if _func is None:
        return inner_decorator
    else:
        return inner_decorator(_func)
