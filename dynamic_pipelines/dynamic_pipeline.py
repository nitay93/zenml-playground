import uuid
from abc import abstractmethod
from functools import partial
from typing import Any, Type, Iterable

from pydantic import BaseModel
from zenml.pipelines import BasePipeline
from zenml.pipelines.base_pipeline import PIPELINE_INNER_FUNC_NAME
from zenml.steps import BaseStep, BaseParameters


def register_steps(pipeline: Type[BasePipeline], steps: Iterable[BaseStep]):
    """
    Registers the steps to the pipeline class with initialization of STEP_SPEC.
    Args:
        pipeline: the pipeline type to register the steps to
        steps: a list of steps to register to the pipeline

    """
    if pipeline.STEP_SPEC != {}:
        raise RuntimeError(f"Steps for pipeline {pipeline} were already initialized"
                           f"Consider generating new pipelines based on this template with ...")
    pipeline.STEP_SPEC = {s.name: type(s) for s in steps}


def new_step(step: Type[BaseStep], step_id: Any = None, parameters: BaseParameters = None) -> BaseStep:
    """
    Creates a new step with modified name. Useful to create multiple steps of the same type with different names,
    so that they can be used in the same pipeline.
    Args:
        step: the type of the new step to create.
        step_id: an id to be used to create a new name for the step. If not provided, generates a random uuid.
        parameters: optional parameters object for the step initialization.

    Returns:
        The new step instance generated.
    """
    name = uuid.uuid1().hex if step_id is None else step_id
    named_step = partial(step, name=step.__name__ if name is None else f"{step.__name__}_{name}")
    return named_step() if parameters is None else named_step(parameters)


def clone_pipeline(pipeline: Type[BasePipeline], name: str, **kwargs):
    """
    Clones a pipeline type. Useful for dynamic pipeline for which the steps specification is determined only at
    initialization.
    Args:
        pipeline: the type of pipeline to be cloned.
        name: the name of the new pipeline
        **kwargs: a dictionary of arguments to add to the type constructor.

    Returns:
    A new pipeline type named by according to the input name that directly inherits from the input pipeline.
    """
    return type(name, (pipeline,), kwargs)


class DynamicSteps:
    @abstractmethod
    def steps(self):
        pass
