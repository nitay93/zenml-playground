import uuid
from functools import partial
from typing import Any, Type, TypeVar

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep, BaseParameters


DP = TypeVar("DP", bound="DynamicPipeline")


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
    named_step = partial(step, name=get_step_name(step, name))
    return named_step() if parameters is None else named_step(parameters)


def get_prefix(step: Type[BaseStep]) -> str:
    """
    The prefix of the names that are generated for specific step types.
    Args:
        step: The type of the step.

    Returns:
        The steps name prefix
    """
    return step.__name__


def get_step_name(step: Type[BaseStep], step_id: Any = None) -> str:
    """
    Generates a name for a step based on the step type and a step_id.
    Args:
        step: the step type.
        step_id: the step id.

    Returns:
        The name of the step.
    """
    return get_prefix(step) if step_id is None else f"{get_prefix(step)}_{step_id}"


class DynamicPipeline(BasePipeline):
    """Abstract class for dynamic ZenML pipelines, enabling creation of pipeline templates without predefining
    the exact number of steps the pipeline can depend on.
    """

    def __init__(self, *steps: BaseStep, **kwargs: Any) -> None:
        """
        Initializes the dynamic pipeline
        Args:
            *steps: the steps to be executed by this pipeline
            **kwargs: the configuration of this pipeline
        """
        if type(self).STEP_SPEC != {}:
            raise RuntimeError(f"A dynamic pipeline {self.__class__.__name__} was already initialized. "
                               f"Consider generating new pipelines based on this template with "
                               f"{self.__class__.__name__}.{self.as_template_of.__name__}()")
        type(self).STEP_SPEC = {s.name: type(s) for s in steps}
        super().__init__(*steps, **kwargs)

    def connect(self, **kwargs: Any) -> None:
        """Function that connects inputs and outputs of the pipeline steps.

        Args:
            **kwargs: The keyword arguments passed to the pipeline.
        """
        super().connect(*self.steps, **kwargs)

    @classmethod
    def as_template_of(cls: Type[DP], pipeline_name: str, **kwargs: Any) -> Type[DP]:
        """
        Generates a new type of pipeline the directly inherits from the current dynamic pipeline.
        This is useful to create multiple dynamic pipelines based on dynamic pipeline class.
        Args:
            pipeline_name: The name of the new pipeline instance.
            **kwargs: dictionary for the type constructor.

        Returns:
            The new pipeline instance generated.
        """
        return type(pipeline_name, (cls,), kwargs) # noqa
