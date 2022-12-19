import uuid
from abc import abstractmethod
from functools import partial
from typing import Any, Type, Iterable

from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep, BaseParameters


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


class StepsGenerator:
    """
    Abstract base class for steps generator for dynamic pipelines. Generates the steps for the pipeline
    """
    @abstractmethod
    def steps(self) -> Iterable[BaseStep]:
        """
        A collection of all steps generated by the concrete StepsGenerator
        Returns:
            A list of all steps generated
        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


# optional dynamic pipeline decorator
# def dynamic_pipeline(_pipeline_func=None, *, name: str = None):
#
#     def decorator(pipeline_func) -> Type[DynamicPipeline]:
#         class DecoratorDynamicPipeline(DynamicPipeline):
#             def connect_generator(self, steps_generator: StepsGenerator) -> None:
#                 pipeline_func(steps_generator)
#
#         DecoratorDynamicPipeline.__name__ = pipeline_func.__name__ if name is None else name
#         return DecoratorDynamicPipeline
#
#     if _pipeline_func is None:
#         return decorator
#     else:
#         return decorator(_pipeline_func)


class DynamicPipeline(BasePipeline):
    """
    An abstract dynamic pipeline class which is a base of dynamic pipelines.
    """
    def __init__(self, steps_generator: StepsGenerator):
        """
        Initializes the dynamic pipeline.
        Args:
            steps_generator: A StepsGenerator object with
        """
        self.steps_generator = steps_generator
        steps = self.steps_generator.steps()

        pipeline_type = type(self)
        if pipeline_type.STEP_SPEC != {}:
            raise RuntimeError(f"Steps for pipeline {pipeline_type} were already initialized"
                               f"Consider generating new pipelines based on this template with ...")
        pipeline_type.STEP_SPEC = {s.name: type(s) for s in steps}
        super().__init__(*steps)

    @abstractmethod
    def connect_generator(self, steps_generator: StepsGenerator) -> None:
        """
        Function that connects inputs and outputs of the pipeline steps, based on the StepsGenerator object.
        Args:
            steps_generator: The StepsGenerator object containing all pipeline steps.

        Raises:
            NotImplementedError: Always.
        """
        #
        raise NotImplementedError

    def connect(self, *args: BaseStep, **kwargs: BaseStep) -> None:
        self.connect_generator(self.steps_generator)

    @classmethod
    def clone(cls, pipeline_name: str, **kwargs):
        """
            Creates a new pipeline type, that directly inherits from the concrete dynamic pipeline.
            Args:
                pipeline_name: the name of the new pipeline
                **kwargs: a dictionary of arguments to add to the type constructor.

            Returns:
            A new pipeline type named according to the input name that directly inherits from the input pipeline.
            """
        return type(pipeline_name, (cls,), kwargs)
