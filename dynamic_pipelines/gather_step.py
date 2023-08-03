from typing import List, TypeVar, Type

from zenml.environment import Environment
from zenml.post_execution import get_run
from zenml.steps import Output, BaseParameters

GP = TypeVar("GP", bound="GatherParameters")


class GatherStepsParameters(BaseParameters):
    """
    Base Parameters class for steps that gather step outputs from an undetermined number of steps.
    The attributes of this class will be used to filter relevant steps.
    Attributes:
        output_steps_prefix: a prefix of step names
        output_steps_names: a list of step names
    """
    output_steps_prefix: str = None
    output_steps_names: List[str] = None


class OutputParameters(BaseParameters):
    """
    A base class for a parameters class to extract the output parameters and value of a step.
    The attributes of the inheriting concrete class should be equal to the output fields of a step.
    These output values of steps with these output fields, can be extracted using the method of this class.
    """
    @staticmethod
    def extract_dictionaries(gather_steps_params: GatherStepsParameters) -> List[dict]:
        """
        Extract output of steps compatible to the parameters of gather_steps_params.
        Args:
            gather_steps_params: The parameters object, to filter the relevant steps.

        Returns:
            A list of dictionaries of the outputs of the steps.
        """
        run_name = Environment().step_environment.run_name
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
    def extract(cls: Type[GP], gather_steps_params: GatherStepsParameters) -> List[GP]:
        """
        Extract output of steps compatible to the parameters of gather_steps_params, and generates an instance of
        the concrete output class given by cls. The assumption is that the concrete class's constructor accepts all the
        entries of the output dictionaries of the requested steps.
        Args:
            gather_steps_params: The parameters object, to filter the relevant steps.

        Returns:
            A list of the objects of the output concrete class with the output values of the relevant steps.
        """
        return [cls(**o) for o in cls.extract_dictionaries(gather_steps_params)]

    @classmethod
    def as_output(cls) -> Output:
        """
        Converts the concrete class into a ZenML Output object, so that, the user will not have to specify the same
        parameters to create both the OutputParameters object for the gather step and the ZenML Output object for the
        return type of the steps to be gathered.
        Returns:
            A ZenML Output object with the same fields and type values as the concrete class.
        """
        return Output(**{field_name: field.type_ for field_name, field in cls.__fields__.items()})
