from dataclasses import dataclass
from itertools import chain
from typing import List, Type, Iterable

import numpy as np
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml.steps import BaseParameters, step, BaseStep, Output

from dynamic_pipelines.dynamic_pipeline import new_step, StepsGenerator, DynamicPipeline
from dynamic_pipelines.gather_step import OutputParameters, GatherStepsParameters


class RandomForestClassifierParameters(BaseParameters):
    n_estimators: int = 100


@step
def load_iris_data() -> Output(X=np.ndarray, y=np.ndarray):
    iris = datasets.load_iris()
    return iris.data[:, :3], iris.target


@step
def load_breast_cancer() -> Output(X=np.ndarray, y=np.ndarray):
    breast_cancer_data = datasets.load_breast_cancer()
    return breast_cancer_data.data, breast_cancer_data.target


@step
def split_data_step(X: np.ndarray, y: np.ndarray) -> Output(X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray,
                                                            y_test=np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


@step
def train_and_predict_rf_classifier(params: RandomForestClassifierParameters, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray) -> np.ndarray:
    clf = RandomForestClassifier(**params.__dict__)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


class TuningPhaseParam(BaseParameters):
    details: str


class EvaluationOutputParams(OutputParameters):
    score: float
    metric_name: str
    tuning_phase: str


@step
def calc_accuracy(param: TuningPhaseParam, y_test: np.ndarray,
                  y_pred: np.ndarray) -> EvaluationOutputParams.as_output():
    return metrics.accuracy_score(y_test, y_pred), "accuracy", param.details


class CompareScoreParams(GatherStepsParameters):
    reduce_min: bool = False
    reduce_max: bool = False


@step
def compare_score(params: CompareScoreParams) -> None:
    outputs = EvaluationOutputParams.extract(params)

    scores = []
    for output in outputs:
        print(f"For {output.tuning_phase} : {output.metric_name}={output.score}")
        scores.append((output.tuning_phase, output.score))

    if params.reduce_min:
        score, phase = min(scores, key=lambda x: x[0])
        print(f"minimal value at {phase}. score = {score}")

    if params.reduce_max:
        score, phase = max(scores, key=lambda x: x[0])
        print(f"maximal value at {phase}. score = {score}")


# class HyperParameterTuning(BasePipeline):
#
#     def __init__(self, load_data_step: Type[BaseStep], train_and_predict_step: Type[BaseStep],
#                  evaluate_step: Type[BaseStep], params_list: List[BaseParameters], **kwargs: Any):
#         self.load_data_step = load_data_step()
#         self.tuning_steps = [(new_step(split_data_step, step_id=self.param_id(param)),
#                               new_step(train_and_predict_step, step_id=self.param_id(param), parameters=param),
#                               new_step(evaluate_step, step_id=self.param_id(param),
#                                        parameters=TuningPhaseParam(details=str(param))))
#                              for param in params_list]
#
#         self.compare_scores = compare_score(
#             CompareScoreParams(reduce_max=True, output_steps_prefix=evaluate_step.__name__))
#
#         steps = [self.load_data_step, self.compare_scores, *chain.from_iterable(self.tuning_steps)]
#         register_steps(pipeline=type(self), steps=steps)
#         super().__init__(*steps, **kwargs)
#
#     @staticmethod
#     def param_id(parameters: BaseParameters):
#         return '_'.join([str(x) for x in parameters.dict().values()])
#
#     def connect(self, **steps: BaseStep) -> None:
#         X, y = self.load_data_step()
#         for split_data, train_and_predict, evaluate in self.tuning_steps:
#             X_train, X_test, y_train, y_test = split_data(X, y)
#             y_pred = train_and_predict(X_train, y_train, X_test)
#             evaluate(y_test, y_pred)
#             self.compare_scores.after(evaluate)
#
#         self.compare_scores()

@dataclass
class HyperParameterTuningSteps(StepsGenerator):
    load_data_step: Type[BaseStep]
    train_and_predict_step: Type[BaseStep]
    evaluate_step: Type[BaseStep]
    params_list: List[BaseParameters]

    def __post_init__(self):
        self.load_data = self.load_data_step()
        self.tuning_steps = [(new_step(split_data_step, step_id=self.param_id(param)),
                              new_step(self.train_and_predict_step, step_id=self.param_id(param), parameters=param),
                              new_step(self.evaluate_step, step_id=self.param_id(param),
                                       parameters=TuningPhaseParam(details=str(param))))
                             for param in self.params_list]

        self.compare_scores = compare_score(
            CompareScoreParams(reduce_max=True, output_steps_prefix=self.evaluate_step.__name__))

    @staticmethod
    def param_id(parameters: BaseParameters):
        return '_'.join([str(x) for x in parameters.dict().values()])

    def steps(self) -> Iterable[BaseStep]:
        return [self.load_data, self.compare_scores, *chain.from_iterable(self.tuning_steps)]


class HyperParameterTuningPipeline(DynamicPipeline):
    def connect_generator(self, steps: HyperParameterTuningSteps) -> None:
        X, y = steps.load_data()
        for split_data, train_and_predict, evaluate in steps.tuning_steps:
            X_train, X_test, y_train, y_test = split_data(X, y)
            y_pred = train_and_predict(X_train, y_train, X_test)
            evaluate(y_test, y_pred)
            steps.compare_scores.after(evaluate)
        steps.compare_scores()


if __name__ == '__main__':
    HyperParameterTuningPipeline.clone('iris_random_forest')(HyperParameterTuningSteps(
        load_data_step=load_iris_data,
        train_and_predict_step=train_and_predict_rf_classifier,
        evaluate_step=calc_accuracy,
        params_list=[RandomForestClassifierParameters(n_estimators=100),
                     RandomForestClassifierParameters(n_estimators=200)])).run(unlisted=True, enable_cache=False)
    HyperParameterTuningPipeline.clone('breast_cancer_random_forest')(HyperParameterTuningSteps(
        load_data_step=load_breast_cancer,
        train_and_predict_step=train_and_predict_rf_classifier,
        evaluate_step=calc_accuracy,
        params_list=[RandomForestClassifierParameters(n_estimators=100),
                     RandomForestClassifierParameters(n_estimators=200),
                     RandomForestClassifierParameters(n_estimators=300)])).run(unlisted=True, enable_cache=False)
