from itertools import chain
from typing import List, Any, Type

import numpy as np
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml.steps import BaseParameters, step, BaseStep, Output

from dynamic_pipelines.dynamic_pipeline import DynamicPipeline
from dynamic_pipelines.gather_step import gather_step, GatherParameters


class RandomForestClassifierParameters(BaseParameters):
    n_estimators: int = 100


@step
def load_iris_data() -> Output(X=np.ndarray, y=np.ndarray):
    iris = datasets.load_iris()
    return iris.data[:, :3], iris.target


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
    name: str


class EvaluationOutputParams(GatherParameters):
    score: float
    metric_name: str
    tuning_phase: str


evaluation_output = EvaluationOutputParams.as_output()


@step
def calc_accuracy(param: TuningPhaseParam, y_test: np.ndarray, y_pred: np.ndarray) -> evaluation_output:
    return metrics.accuracy_score(y_test, y_pred), "accuracy", param.name


class ReduceScoreParams(BaseParameters):
    reduce_min: bool = False
    reduce_max: bool = False


@gather_step(gather_outputs_of_type=EvaluationOutputParams)
def compare_score_gather_step(params: ReduceScoreParams, outputs: List[EvaluationOutputParams]) -> None:
    scores = []
    for output in outputs:
        print(f"For {output.score} : {output.metric_name}={output.score}")
        scores.append((output.score, output.tuning_phase))

    if params.reduce_min:
        score, phase = min(scores, key=lambda x: x[0])
        print(f"minimal value at {phase}. score = {score}")

    if params.reduce_max:
        score, phase = max(scores, key=lambda x: x[0])
        print(f"maximal value at {phase}. score = {score}")


class HyperParameterTuning(DynamicPipeline):

    def __init__(self, load_data_step: Type[BaseStep], train_and_predict_step: Type[BaseStep],
                 evaluate_step: Type[BaseStep], params_list: List[BaseParameters], **kwargs: Any):
        self.load_data_step = load_data_step()
        self.tuning_steps = [(self.new_step(split_data_step),
                              self.new_step(train_and_predict_step, param=param),
                              self.new_step(evaluate_step, param=TuningPhaseParam(name=param.json())))
                             for param in params_list]

        compare_scores_step = self.define_gather_step(compare_score_gather_step, by_type=evaluate_step)
        self.compare_scores = compare_scores_step(ReduceScoreParams(reduce_max=True))

        super().__init__(self.load_data_step,
                         self.compare_scores,
                         *chain.from_iterable(self.tuning_steps),
                         **kwargs)

    def connect(self, **steps: BaseStep) -> None:
        X, y = self.load_data_step()
        for split_data, train_and_predict, evaluate in self.tuning_steps:
            X_train, X_test, y_train, y_test = split_data(X, y)
            y_pred = train_and_predict(X_train, y_train, X_test)
            evaluate(y_test, y_pred)
            self.compare_scores.after(evaluate)

        self.compare_scores()


if __name__ == '__main__':
    HyperParameterTuning(
        load_data_step=load_iris_data,
        train_and_predict_step=train_and_predict_rf_classifier,
        evaluate_step=calc_accuracy,
        params_list=[RandomForestClassifierParameters(n_estimators=100),
                     RandomForestClassifierParameters(n_estimators=200)]).run()
