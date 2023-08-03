from functools import partial

import numpy as np
from pydantic import BaseModel
from sklearn import metrics
from zenml import step


class EvaluationResult(BaseModel):
    metric: str
    score: float
    model_parameters: dict


accuracy_result = partial(EvaluationResult, metric="accuracy")


@step
def calc_accuracy(
        model_parameters: dict, y_test: np.ndarray, y_pred: np.ndarray, final_performance=False
) -> EvaluationResult:
    """Calculates the accuracy of the prediction."""
    score = metrics.accuracy_score(y_test, y_pred)

    if final_performance:
        print(
            f"The chosen hyperparameters are: {model_parameters}. \n"
            f"The final accuracy score of the chosen model is: {score * 100:.2f}%"
        )
    else:
        print(
            f"model with parameters: {model_parameters} scored {score * 100:.2f}% "
            "accuracy"
        )
    return accuracy_result(score=score, model_parameters=model_parameters)
