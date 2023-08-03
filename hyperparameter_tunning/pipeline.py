import itertools
from typing import Any

from zenml import pipeline, step

from hyperparameter_tunning.steps.classifier.random_forest_classifier import train_and_predict_rf_classifier
from hyperparameter_tunning.steps.data.load_iris_data import get_iris_data_arrays
from hyperparameter_tunning.steps.data.split_data import split_data
from hyperparameter_tunning.steps.evaluation.accuracy import calc_accuracy
from hyperparameter_tunning.steps.evaluation.compare_scores import compare_score


@step
def print_value(cls: Any):
    print(cls)


@pipeline
def hyperparameter_tuning_compare_models(configs):
    X, y = get_iris_data_arrays()
    X_train, X_test, y_train, y_test = split_data(X, y)
    evaluation_steps = []
    for i, config in enumerate(configs):
        y_pred = train_and_predict_rf_classifier(X_train,
                                                 y_train,
                                                 X_test,
                                                 random_forest_config=config)

        evaluation_step_id = f"evaluation_{i + 1}"
        calc_accuracy(model_parameters=config,
                      y_test=y_test,
                      y_pred=y_pred,
                      id=evaluation_step_id)
        evaluation_steps.append(evaluation_step_id)

    compare_score(evaluation_steps=evaluation_steps, after=evaluation_steps)


@pipeline
def hyperparameter_tuning(configs):
    X, y = get_iris_data_arrays()
    X_train_val, X_test, y_train_val, y_test = split_data(X, y)
    evaluation_steps = []
    for i, config in enumerate(configs):
        X_train, X_val, y_train, y_val = split_data(X_train_val, y_train_val)
        y_pred = train_and_predict_rf_classifier(X_train,
                                                 y_train,
                                                 X_val,
                                                 random_forest_config=config)

        evaluation_step_id = f"evaluation_{i + 1}"
        calc_accuracy(model_parameters=config,
                      y_test=y_val,
                      y_pred=y_pred,
                      id=evaluation_step_id)
        evaluation_steps.append(evaluation_step_id)

    best_hyperparameter_config = compare_score(evaluation_steps=evaluation_steps, after=evaluation_steps)

    y_pred = train_and_predict_rf_classifier(
        X_train_val,
        y_train_val,
        X_test,
        random_forest_config=best_hyperparameter_config)
    calc_accuracy(model_parameters=best_hyperparameter_config,
                  y_test=y_test,
                  y_pred=y_pred,
                  final_performance=True)


if __name__ == '__main__':
    estimators = range(100, 300, 100)
    max_depths = range(1, 4)
    criterions = ['gini', 'log_loss']
    hyperparameter_configs = [{
        'n_estimators': estimator,
        'max_depth': max_depth,
        'criterion': criterion
    } for
        estimator,
        max_depth,
        criterion in itertools.product(estimators,
                                       max_depths,
                                       criterions)]

    hyperparameter_tuning(hyperparameter_configs)
