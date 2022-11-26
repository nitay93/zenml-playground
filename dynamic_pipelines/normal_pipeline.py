import numpy as np
from zenml.pipelines import pipeline
from zenml.steps import step, Output


@pipeline
def first_pipeline(step_1, step_2):
    X_train, X_test, y_train, y_test = step_1()
    step_2(X_train, y_train)


@step
def get_data() -> Output(X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray):
    return np.zeros(5), np.zeros(1), np.zeros(5), np.zeros(1)


@step
def train_data(X_train: np.ndarray, y_train: np.ndarray) -> None:
    print(f"Training data {X_train}, {y_train}")


if __name__ == '__main__':
    first_pipeline(step_1=get_data(), step_2=train_data()).run()