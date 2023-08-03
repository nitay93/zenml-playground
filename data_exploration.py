import pandas as pd
from zenml import step, pipeline

from hyperparameter_tunning.steps.data.load_iris_data import get_iris_dataframe


@step
def explor_data(X: pd.DataFrame, y: pd.Series):
    print(f'Feature columns: {", ".join(X.columns)}')
    print(f'Number of samples: {len(X)}')
    print('Correlation between the features and the label:')
    print(X.corrwith(y))


@pipeline
def explor_iris_data():
    features, labels = get_iris_dataframe()
    explor_data(features, labels)


if __name__ == '__main__':
    explor_iris_data()
