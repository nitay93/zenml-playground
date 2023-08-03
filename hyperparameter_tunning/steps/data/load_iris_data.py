from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import datasets
from zenml import step


@step
def get_iris_dataframe() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris()
    return pd.DataFrame(iris.data, columns=iris.feature_names), pd.Series(iris.target, name='label')


@step
def get_iris_data_arrays() -> Tuple[np.ndarray, np.ndarray]:
    iris = datasets.load_iris()
    return iris.data, iris.target
