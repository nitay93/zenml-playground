from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from zenml import step


@step
def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits train and test data into train and test with a ratio of 80%-20%"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
