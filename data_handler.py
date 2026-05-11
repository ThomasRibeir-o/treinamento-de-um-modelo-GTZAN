from sklearn.model_selection import train_test_split
from csv_getter import get_from_csv
from typing import Tuple
import numpy as np

class DataHandler:

    def __init__(self, path: str) -> None:

        self.path = path

    def separate_X_y(
        self, test_size, random_state: int = 67
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X,y = get_from_csv(self.path)
        return train_test_split(
            X,y, test_size=test_size, random_state=random_state,stratify=y
        )