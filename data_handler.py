from sklearn.model_selection import train_test_split
from csv_getter import get_from_csv
from typing import Tuple
import numpy as np

class DataHandler:

    def __init__(self, path: str) -> None:

        self.path = path

    # def featuresRemover(self, X):
    #     featuresPodi = ["spectral_centroid_mean", "spectral_bandwidth_mean", "rolloff_mean", "mfcc2_mean", "mfcc5_mean", "mfcc15_var", "zero_crossing_rate_var", "mfcc8_mean", "mfcc8_var", "mfcc10_mean", "mfcc10_var"]
    #     return X.drop(columns=featuresPodi)


    def separate_X_y(
        self, test_size: float = 0.2, random_state: int = 67
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X,y = get_from_csv(self.path)
        # X = self.featuresRemover(X)
        return train_test_split(
            X,y, test_size=test_size, random_state=random_state,stratify=y
        )