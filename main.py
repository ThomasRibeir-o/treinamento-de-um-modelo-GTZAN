from csv_getter import get_from_csv
from data_handler import DataHandler

path = "features_30_sec.csv"

data_handler = DataHandler(path)
X_train, X_test, y_train, y_test = data_handler.separate_X_y()