from csv_getter import get_from_csv
from data_handler import DataHandler
from model_trainer import ModelTrainer
from display_metrics import display_values

path = "features_30_sec.csv"

data_handler = DataHandler(path)
X_train, X_test, y_train, y_test = data_handler.separate_X_y()#test_size: float = 0.2, random_state: int = 67
model_trainer = ModelTrainer()
grid = model_trainer.trainer(X_train, y_train)
display_values(grid = grid, X_test = X_test, y_test = y_test)
