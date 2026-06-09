from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import(
    StandardScaler
)

from model_config import ModelConfig

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self):
        self.param_grid = ModelConfig.get_grid()
    
    def get_pipeline(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier()),
        ])
    
    def trainer(self,X,y):
        pipe_line = self.get_pipeline()
        grid = GridSearchCV(estimator = pipe_line, param_grid = self.param_grid, cv = 5, n_jobs=-1)
        grid.fit(X,y)
        return grid