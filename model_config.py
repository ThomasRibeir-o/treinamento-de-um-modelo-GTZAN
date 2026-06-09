from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import(
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
)

class ModelConfig:
    @staticmethod 
    def get_scaler():
        return [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler(), None]
    @staticmethod
    def get_grid():
        scalers = ModelConfig.get_scaler()
        return [
            {
                "clf": [SVC(random_state=89, class_weight="balanced")],
                "clf__C": [0.1, 1, 5, 10],#entortamento de reta
                "clf__kernel": ["rbf",],#tirei o poly
                "clf__gamma": ["scale", "auto"],
                "scaler": scalers,
            },
            {
                "clf": [LinearSVC(random_state=89, class_weight="balanced")],
                "clf__C": [0.1, 1, 5, 10, 50, 100],#entortamento de reta
                "clf__max_iter": [1000],# bastante, mas cuidado com o overfeeting
                "scaler": scalers,
            },
            {
                "clf": [KNeighborsClassifier()],
                "clf__n_neighbors": [3, 5, 11],#numero de vizinhos, nao colocar par pois pode empatar e dar ruim# antes estava"clf__n"
                "clf__weights": ['uniform', 'distance'],
                "scaler": scalers,
            },
            {
                "clf": [RandomForestClassifier(random_state=89, class_weight="balanced")],
                "clf__max_depth":[None, 10, 20],#none: os nós expandem ate onde achar mlr
                "scaler": [None], #random forest nao precisa de scaler pois é um monte de if
            },
            {
                "clf": [MLPClassifier(random_state=89)],
                "clf__hidden_layer_sizes":[(32,), (64,), (128,)],#quantos neuronios tem em cada camada
                "clf__learning_rate_init":[0.0008],#velocidade que ele aprende
                "scaler": scalers,
            }
        
        ]