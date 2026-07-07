from csv_getter import get_from_csv
from data_handler import DataHandler
from model_trainer import ModelTrainer
from display_metrics import display_values
import warnings
from plot_charts import plot_balanceamento
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.inspection import permutation_importance

path = "features_30_sec.csv"

warnings.filterwarnings("ignore")

data_handler = DataHandler(path)
def trainer():
    X_train, X_test, y_train, y_test = data_handler.separate_X_y()#test_size: float = 0.2, random_state: int = 67
    model_trainer = ModelTrainer()
    grid = model_trainer.trainer(X_train, y_train)
    display_values(grid = grid, X_test = X_test, y_test = y_test)



def trainer_classifier(model, X_train, y_train, X_test, y_test):
    y_pred_treino = model.predict(X_train)
    y_pred_teste = model.predict(X_test)
    acc_treino = accuracy_score(y_train, y_pred_treino)
    acc_teste = accuracy_score(y_test, y_pred_teste)
    valor_minimo = 0.75 #accuracy minima (identificar underfitting)
    max_dif  = 0.15 #diferencia max entre o treino e o teste #ele nao pode ser muito melhor no treino em comparacao ao teste (identificar overfitting)
    result = ""
    if acc_treino >= valor_minimo and acc_teste < valor_minimo:
        result = "overfetting"
    elif acc_treino < valor_minimo and acc_teste < valor_minimo:
        result = "underfetting"
    elif (acc_treino - acc_teste) > max_dif:
        result = "overfetting leve" 
    else:
        result = "bom"
    print(f"o modelo foi caracterizado como {result}")

def plot_matriz_confusao(y_val, y_predicao, class_names, dataset_name):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_val, y_predicao)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def pegar_nomes_das_features(path):#retorna os nomes das features
    df= pd.read_csv(path)
    feature_names = df.columns[1:-1].tolist()  # Pegando os nomes das colunas excluindo a ultima

    return feature_names

def plot_features(X, y, path, model):
    feature_names = pegar_nomes_das_features(path)
    #X_df = pd.DataFrame(X, columns = feature_names)
    model_random_florest = RandomForestClassifier(n_jobs = -1)
    model_random_florest.fit(X, y)
    model_logistic_regression = LogisticRegression(n_jobs = -1)
    model_logistic_regression.fit(X,y)
    rf_feature_importances = model_random_florest.feature_importances_#pegar as melhores features
    if len(model_logistic_regression.coef_.shape) == 2 and model_logistic_regression.coef_.shape[0] > 1:
        lr_coefficients = np.mean(np.abs(model_logistic_regression.coef_), axis=0)
    else:
        lr_coefficients = np.abs(model_logistic_regression.coef_.flatten())#se a base fosse binaria
    #deixar o coeficiente positivo e depois faz a media de todos os casos
    #quanto maior o coeficiente do logistc regression melhor a coluna é
    result = permutation_importance(model, X, y, n_repeats=6, random_state=67, n_jobs=-1)
    df_importance = pd.DataFrame({
        'feature': feature_names, 
        'importance_features' : result.importances_mean
    })
    
    # Permutation
    fig, axes = plt.subplots(1,3, figsize = (18,6))
    perm_imp = df_importance.set_index('feature')['importance_features']#setando qual feature pertence a qual importacia
    axes[0].barh(feature_names, perm_imp.values, color='steelblue')
    axes[0].set_xlabel('Queda na Acurácia')#quanto maior a queda, mais importante
    axes[0].set_title('Permutation Importance', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', linestyle=':', alpha=0.6)

    # Random Forest Feature Importances
    axes[1].barh(feature_names, rf_feature_importances, color='darkorange')
    axes[1].set_xlabel('Importância (Gini Impurity Decrease)')
    axes[1].set_title('Random Forest Native', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)

    # Logistic Regression Coefficients
    axes[2].barh(feature_names, lr_coefficients, color='forestgreen')
    axes[2].set_xlabel('Média dos Coeficientes Absolutos')
    axes[2].set_title('Logistic Regression Coefficients', fontsize=12, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', linestyle=':', alpha=0.6)

    plt.suptitle('Comparação de Importância de Features - GTZAN', 
                    fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# def featuresRemove(X):
#     featuresPodi = ["spectral_centroid_mean", "spectral_banwidth_mean", "rolloff_mean", "mfcc_2mean", "mfcc5_mean", "mfcc15_var", "zero_crossing_rate_var", "mfcc8_mean", "mfcc8_var", "mfcc10_mean", "mfcc10_var"]
#     return X.drop(columns=featuresPodi)

def features():
    X,y = get_from_csv(path)
    #spectral_centroid_mean, spectral_banwidth_mean, rolloff_mean, mfcc_2mean, mfcc5_mean, mfcc15_var, zero_crossing_rate_var, mfcc8_mean, mfcc8_var, mfcc10_mean, mfcc10_var
    plot_balanceamento(y)
    X_train, X_test, y_train, y_test = data_handler.separate_X_y()#test_size: float = 0.2, random_state: int = 67
    model_trainer = ModelTrainer()
    grid = model_trainer.trainer(X_train, y_train)
    trainer_classifier(grid.best_estimator_, X_train, y_train, X_test, y_test)
    y_pred_test = grid.predict(X_test)#predictando os valores do y
    plot_matriz_confusao(y_test, y_pred_test, class_names=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], dataset_name = "GTZAN")
    plot_features(X, y, path, grid)

#pip install -r requirements.txt
#cd .\treinamento-de-um-modelo-GTZAN\
#python controller.py


features()
#trainer()

