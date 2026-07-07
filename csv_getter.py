import pandas as pd

def get_from_csv(caminho_planilha):


    df = pd.read_csv(caminho_planilha)

    X = df.iloc[:, 1:-1]

    y = df['label'].values

    return X,y