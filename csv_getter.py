import pandas as pd

def get_from_csv(caminho_planilha):


    print(caminho_planilha)
    df = pd.read_csv(caminho_planilha)
    print(df.to_string())
    X = df.iloc[:, 1:-1].values

    y = df['label'].values

    return X,y