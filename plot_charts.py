import numpy as np
import matplotlib.pyplot as plt

def plot_balanceamento(y):
    classes, contagem = np.unique(y, return_counts=True)
    plt.bar(classes, contagem)
    plt.xlabel("classes")
    plt.ylabel("contagem")
    plt.title("balanceamento")
    #plt.xticks("classes")
    plt.show()