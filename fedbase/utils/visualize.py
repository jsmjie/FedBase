from sklearn.manifold import TSNE
# from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

def dimension_reduction(data, label , method):
    x = data
    y = np.array(label)

    print(x.shape)
    print(y.shape)

    if method == 'tsne':
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x)
    elif method == 'pca':
        pca = decomposition.PCA(n_components=2)
        pca.fit(x)
        z = pca.transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["v1"] = z[:,0]
    df["v2"] = z[:,1]

    sns.set_theme(style="darkgrid")
    sns.scatterplot(x="v1", y="v2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y))),
                    data=df, s = 12)
    # plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    plt.legend().remove()
    plt.xlabel('v1', fontsize=16)
    plt.ylabel('v2', fontsize=16)
    plt.show()