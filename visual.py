from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    color = ['c', 'g', 'y', 'r', 'k', 'b', 'm', 'grey', 'lightgray', 'lightcoral', 'lightcyan', 'lightyellow','lightsalmon']
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(14.94,7.58))
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=color[label[i].item()],s=20)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def visual(data,label,dataname,modelname):
    n_samples=data.shape[0]
    n_features = data.shape[1]
    print('label',label)
    print('label中数字有',len(set(label.tolist())),'个不同的数字')
    print('data有',n_samples,'个样本')
    print('每个样本',n_features,'维数据')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2)
    t0 = time()
    result = tsne.fit_transform(data)
    print('result.shape',result.shape)
    fig = plot_embedding(result, label,'t-SNE embedding of the digits (time %.2fs)'% (time() - t0))
    plt.savefig('C:/Users/l/Desktop/visual/'+dataname+'_'+modelname+'.png')

