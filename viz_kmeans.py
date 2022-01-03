import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

def make_blob():
    X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.7, random_state=0)
    plt.figure(figsize = (10,6))
    plt.scatter(X[:, 0], X[:, 1], s=30, color = 'gray')
    return plt.show()

def viz_iris(iris, kmeans):
    
    centroids = np.array(iris.groupby('cluster')['petal_length', 'sepal_length'].mean())
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]
    # cen_x = [i[0] for i in kmeans.cluster_centers_]
    # cen_y = [i[1] for i in kmeans.cluster_centers_]
    iris['cen_x'] = iris.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    iris['cen_y'] = iris.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

    colors = ['#DF2020','#2095DF', '#81DF20' ]
    iris['c'] = iris.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    #plot scatter chart for Actual species and those predicted by K - Means

    #specify custom palette for sns scatterplot
    colors1 = ['#2095DF','#81DF20' ,'#DF2020']
    customPalette = sns.set_palette(sns.color_palette(colors1))

    #plot the scatterplots

    #Define figure (num of rows, columns and size)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # plot ax1 
    ax1 = plt.subplot(2,1,1) 
    sns.scatterplot(data = iris, x = 'petal_length', y = 'sepal_length', ax = ax1, hue = 'species', palette=customPalette)
    plt.title('Actual Species')

    #plot ax2
    ax2 = plt.subplot(2,1,2) 
    ax2.scatter(iris.petal_length, iris.sepal_length, c=iris.c, alpha = 0.6, s=10)
    ax2.set(xlabel = 'petal_length', ylabel = 'sepal_length', title = 'K - Means')

    # plot centroids on  ax2
    ax2.scatter(cen_x, cen_y, marker='X', c=colors, s=200)
    
    
    iris.drop(columns = ['cen_x', 'cen_y', 'c'], inplace = True)
    plt.tight_layout()
    plt.show()