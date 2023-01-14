import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

df = pd.read_excel('/home/ataur/Downloads/Dataset_Assignment Clustering/EastWestAirlines.xlsx')

df.info()
df.duplicated().sum()
df = df.drop_duplicates()
df.duplicated().sum()

df.isnull().sum()
df = df.dropna(how="any")
df.isnull().sum()

df1 = pd.get_dummies(df)

def Norm_func(i):
    x = (i-i.min()) / (i.max() - i.min())
    return(x)

df_norm = Norm_func(df1)
df_norm.describe()

y = linkage(df_norm, method='complete', metric='euclidean')

plt.figure(figsize=(20,8));plt.title('Clustering Dendogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(x, leaf_rotation=0, leaf_font_size=10)
plt.show()
 
# AgglomerativeClustering

Ag_cluster = AgglomerativeClustering(n_clusters=2, linkage="complete", affinity="euclidean").fit(df_norm)
Ag_cluster.labels_

cluster_labels = pd.Series(Ag_cluster.labels_)

df['clust'] = cluster_labels

df1 = df.iloc[:, [5,0,1,2,3,4]]
