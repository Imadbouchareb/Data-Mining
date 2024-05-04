"""
Clustering hiérarchique
Clustering :

Diviser un ensemble d'objets en groupes (clusters)
Maximiser la similarité intra-clusters et minimiser la similarité inter-clusters
Le clustering est une tâche non-supervisée
Clustering hiérarchique ascendant :

Initialisation avec un cluster par objet
A chaque étape deux clusters sont fusionnés
Coupure de l'arbre pour obtenir une partition

Caclul de distance
Distance euclidienne 
Nécessite une mise à l'échelle des propriétés (age vs. salaire)
Plusieurs techniques de mise à l'echelle existent (e.g. MinMax :  z = x−min(x)/max(x)−min(x) )
"""

import pandas as pd
import math

clients = {'Client': ['Marie', 'Bruno', 'Laurent'], 'Age': [24,31,27], 'Salary': [1800,2500,1500]}
df_clients = pd.DataFrame(data=clients)
print(df_clients)
print('\n')

df_clients[['Age','Salary']] = df_clients[['Age','Salary']].apply(lambda x:(x-x.min()) / (x.max()-x.min()))
print(df_clients.round(2))

matDist = np.empty([3, 3])
for index, row in df_clients[['Age','Salary']].iterrows():
  for index2, row2 in df_clients[['Age','Salary']].iterrows():
    matDist[index,index2] = math.sqrt((row['Age'] - row2['Age'])**2 + (row['Salary'] - row2['Salary'])**2)
print(np.round(matDist,2))

"""
Exercice : Lien minimum
Agrégation selon le lien minimum :
"""

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt

distMatrix = [[0,23,35,43,50],
              [23,0,21,32,45],
              [35,21,0,11,25],
              [43,32,11,0,17],
              [50,45,25,17,0]]

distArray = ssd.squareform(distMatrix)
linked = linkage(distArray, 'single')
labelList = ['a','b','c','d','e']

dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

"""
Exercice : Lien maximum
Agrégation selon le lien maximum :
"""

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt

distMatrix = [[0,23,35,43,50],
              [23,0,21,32,45],
              [35,21,0,11,25],
              [43,32,11,0,17],
              [50,45,25,17,0]]

distArray = ssd.squareform(distMatrix)
linked = linkage(distArray, 'complete')
labelList = ['a','b','c','d','e']

dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

"""
Exercice : Lien minimum
Agrégation selon le lien minimum :
"""

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt

distMatrix = [[0,662,877,255,412,996],
              [662,0,295,468,268,400],
              [877,295,0,754,564,138],
              [255,468,754,0,219,869],
              [412,268,564,219,0,669],
              [996,400,138,869,669,0]]

distArray = ssd.squareform(distMatrix)
linked = linkage(distArray, 'single')
labelList = ['BA','FI','MI','NA','RM','TO']

dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

"""
Exercice : Données countries
Calculer un clustering hiérarchique à l'aide des fonctions linkage et dendrogram du package scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# importation des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\countries.csv", header=0)

data = df.iloc[:,1:]
lbl = df['Country'].tolist()

# calcul du clustering hiérarchique
fig, ax = plt.subplots()
data_dist = pdist(data,'euclidean') 
data_link = linkage(data_dist,method='complete')

# affichage
dendrogram(data_link,labels=lbl,leaf_rotation=90)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Complete', fontweight='bold', fontsize=14)

#Scikit-learn
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X)  

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Result of hierarchical clustering')
plt.show()

"""
Exercice : Variation du nombre de clusters
A l'aide de Scikit-learn et de ses fonctions metrics, calculer l'évolution de l'Adjusted Rand index en fonction du nombre de clusters (de 1 à 20) pour les données Iris.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

# boucle sur le nombre de clusters
rand = []
for i in range(1,21):
    cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='average')  
    cluster.fit_predict(X)  
    rand.append(metrics.adjusted_rand_score(Y, cluster.labels_))
    
# affichage du résultat
plt.xticks(np.arange(1, 21, 1.0))
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted Rand Index')
plt.plot(np.arange(1, 21, 1.0),rand)
plt.axvline(x=3,c='black',ls='-.',lw=0.8)
plt.show()

"""
Conclusion :
Avantages

Facile à comprendre
Permet de facilement faire varier le nombre de clusters
Visualisation intuitive
Inconvénients

Coût du calcul de la matrice distance
Le critère de regroupement dépends des groupes déjà construits
Résultats différents en fonction du critère de regroupement
"""