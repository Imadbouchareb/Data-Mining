"""
KMeans
Méthode des K-Moyennes (K-means)
Construit une partition et la corrige jusqu'à obtenir des groupes satisfaisant
Nécessite le calcul d'une moyenne (pas toujours trivial)

Exercice : Commerce en ligne
Appliquer K-Means afin d'étudier les clients d'un site de commerce
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy as np

data = [[2,1],[1,1],[2,2],[1,2],[6,6],[5,5],[5,6],[5,3],[5,2],[4,2]]

df = pd.DataFrame(data)
plt.scatter(df.iloc[:,0],df.iloc[:,1])
plt.xlabel('Nombre de visites')
plt.ylabel('Nombre d\'achats')

C1 = [1.5,3]
C2 = [4.0,0.5]
C3 = [2.5, 5.0]

clusters = np.array([C1,C2,C3])

for c in clusters:
  plt.scatter(c[0],c[1], marker='*', s=128)


affect = []

for d in data:
  minDist = float('Inf')
  index = -1
  for i,c in enumerate(clusters):
    dist = math.sqrt((d[0]-c[0])**2 + (d[1]-c[1])**2)
    if dist < minDist:
      index = i
      minDist = dist
  affect.append(index)

fig, ax = plt.subplots()
scatter = ax.scatter(df.iloc[:,0],df.iloc[:,1],c=affect)
ax.legend(*scatter.legend_elements(),loc="upper left", title="Clusters")

plt.xlabel('Nombre de visites')
plt.ylabel('Nombre d\'achats')

affect = np.array(affect)

for i,c in enumerate(clusters):
  subset = df[affect == i]
  clusters[i] = [subset.iloc[:,0].mean(),subset.iloc[:,1].mean()]

plt.scatter([row[0] for row in clusters], [row[1] for row in clusters], marker='*', s=128, c=[0,1,2])
plt.show()

for row in clusters:
    print(np.round(row,2))
    
"""
Exercice : KMeans sur les Iris
Écrire l'algorithme KMeans en Python et l'appliquer aux données Iris.
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import randint

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

def eucDist(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    
centers = [X[randint(0, len(X))],X[randint(0, len(X))],X[randint(0, len(X))]]
for i in range(10):
    affect = np.zeros(len(X))
    for j in range(len(X)):
        minDist = float('inf')
        minK = -1
        for k in range(len(centers)):
           dist = eucDist(centers[k],X[j])
           if(dist < minDist):
               minDist = dist
               minK = k
        affect[j] = minK
    newCenters = []
    for k in range(len(centers)):
        kX = X[affect == k]
        newCenters.append(kX.mean(axis=0))
    centers = newCenters
plt.scatter(X[:,0],X[:,1], c=affect, cmap='rainbow')
plt.show() 

#Scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

cluster = KMeans(n_clusters=3, random_state=0)  
cluster.fit_predict(X)  
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.show()
print(round(metrics.adjusted_rand_score(Y, cluster.labels_),2)) 

"""
Exercice : Variation de k
A l'aide de Scikit-learn et de ses fonctions metrics, calculer l'évolution de l'Adjusted Rand index en fonction du nombre de clusters (de 1 à 20) pour les données Iris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

# boucle sur le nombre de clusters
rand = []
for i in range(1,21):
    cluster = KMeans(n_clusters=i, random_state=0)   
    cluster.fit_predict(X)  
    rand.append(metrics.adjusted_rand_score(Y, cluster.labels_))

# affichage du résultat
plt.plot(np.arange(1, 21, 1.0),rand)
plt.xticks(np.arange(1, 21, 1.0))
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted Rand Index')
plt.axvline(x=3,c='black',ls='-.',lw=0.8)
plt.show()

"""
Conclusion :
Avantages

Très rapide (temps linéaire)
Simple à interpréter : clusters centrés autour d'un prototype
Peu de paramètres : nombre de clusters, nombre d'itérations
Inconvénients

Le nombre de clusters est un paramètre de l'algorithme
Ne peut trouver que des clusters centrés autour d'un prototype
Sensibilité à l’initialisation des clusters
"""