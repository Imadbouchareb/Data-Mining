"""
Réseaux de neurones
Exercice : Perceptron
Coder en Python l'apprentissage du OU booléen à l'aide d'un perceptron.
"""

import matplotlib.pyplot as plt
import numpy as np

data = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
c = [0,1,1,1]
weights = [0,1,-1]

# apprentissage des poids
error = []
for i in range(5):
  for j, d in enumerate(data):
    s = 0
    for k in range(len(d)):
      s+= d[k] * weights[k]
    o = 1 if s > 0 else 0
    for k in range(len(weights)):
      weights[k] = weights[k] +  (c[j] - o) * d[k]
    error.append(abs(o-c[j]))
    print('w: '+str(weights)+' s: '+str(s)+' o: '+str(o)+' c: '+str(c[j]))
plt.plot(np.arange(1, 5*len(data)+1, 1.0), error)
plt.xticks(np.arange(1, 5*len(data)+1, 2.0))
plt.ylabel('errors')
plt.xlabel('iterations')
plt.show()

#Scikit-learn
import pandas as pd   
from sklearn import preprocessing    
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import classification_report, confusion_matrix  

df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)  
X = df.iloc[:, 0:4]
y = df.select_dtypes(include=[object])  

le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())  
predictions = mlp.predict(X_test)  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  

#Visualisation de la frontière des classes
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier  

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
iris = load_iris()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000).fit(X, y)

    plt.subplot(2, 3, pairidx + 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
plt.suptitle("Decision surface of a MLP using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()

"""
Conclusion :
Avantages

Bonnes performances
Adapté à l'exécution sur cartes graphiques
Écosystème dynamique (Deep Learning)
Inconvénients

Coté "boite noire"
Choix des hyperparamètres
Ressources nécessaires pour entraîner de gros réseaux
"""