"""
K plus proches voisins
La méthode du 1-NN (1 nearest neighbor) consiste à trouver l'instance  A  la plus proche de l'instance  B  à classer dans les données disponibles. La classe de  B  est ensuite affectée à  A 
La méthode du K-NN (K nearest neighbors) consiste à rechercher les  K  plus proches voisins et affecter la classe majoritaire à l'instance à classer
C'est une méthode d'apprentissage paresseuse (lazy learning) car il n'y a pas d'étape de construction d'un modèle
Il est nécessaire de disposer d'une distance entre les instances (pas trivial en fonction du type de données)

Exercice : Classifieur du 1 plus proche voisin
Compléter le code ci-dessous afin d'écrire un classifieur 1 plus proche voisin sur les données Iris
Commencer par écrire une fonction qui calcule la distance euclidienne entre deux lignes du DataFrame Iris
Créer ensuite les ensembles train/test et enfin le classifieur
Terminer par l'évaluation du classifieur réalisé
"""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# lecture des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

# fonction de calcul de la distance euclidienne
def euclidean_distance(a, b):
    sum = 0
    for i in range(a.size-1):
        sum += (a[i]-b[i])**2
    return math.sqrt(sum)

# création des ensembles train / test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1], test_size=0.2)

prediction = np.zeros(y_test.shape[0],dtype='object')

# recherche du plus proche voisin
for i in range(X_test.shape[0]):
    distMin = np.inf
    indexMin = -1;
    current = X_test.iloc[i,:]
    for j in range(X_train.shape[0]):
        t = X_train.iloc[j,:]
        dist = euclidean_distance(current,t)
        if dist < distMin:
            distMin = dist
            indexMin = j
    prediction[i] = y_train.iloc[indexMin]

# évaluation du classifieur
cnf_matrix = confusion_matrix(prediction, y_test)
print(cnf_matrix)
print(classification_report(prediction, y_test))

#----------------------------------------------------------------------------
# Scikit-learn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# lecture des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

# création des ensembles train / test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1], test_size=0.2)

# création du classifieur
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

# évaluation du classifieur
print(confusion_matrix(predictions, y_test))
print(classification_report(predictions, y_test))

#----------------------------------------------------------------------------
# Visualisation de la frontière des classes
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
iris = load_iris()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)

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
plt.suptitle("Decision surface of a 3-NN using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()

#----------------------------------------------------------------------------
"""
Exercice : Recherche du meilleur k
Un des problèmes du K plus proches voisins est de choisir la valeur de K
On choisi souvent un nombre impair pour éviter les égalités lors du vote (dans le cas d'une classification binaire)
Écrire un programme Python qui calcule la précision de classification pour K variant de 1 à 10 pour le jeu de données Iris
Visualiser le résultat sous la forme d'une courbe (K vs. Précision)
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# lecture des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\iris.csv", header=0)

# création des ensembles train / test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1], test_size=0.2, random_state=42)

# boucle sur le nombre de voisins
perf = []

for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    perf.append(precision_score(predictions, y_test, average='micro'))
plt.plot(perf)
plt.show()

"""
Exercice : Classification d'images avec le KNN
Dans cet exercice, nous allons explorer l'algorithme k-NN à l'aide d'un ensemble
 de données d'image. Nous utiliserons l'ensemble de données de chiffres manuscrits
 de sklearn, qui se compose d'images de 8 x 8 pixels de chiffres manuscrits (0-9). Allons
"""
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Display the first few images
fig, axes = plt.subplots(1, 5, figsize=(10, 4))
for ax, image in zip(axes, digits.images):
    ax.imshow(image, cmap=plt.cm.gray_r)
plt.show()

"""
Compléter le code ci-dessous pour lancer le classifieur KNN sur ces données.
Ecrire un programme pour afficher les 5 plus proches voisins d'une image donnée
Evaluer le classifieur
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Initialize and train the k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train, y_train)

import numpy as np

# Select a sample from the test set
sample_index = 25
sample_image = X_test[sample_index].reshape(1, -1)

# Find the indices of the 5 nearest neighbors from the training set
neighbors_indices = knn_classifier.kneighbors(sample_image, n_neighbors=5, return_distance=False)

# Display the sample image
plt.imshow(sample_image.reshape(8, 8), cmap=plt.cm.gray_r)
plt.title("Sample Image")
plt.show()

# Display the 5 nearest neighbors
fig, axes = plt.subplots(1, 5, figsize=(10, 4))
for ax, neighbor_idx in zip(axes, neighbors_indices[0]):
    ax.imshow(X_train[neighbor_idx].reshape(8, 8), cmap=plt.cm.gray_r)
plt.suptitle("5 Nearest Neighbors")
plt.show()

# Predict the labels for the test set
knn_y_pred = knn_classifier.predict(X_test)

# Evaluate the accuracy
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("Accuracy of the k-NN model with k=5:", knn_accuracy)

cnf_matrix = confusion_matrix(knn_y_pred, y_test)
print(cnf_matrix)
print(classification_report(knn_y_pred, y_test))






