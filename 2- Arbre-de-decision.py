"""
rbre de décision
Représentation arborescente d’une fonction de classification
Une des méthodes les plus connues et appliquées en classification
Toute une famille d'algorithmes (e.g. ID3, ID4, C4.5, C5.0)
Permet de traiter les données numériques et catégorielles
Calcul de l'entropie H(E)
Mesure d’une quantité d’information ou de l’incertitude
Soit pi la proportion d'exemple de la class i dans E :
H(E)=−∑pilog2(pi)
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def entropy(p):
  return -(p * np.log2(p))-((1-p) * np.log2((1-p)))

x = pl.linspace(0.001,0.999,100)
plt.plot(x, entropy(x), 'b')
plt.text(0.25, 0.3, r'$H(E) = -\sum_{i=1}^c p_i \log_2 p_i$', fontsize=15)
plt.grid()

#Exemple de calculs de l'entropie
import pandas as pd
import math

def entropy(ncounts):
  entropy = 0
  for index, value in ncounts.items():
    entropy += -(value * math.log2(value))
  return entropy

tab = pd.Series(['oui','oui','oui','oui','non','non','non','non'])
print(entropy(tab.value_counts(normalize=True)))

tab = pd.Series(['oui','oui','oui','oui','oui','oui','oui','oui'])
print(entropy(tab.value_counts(normalize=True)))

tab = pd.Series(['oui','oui','oui','oui','oui','oui','oui','non'])
print(entropy(tab.value_counts(normalize=True)))

"""
Exercice : Calcul du gain d'entropie
Écrire un programme qui calcule automatiquement le gain d'entropie pour chaque attribut à la racine de l’arbre pour les données Golf
"""
import math
import pandas as pd

# lecture des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\golf.csv", header=0)

attributes = list(df.columns.values)[:-1]

# calcul du gain d'entropie pour chaque attribut
class_count = pd.value_counts(df.iloc[:,-1])
hp = 0
for cl in class_count:
    hp += -(cl/df.shape[0])*math.log2(cl/df.shape[0])
 
for att in attributes:
    vals = df[att].unique()
    ht = 0
    for v in vals:
        subset = df.loc[df[att] == v]
        count = pd.value_counts(subset.iloc[:,-1])
        nrow = subset.shape[0]
        e = 0
        for cl in count:
            e += -(cl/nrow)*math.log2(cl/nrow)
        ht += (nrow / df.shape[0]) * e
    print(att+' '+str(hp-ht))

#Scikit-learn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  precision_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# lecture des données
df = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\golf.csv", header=0)

dummies = [pd.get_dummies(df[c]) for c in df.drop('Play golf', axis=1).columns]
binary_data = pd.concat(dummies, axis=1)

X = binary_data.values

le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Play golf'].values)

# création des ensembles train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# création du classifieur
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# évaluation du classifieur
accuracy = precision_score(y_test, y_pred)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))

# affichage de l'arbre
plt.figure(figsize=(7,7))
tree.plot_tree(clf.fit(X_train, y_train)) 

#Visualisation de la frontière des classes
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
iris = load_iris()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = DecisionTreeClassifier().fit(X, y)

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
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()

"""
Conclusion :
Avantages

Facile à comprendre
Temps d'apprentissage relativement court
Temps de classification très rapide
Gère les données numériques et catégorielles

Inconvénients

Le modèle peut devenir compliqué avec beaucoup d'attributs
Difficulté de mise à jour du modèle
Pas toujours performant
"""