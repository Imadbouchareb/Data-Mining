"""
Classifieur bayésien
On associe a chaque hypothèse une probabilité d’être la solution
L'observation des instances peut modifier cette probabilité
On peut parler de l’hypothèse la plus probable
Basée sur les probabilités conditionnelles (et la règle de Bayes)
Suppose l'indépendance des attributs
"""

#Scikit-learn

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# lecture des données
df = pd.read_csv('https://germain-forestier.info/dataset/iris.csv', header=0)

# création des ensembles train / test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1], test_size=0.2)

#création du classifieur
gnb = GaussianNB()
gnb.fit(X_train, y_train)

predictions = gnb.predict(X_test)

# évaluation du classifieur
print(confusion_matrix(predictions, y_test))
print(classification_report(predictions, y_test))

#Visualisation de la frontière des classes
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
iris = load_iris()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = GaussianNB().fit(X, y)

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
plt.suptitle("Decision surface of a GaussianNB using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


"""
Exercice : Classification de tweets
Entraîner un classifieur bayésien avec les données data ci-dessous
Utiliser CountVectorizer de sklearn pour compter les occurrences des mots
Utiliser un classifieur MultinomialNB
"""

data = [
['I love this sandwich.', 'pos'],
['This is an amazing place!', 'pos'],
['I feel very good about these beers.', 'pos'],
['This is my best work.', 'pos'],
['What an awesome view', 'pos'],
['I do not like this restaurant', 'neg'],
['I am tired of this stuff.', 'neg'],
['I can\'t deal with this', 'neg'],
['He is my sworn enemy!', 'neg'],
['My boss is horrible.', 'neg'],
['The beer was good.', 'pos'],
['I do not enjoy my job', 'neg'],
['I ain\'t feeling dandy today.', 'neg'],
['I feel amazing!', 'pos'],
['Gary is a friend of mine.', 'pos'],
['I can\'t believe I\'m doing this.', 'neg']
]

#Corrigé 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

df = pd.DataFrame(data)

df[1] = df[1].map({'pos': 0, 'neg': 1}) 
df[0] = df[0].map(lambda x: x.lower())  
df[0] = df[0].str.replace('[^\w\s]', '')  

vectorizer = CountVectorizer()  
counts = vectorizer.fit_transform(df[0]) 
#print(vectorizer.get_feature_names())
#print(counts.toarray())

X_train, X_test, y_train, y_test = train_test_split(counts, df[1], test_size=0.5, random_state=69)  
model = MultinomialNB().fit(X_train, y_train)  

predicted = model.predict(X_test)
print(np.mean(predicted == y_test)) 

"""
Conclusion :
Avantages

Mise à jour du modèle possible
Temps d'apprentissage relativement court
Temps de classification très rapide
Inconvénients

Hypothèse d'indépendance des attributs
Problème de ZeroFrequency
"""
