"""
Régression
Les méthodes de régression permettent de faire des prédictions de valeurs continues contrairement à la classification qui cible des valeurs discrètes
Le modèle le plus simple de régression est la régression linéaire avec la méthode des moindres carrés
"""

#Scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\lsd.csv")
data.head()

X = data['Tissue Concentration'].values[:,np.newaxis]
y = data['Test Score'].values

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y,color='r')
plt.plot(X, model.predict(X),color='k')
plt.xlabel('Tissue Concentration')
plt.ylabel('Test Score')
plt.show()

"""
Exercice : (manager de food truck)
Votre êtes manager d'une entreprise de food truck. Vous avez acquis des données afin d'observer les profits de votre entreprise en fonction de la population des villes dans lesquelles votre entreprise est implantée.

Effectuer une régression linéaire afin d'observer la corrélation entre la population et les profits

Afficher la régression avec matplotlib puis avec seaborn lmplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# lecture des données
data = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\lsd.csv")

data.head()

X = data['Population'].values[:,np.newaxis]
y = data['Profit'].values

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y,color='r')
plt.plot(X, model.predict(X),color='k')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()


import pandas as pd
import seaborn as sns

d = pd.read_csv('https://germain-forestier.info/dataset/ex1data1.csv')
g = sns.lmplot(x='Population', y='Profit', data=d)
plt.show()


"""
Exercice : (prévision météo)
Vous travaillez pour météo France et vous voulez prédire la température maximum à partir de la température minimum prévue

Effectuer une régression linéaire afin d'observer la corrélation entre température minimum et maximum

Afficher la régression avec matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# lecture des données
data = pd.read_csv(r"C:\Users\IMAD\Documents\Fouille de données\weather-values.csv", low_memory=False)

data.head()

data.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

X = data['MinTemp'].values[:,np.newaxis]
y = data['MaxTemp'].values[:,np.newaxis]

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y,color='r')
plt.plot(X, model.predict(X),color='k')
plt.show()