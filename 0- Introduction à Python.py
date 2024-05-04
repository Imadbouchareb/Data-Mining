"""
Types de données en Python :
float : nombre réel
int : nombre entier
str : chaîne de caractère, texte
bool : vrai ou faux
type() : renvoie le type d'une variable
"""
tall = True

height = 1.79
print(type(height))
weight = 68.7

bmi = weight / height ** 2

print(bmi)

"""
Exercice :
Créer une variable factor égale à 1.10
Utiliser savings et factor pour calculer le montant obtenu après 7 ans
Afficher le résultat
"""
# Create a variable savings
savings = 100

# Create a variable factor
factor = 1.10

# Calculate result
results = savings * factor ** 7

# Print out result
print(results)

"""
Exercice :
Exécuter le code ci-dessous et essayer de comprendre l'erreur
Corriger le code à l'aide de la fonction str()
Convertir la variable pi_string en float dans une nouvelle variable pi_float
"""
# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

"""
Exercice :
Les listes en Python :
Exemple de liste :    
"""
fam = [1.73, 1.68, 1.71, 1.89]
#Permet de nommer une collection de valeurs
#Peut contenir plusieurs types de données
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
#* Possibilité de créer une liste de listes :
fam2 = [["liz", 1.73],
        ["emma", 1.68],
        ["mom", 1.71],
        ["dad", 1.89]]

"""
Exercice :
Créer une liste, areas, qui contiendra la taille des différentes pièces
Afficher la liste à l'aide de la fonction print()
"""
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall,kit,liv,bed,bath]

# Print areas
print(areas)

"""
Exercice :
Finir le code pour que chaque pièce possède son propre nom
"""
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = ["hallway", hall,"kitchen", kit, "living room", liv, "bedroom",bed, "bathroom", bath]

# Print areas
print(areas)

"""
Exercice :
Terminer la liste de listes pour qu'elle contienne toutes les pièces puis afficher la
 variable house et son type (avec la fonction type())
"""
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))


#Manipulation des listes :
#Récupération d'un élément d'une liste

fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
print(fam[3])
print(fam[-1])

#Découpage d'une liste (slicing)
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
print(fam[3:5])

"""
Exercice :
Afficher le deuxième élément de la liste areas (11.25)
Extraire et afficher le dernier élément de areas (9.50), utiliser un index négatif
Sélectionner et afficher l'aire de living room
"""
 Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[5])

"""
Exercice :
Créer une variable eat_sleep_area qui contiendra la somme des aires de kitchen et bedroom
Afficher cette nouvelle variable
"""
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[-3]

# Print the variable eat_sleep_area
print(eat_sleep_area)

"""
Exercice :
Utiliser la sélection pour créer une liste downstairs qui contient les 6 premiers éléments de areas
Faire la même chose pour créer upstairs qui contiendra les 4 derniers éléments de areas
"""
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs = areas[0:6]

# Use slicing to create upstairs
upstairs = areas[6:10]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)

"""
Exercice :
Que retournerait house[-1][1] ?
"""
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]
"""
Un réel : la taille de kitchen
Une chaîne: kitchen
Un réel : la taille de bathroom
Une chaîne : bathroom
"""

#Changer une valeur dans une liste
fam = ["liz", 1.73, "emma", 1.68, "mom", 1.71, "dad", 1.89]
print(fam)

fam[7] = 1.86
print(fam)
    
fam[0:2] = ["lisa", 1.74]
print(fam)

#Ajouter et retirer un élément d'une liste 
fam + ["me", 1.79]
print(fam)

del(fam[2])
print(fam)

del(fam[2])
print(fam)

"""
Exercice :
Changer la taille de la piscine en 10.50 à la place de 9.50
Changer la pièce living room en chill zone
"""

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[9] = 10.5

# Change "living room" to "chill zone"
areas[4] = "chill zone"

# Print areas
print(areas)

"""
Exercice :
Utiliser l'opérateur + pour rajouter ["poolhouse", 24.5] à la fin de la liste areas. Stocker le résultat dans areas_1
Ajouter un garage à areas_1 de taille 15.45 et stocker le résultat dans areas_2
"""
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]

# Print areas_1 and areas_2
print(areas_1)
print(areas_2)

"""
Exercice :
Quelle commande exécuter pour retirer la piscine (poolhouse) ?
"""
areas = ["hallway", 11.25, "kitchen", 18.0,
         "chill zone", 20.0, "bedroom", 10.75,
         "bathroom", 10.50, "poolhouse", 24.5,
         "garage", 15.45]

del(areas[-4:-2])


#Les fonctions
fam = [1.73, 1.68, 1.71, 1.89]
print(fam)

tallest = max(fam)
print(tallest)

print(round(1.68, 1))
help(round)

"""
Exercice :
Utiliser print() avec type() pour afficher le type de var1
Utiliser len() pour obtenir la longueur de var1
Utiliser int() pour convertir var2 en entier. Stocker le résultat dans out2
"""
# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True

# Print out type of var1
print(type(var1))

# Print out length of var1
print(len(var1))

# Convert var2 to an integer: out2
out2 = int(var2)

"""
Exercice :
Utiliser + pour fusionner les contenus de first et second et stocker le résultat dans une liste full
Appeler sorted() sur full et spécifier l’argument reverse à True
Enregistrer le résultat dans full_sorted
Terminer par afficher full_sorted
"""
# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse=True)

# Print out full_sorted
print(full_sorted)

#Méthodes sur les listes et les strings
fam = ['liz', 1.73, 'emma', 1.68, 'mom', 1.71, 'dad', 1.89]
print(fam)
print(fam.index("mom"))
print(fam.count(1.73))

sister = 'liz'
print(sister.capitalize())
print(sister.replace("z", "sa"))

"""
Exercice :
Utiliser la méthode upper() sur room et stocker le résultat dans room_up
Afficher room et room_up
Afficher le nombre de o dans la variable room en appelant la méthode count() sur room
"""
# string to experiment with: room
room = "poolhouse"

# Use upper() on room: room_up
room_up = room.upper()

# Print out room and room_up
print(room)
print(room_up)

# Print out the number of o's in room
print(room.count("o"))

"""
Exercice :
Utiliser la méthode index() pour obtenir l'index de l'élément égal à 20.0. Afficher cet index
Appeler la méthode count() sur areas afin de trouver combien de fois 14.5 apparaît dans la liste
Afficher le résultat
"""
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 14.5 appears in areas
print(areas.count(14.5))

"""
Exercice :
Utiliser append() deux fois afin d'ajouter la taille de poolhouse et garage : 24.5 et 15.45
Afficher areas
Utiliser la méthode reverse() afin de d'inverser l'ordre des éléments de areas
Afficher areas
"""
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)

"""
Packages :
Ensemble de scripts Python
Spécifie des fonctions, des méthodes et des types
Des milliers de packages sont disponibles (numpy, matplotlib, scikit-learn)
"""
import numpy
numpy.array([1, 2, 3])

import numpy as np
np.array([1, 2, 3])

from numpy import array
array([1, 2, 3])

"""
Exercice :
Importer le package math. Vous pouvez maintenant utiliser math.pi
Calculer la circonférence du cercle et stocker le dans C
Calculer l'aire du cercle et stocker le dans A
"""
# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2 * math.pi * r

# Calculate A
A = math.pi * r ** 2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))

"""
Exercice :
Effectuer un import sélectif du package math pour importer uniquement la fonction radians
Calculer la distance parcourue par la lune sur 12 degrés de son orbite.
Assigner le résultat à dist.
Vous pouvez la calculer par r * phi, où r est le rayon et phi est l'angle en radians.
Pour convertir l'angle de degré à radians, utiliser la fonction radians() que vous venez d'importer
Afficher dist
"""
# Definition of radius
r = 192500

# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
dist = r * radians(12)

# Print out dist
print(dist)

"""
Numpy :
Numerical Python
Alternative aux listes Python : Numpy Array
Facilite les calculs sur des tableaux
Très efficace pour faire des calculs
"""
import numpy as np

np_height = np.array([ 1.73, 1.68, 1.71, 1.89, 1.79])
print(np_height)

np_weight = np.array([ 65.4, 59.2, 63.6, 88.4, 68.7])
print(np_weight)

bmi = np_weight / np_height ** 2
print(bmi)

#Un Numpy Array ne contient qu'un seul type
array = np.array([1.0, "is", True])
print(array)

python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])

print(python_list + python_list)
print(numpy_array + numpy_array)

#Sélection avec un Numpy Array 
print(bmi)
print(bmi[1])
print(bmi > 23)
print(bmi[bmi > 23])

"""
Exercice :
Importer le package numpy avec l'alias np
Utiliser np.array() pour créer un tableau numpy à partir de baseball. Appeler le np_baseball
Afficher le type de np_baseball pour vérifier son type
"""
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))

"""
Exercice :
Créer un tableau numpy height appelé np_height
Afficher np_height
Multiplier np_height par 0.0254 afin de convertir le tableau des pouces aux mètres. Stocker les valeurs dans np_height_m
Afficher np_height_m et vérifier le résultat
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# height is available as a regular list
from baseballdata import *

# Import numpy
import numpy as np

# Create a numpy array from height: np_height
np_height = np.array(height)

# Print out np_height
print(np_height)

# Convert np_height to m: np_height_m
np_height_m = np_height * 0.0254

# Print np_height_m
print(np_height_m)

"""
Exercice :
Créer un tableau numpy pour la liste de poids avec la bonne unité
Multiplier par 0.453592 pour passer des livres aux kilos. Stocker le résultat dans np_weight_kg
Utiliser np_height_m et np_weight_kg pour calculer le BMI (IMC) de chaque joueur. Utiliser l'équation suivante :  BMI=weight(kg)height(m)2 
Stocker le résultat dans la variable bmi et l'afficher
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# height and weight are available as a regular lists
from baseballdata import *

# Import numpy
import numpy as np

# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254

# Create array from weight with correct units: np_weight_kg
np_weight_kg = np.array(weight) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m ** 2

# Print out bmi
print(bmi)

"""
Exercice :
Créer un tableau numpy de boolean: les éléments du tableau doivent être True si le BMI du joueur correspondant est inférieur à 21 (vous pouvez utiliser l'opérateur  < ). Nommer le tableau light et l'afficher.
Afficher le tableau light
Afficher un tableau numpy avec les BMI de tous les joueurs qui ont un BMI inférieur à 21. Utiliser le tableau light pour faire la sélection sur bmi.
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# height and weight are available as a regular lists
from baseballdata import *

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])

"""
Exercice :
Afficher l’élément à l'index 50 de np_weight
Afficher le sous tableau de np_weight qui contient les éléments de 100 à 110 inclus
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# height and weight are available as a regular lists
from baseballdata import *

# Import numpy
import numpy as np

# Store weight and height lists as numpy arrays
np_weight = np.array(weight)
np_height = np.array(height)

# Print out the weight at index 50
print(np_weight[50])

# Print out sub-array of np_height: index 100 up to and including index 110
print(np_height[100:111])


#2D Numpy Arrays
#Tableau à deux dimensions
import numpy as np

np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
                  [65.4, 59.2, 63.6, 88.4, 68.7]])

print(np_2d)
print(np_2d.shape)

#Sélection dans un tableau 2D
print(np_2d[0])
print(np_2d[0][2])
print(np_2d[0,2])
print(np_2d[:,1:3])
print(np_2d[1,:])

"""
Exercice :
Utiliser np.array() pour créer un tableau numpy 2D pour baseball, appeler le np_baseball et afficher le
Afficher la forme du tableau, utiliser np_baseball.shape
"""
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)

"""
Exercice :
Utiliser np.array() pour créer un tableau numpy 2D à partir de baseball et l'appeler np_baseball
Afficher la forme de np_baseball
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# baseball is available as a regular list of lists
from baseballdata import *

# Import numpy package
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)

"""
Exercice :
Afficher la 50ème ligne de np_baseball
Créer une nouvelle variable, np_weight, contenant la deuxième colonne de np_baseball
Sélectionner la taille (première colonne) du joueur 124 dans np_baseball et l'afficher
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata.py

# baseball is available as a regular list of lists
from baseballdata import *

# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123, 0])

"""
Exercice :
Vous avez récupéré les changements de poids taille et age des joueurs de baseball. Ils sont stockés dans le tableau numpy 2D array. Additionner np_baseball et updated et afficher le résultat.
Vous souhaitez convertir les unités de taille et poids. Créer une tableau numpy avec les valeurs 0.0254, 0.453592 et 1. Nommer ce tableau conversion.
Multiplier np_baseball et conversion et afficher le résultat.
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata2.py

# updated is available as 2D numpy array
from baseballdata2 import *

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)

"""
Numpy statistiques :
Comprendre vos données
Peu de données : simplement les observer
Beaucoup de données : ?
"""
import numpy as np
np_city = np.array([[ 1.64, 71.78],
                    [ 1.37, 63.35],
                    [ 1.6 , 55.09],
                    [ 2.04, 74.85],
                    [ 2.04, 68.72],
                    [ 2.01, 73.57]])
print(np.mean(np_city[:,0]))
print(np.median(np_city[:,0]))

print(np.corrcoef(np_city[:,0], np_city[:,1]))
print(np.std(np_city[:,0]))

"""
Exercice :
Créer un tableau numpy np_height qui contient la première colonne de np_baseball
Afficher la moyenne de np_height
Afficher le médiane de np_height
Constatez vous une erreur ? Quelle statistique est la plus adaptée pour détecter cette erreur ?
"""
# donwload the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata3.py

# np_baseball is available
from baseballdata3 import *

# Import numpy
import numpy as np

# Create np_height from np_baseball
np_height = np_baseball[:,0]

# Print out the mean of np_height
print(np.mean(np_height))

# Print out the median of np_height
print(np.median(np_height))

"""
Exercice :
Compléter le code pour calculer la médiane.
Utiliser np.std() sur la première colonne de np_baseball pour calculer l'écart type.
Est-ce-que les grands joueurs sont plus lourd ? Utiliser np.corrcoef() pour stocker la corrélation entre la première et le deuxième colonne de np_baseball dans corr.
"""
# download the baseball data
!wget -nc https://germain-forestier.info/dataset/baseballdata3.py

# np_baseball is available
from baseballdata3 import *

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))

"""
Exercice :
Convertir heights et positions en tableaux numpy np_heights et np_positions
Extraire les tailles des goalkeepers dans gk_heights (utiliser np_positions == 'GK')
Extraire les tailles des autres joueurs dans other_heights (utiliser np_positions != 'GK')
Afficher la taille médiane des goalkeepers avec np.median()
Afficher la taille médiane des autres joueurs avec np.median()
"""
# donwload the football data
!wget -nc https://germain-forestier.info/dataset/foot.py

# heights and positions are available as lists
from foot import *

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_heights = np.array(heights)
np_positions = np.array(positions)

# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK' ]

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK' ]

# Print out the median height of goalkeepers. Replace 'None'
mhg = np.median(gk_heights)
print("Median height of goalkeepers: " + str(mhg))

# Print out the median height of other players. Replace 'None'
mho = np.median(other_heights)
print("Median height of other players: " + str(mho))