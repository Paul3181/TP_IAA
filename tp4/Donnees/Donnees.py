
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, plot_training, plot_vraissemblance
from bayes import GaussianBayes
from mpl_toolkits.mplot3d import Axes3D

n_fleurs = 10

def Pretraitement(name):
    path = "./Fleurs/"
    image = img.imread(path+name)
    if image.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
        image = (image * 255).astype(np.uint8)

    np_px = 0
    sum_x = 0
    sum_y = 0

    #parcours de tout les pixels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #normalisation
            px = image[i,j]
            tot = int(px[0])+int(px[1])+int(px[2])
            x = px[0]/max(1,tot)
            y = px[1] / max(1, tot)

            sum_x += x
            sum_y += y
            np_px += 1

    return(sum_x/np_px, sum_y/np_px)



color, labels = load_dataset("./couleurs_moyennes.csv")

#affichage nuage de point
#plot_training(color,labels)

# Instanciation de la classe GaussianB
g = GaussianBayes()

# Apprentissage
g.fit(color, labels)

#affichage vraissemblance

tab_proba = g.predict(color)
print(tab_proba)


plot_vraissemblance(color, labels, tab_proba)


"""
for i in range(n_fleurs):
    file_name = "ch"
    name = file_name + str(i+1) + ".png"
    a, b = Pretraitement(name)
    print(a,",",b,",",0)

for i in range(n_fleurs):
    file_name = "oe"
    name = file_name + str(i+1) + ".png"
    a, b = Pretraitement(name)
    print(a,",",b,",",1)

for i in range(n_fleurs):
    file_name = "pe"
    name = file_name + str(i+1) + ".png"
    a, b = Pretraitement(name)
    print(a,",",b,",",2)
"""

"""

# Affichage des images de fleurs Chrysanthemes
for i in range(n_fleurs):
    path = "./Fleurs/"
    file_name = "ch"
    name = path + file_name + str(i+1) + ".png"
    print(name)
    image = img.imread(name)

    image2 = np.asarray(image)
    plt.figure(1)
    plt.subplot(3, 4, i+1)
    print(image2.shape)
    plt.imshow(image2)

plt.show()

# Affichage des images de fleurs oeillets
for i in range(n_fleurs):
    path = "./Fleurs/"
    file_name = "oe"
    name = path + file_name + str(i+1) + ".png"
    print(name)
    image = img.imread(name)

    image2 = np.asarray(image)
    plt.figure(2)
    plt.subplot(3, 4, i+1)
    print(image2.shape)
    plt.imshow(image2)

plt.show()

# Affichage des images de fleurs pensees
for i in range(n_fleurs):
    path = "./Fleurs/"
    file_name = "pe"
    name = path + file_name + str(i+1) + ".png"
    print(name)
    image = img.imread(name)

    image2 = np.asarray(image)
    plt.figure(3)
    plt.subplot(3, 4, i+1)
    print(image2.shape)
    plt.imshow(image2)

plt.show()

"""