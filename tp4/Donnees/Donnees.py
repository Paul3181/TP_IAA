import matplotlib.image as img
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import load_dataset, plot_training
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

def PretraitementAmeliore(name):
    path = "./Fleurs/"
    image = img.imread(path+name)
    if image.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
        image = (image * 255).astype(np.uint8)

    np_px = 0
    sum_x = 0
    sum_y = 0

    sum_c = 0
    np_px_c = 0

    #parcours de tout les pixels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i < (image.shape[0] / 2) - 20 or i > (image.shape[0] / 2)+20) and (j < (image.shape[1] / 2)-20 or j > 3*(image.shape[1] / 2)+20):
                #normalisation
                px = image[i,j]
                tot = int(px[0])+int(px[1])+int(px[2])
                x = px[0]/max(1,tot)
                y = px[1] / max(1, tot)

                sum_x += x
                sum_y += y
                np_px += 1
            else:
                # normalisation
                px = image[i, j]
                tot = int(px[0]) + int(px[1]) + int(px[2])
                c = px[0] / max(1, tot)

                sum_c += c
                np_px_c += 1


    return(sum_x/np_px, sum_y/np_px, sum_c/np_px_c)


def plot_vraissemblance(X: np.ndarray, Y: np.ndarray,Z) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(mu.shape[0]):
        xd = [i[0] for i in X]
        yd = [i[1] for i in X]
        ax.scatter(xd[:10], yd[:10],Z[:10], c='blue')
        ax.scatter(xd[10:20], yd[10:20],Z[10:20], c='green')
        ax.scatter(xd[20:30], yd[20:30],Z[20:30], c='red')

    ax.set_xlabel('r')
    ax.set_ylabel('v')
    ax.set_zlabel('Vraisemblance')

    plt.show()


color, labels = load_dataset("./couleurs_moyennes_better.csv")

#affichage nuage de point
#plot_training(color,labels)

# Instanciation de la classe GaussianB
g = GaussianBayes(priors=[0.33,0.3,0.326])

# Apprentissage
mu, sig = g.fit(color, labels)

#tab_proba = g.predict(color)

#affichage vraissemblance
#plot_vraissemblance(color,labels,tab_proba)

g.predict(color)

print(g.score(color,labels))


"""
for i in range(n_fleurs):
    file_name = "ch"
    name = file_name + str(i+1) + ".png"
    a, b, c = PretraitementAmeliore(name)
    print(a,",",b,",",c,",",0)

for i in range(n_fleurs):
    file_name = "oe"
    name = file_name + str(i+1) + ".png"
    a, b, c = PretraitementAmeliore(name)
    print(a,",",b,",",c,",",1)

for i in range(n_fleurs):
    file_name = "pe"
    name = file_name + str(i+1) + ".png"
    a, b, c = PretraitementAmeliore(name)
    print(a,",",b,",",c,",",2)
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