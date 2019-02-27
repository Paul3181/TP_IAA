# -*- coding: utf-8 -*-
"""
Created on 08/02/19

@author: Thomas Pellegrini
"""

im='./images/joconde.jpg'

from scipy.misc import imread
import matplotlib.image as img
import numpy as np
import csv
from utils import load_dataset
from kmeans import KMeans
from PIL import Image, ImageFilter

def im_to_csv(data):

    with open('joconde.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        #parcours de tout les pixels
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                px = data[i, j]
                writer.writerow([float(px[0]),float(px[1]),float(px[2]),0])

def recolorisation(name_image,data, y,clusters):

    im = Image.open('./images/'+name_image+'.jpg')
    im_reco = im

    #choix des couleurs pour la recoloration
    colors = []
    for i in clusters:
        c = (int(i[0]), int(i[1]), int(i[2]))
        colors.append(c)

    long = data.shape[0]
    larg = data.shape[1]

    compt = 0
    #On parcours tout les pixels
    for i in range(0,long):
        for j in range(0,larg):
            # recolorisation
            im_reco.putpixel((j,i),colors[int(y[compt])])
            compt += 1


    im_reco.save('./images/' + name_image + '_recolorized_6clusters.jpg', 'JPEG')

    return 0

def calculDiff(image1, image2):
    im1 = img.imread('./images/' + image1 + '.jpg')
    im2 = img.imread('./images/' + image2 + '.jpg')

    diff = 0

    # On parcours tout les pixels
    for i in range(0, im1.shape[0]):
        for j in range(0, im1.shape[1]):
            px1 = im1[i, j]
            px2 = im2[i, j]

            diff += (abs(int(px1[0])-int(px2[0]))) + (abs(int(px1[1])-int(px2[1]))) + (abs(int(px1[2])-int(px2[2])))

    diff = (diff/(im1.shape[0]*im1.shape[1]))/765

    return diff

#recuperation de l'image en npdarray
data = imread(im)
if data.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
    data = (data * 255).astype(np.uint8)

#normalisation
#im_to_csv(data)

#chargement des données csv
rv, labels = load_dataset('joconde.csv')

# initialisation de l'objet KMeans
kmeans = KMeans(n_clusters=6,
                max_iter=100,
                early_stopping=True,
                tol=1e-6,
                display=False)

# calcule les clusters
#classes, clusters = kmeans.fit(rv)


#print(clusters)
#recolorisation('joconde', data, classes, clusters)



#calcul de la diff entre 2 images de meme taille!
print("diff =", calculDiff("joconde", "joconde_recolorized_6clusters"))




