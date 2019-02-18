# -*- coding: utf-8 -*-
"""
Created on 08/02/19

@author: Thomas Pellegrini
"""

im='./images/matisse.jpg'

from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import csv
from utils import load_dataset
from kmeans import KMeans
from PIL import Image, ImageFilter

def im_to_csv(data):

    nb_pixel = data.shape[0] * data.shape[1]

    with open('matisse.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        #parcours de tout les pixels
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                px = data[i, j]
                tot = int(px[0]) + int(px[1]) + int(px[2])
                x = float(px[0] / max(1, tot))
                y = float(px[1] / max(1, tot))
                writer.writerow([x,y])

def recolorisation(name_image,data, y):

    im = Image.open('./images/'+name_image+'.jpg')
    im_reco = im
    #im_reco.putpixel((1, 1), (0, 0, 250))
    #im_reco.putpixel((3, 1), (0, 0, 250))


    colors = [(0,0,250), (250,0,0), (0,250,0),(153,51,102), (102,102,0), (51,0,0),]

    long = data.shape[0]
    larg = data.shape[1]

    print(long)
    print(larg)

    compt = 0

    #On parcours tout les pixels
    for i in range(0,long):
        for j in range(0,larg):
            # recolorisation
            #print(y[compt])
            im_reco.putpixel((j,i),colors[int(y[compt])])
            #im_reco.putpixel((i,j), (50,150,50))
            compt += 1


    im_reco.save('./images/' + name_image + '_recolorized.jpg', 'JPEG')

    return 0



#recuperation de l'image en npdarray
data = imread(im)
if data.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
    data = (data * 255).astype(np.uint8)

#normalisation
#im_to_csv(data)

#chargement des données csv
rv, labels = load_dataset('matisse.csv')

# initialisation de l'objet KMeans
kmeans = KMeans(n_clusters=6,
                max_iter=100,
                early_stopping=True,
                tol=1e-6,
                display=True)

# calcule les clusters
classes = kmeans.fit(rv)

recolorisation('matisse', data, classes)













#affichage
#plt.imshow(data)
#plt.axis('off')
#plt.show()
