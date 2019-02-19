# -*- coding: utf-8 -*-
"""
Created on 08/02/19

@author: Thomas Pellegrini
"""

im='./images/miro.jpg'

from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import csv
from utils import load_dataset
from kmeans import KMeans
from PIL import Image, ImageFilter

def im_to_csv(data):

    with open('miro.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        #parcours de tout les pixels
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                px = data[i, j]
                tot = int(px[0]) + int(px[1]) + int(px[2])
                x = float(px[0] / max(1, tot))
                y = float(px[1] / max(1, tot))
                writer.writerow([x,y,0])

def recolorisation(name_image,data, y,clusters):

    im = Image.open('./images/'+name_image+'.jpg')
    im_reco = im


    #colors = [(0,0,250), (250,0,0), (0,250,0),(153,51,102), (102,102,0), (51,0,0)]
    #choix des couleurs pour la recoloration
    colors = []
    for i in clusters:
        print(i)
        #denormalisation
        b = int((1-(i[0]+i[1]))*255)
        if(b<=0):
            b=0

        c = (int(i[0]*255),int(i[1]*255),b)
        print(c)

        colors.append(c)

    #print(colors)
    long = data.shape[0]
    larg = data.shape[1]

    compt = 0

    #On parcours tout les pixels
    for i in range(0,long):
        for j in range(0,larg):
            # recolorisation
            im_reco.putpixel((j,i),colors[int(y[compt])])
            compt += 1


    im_reco.save('./images/' + name_image + '_recolorized2.jpg', 'JPEG')

    return 0



#recuperation de l'image en npdarray
data = imread(im)
if data.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
    data = (data * 255).astype(np.uint8)

#normalisation
#im_to_csv(data)

#chargement des données csv
rv, labels = load_dataset('miro.csv')

# initialisation de l'objet KMeans
kmeans = KMeans(n_clusters=6,
                max_iter=100,
                early_stopping=True,
                tol=1e-6,
                display=False)

# calcule les clusters
classes, clusters = kmeans.fit(rv)


#print(clusters)
recolorisation('miro', data, classes, clusters)













#affichage
#plt.imshow(data)
#plt.axis('off')
#plt.show()
