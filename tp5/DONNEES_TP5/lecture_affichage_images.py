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


#recuperation de l'image
data = imread(im)
if data.dtype == np.float32:  # Si le résultat n'est pas un tableau d'entiers
    data = (data * 255).astype(np.uint8)

#normalisation
#im_to_csv(data)

#chargement des données csv
rv, labels = load_dataset('matisse.csv')

# initialisation de l'objet KMeans
kmeans = KMeans(n_clusters=3,
                max_iter=100,
                early_stopping=True,
                tol=1e-6,
                display=True)

# calcule les clusters
classes = kmeans.fit(rv)
colors = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#04AF00', '#39A7FF',
                     '#7519CC', '#79E7FF', '#1863C15', '#B72EB9', '#EC2328', '#C86D39']

compt = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        #recolorisation

        compt += 1











#affichage
#plt.imshow(data)
#plt.axis('off')
#plt.show()
