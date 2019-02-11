# author: Benjamin Chamand, Thomas Pelligrini


import shutil
import numpy as np
import random

from utils import plot_training


class KMeans(object):
    def __init__(self, n_clusters:int, max_iter:int, early_stopping:bool=False,
                 tol:float=1e-4, display:bool=True) -> None:
        self.n_clusters = n_clusters            # Nombre de clusters
        self.max_iter = max_iter                # Nombre d'itération
        self.early_stopping = early_stopping    # arrête l'apprentissage si 
        self.tol = tol                          # seuil de tolérance entre 2 itérations
        self.display = display                  # affichage des données

        self.cluster_centers = None             # Coordonnées des centres de regroupement
                                                # (centre de gravité des classes)
    
    def _compute_distance(self, vec1:np.ndarray, vec2:np.ndarray) -> np.ndarray:
        """Retourne la distance quadratique entre vec1 et vec2 (squared euclidian distance)
        """

        dist = (np.linalg.norm(vec1 - vec2))*(np.linalg.norm(vec1 - vec2))

        return dist
    
    def _compute_inertia(self, X:np.ndarray, y:np.ndarray) -> float:
        """Retourne la Sum of Squared Errors entre les points et le centre de leur
        cluster associe
        """
        sse = 0

        for i in range (0,len(X)-1):

            d = self._compute_distance(X[i],self.cluster_centers[int(y[i])])
            sse = sse + d

        return sse

    def _update_centers(self, X:np.ndarray, y:np.ndarray) -> None:
        """Recalcule les coordonnées des centres des clusters
        """

        sum = []

        dim = len(X[0]-1)

        # parcours des clusters
        for i in range(self.n_clusters):

            compt = 0

            sum.clear()

            for d in range(dim):
                sum.append(0)


            numPts = 0

            # parcours de tout les points
            for j in y:

                # si le point appartient au cluster
                if (i == j):
                    compt += 1
                    #somme des coordonnees
                    for k in range(dim):
                        sum[k] = sum[k]+X[numPts][k]

                numPts += 1

            #maj coordonnes cluster
            for l in range(dim-1):

                if (compt==0):
                    self.cluster_centers.itemset((i, l), sum[l])
                else:
                    self.cluster_centers.itemset((i,l),sum[l] * (1/compt) )



    def predict(self, X:np.ndarray) -> np.ndarray:
        """attribue un indice de cluster à chaque point de data

        X = données
        y = cluster associé à chaque donnée
        """
        # nombre d'échantillons
        n_data = X.shape[0]

        y = np.ndarray(shape=(n_data,))

        infini = float('inf')

        for i in range (n_data):

            min = infini

            for j in range (self.n_clusters):

                dist = self._compute_distance(X[i],self.cluster_centers[j])
                if (dist <= min):
                    min = dist
                    idCluster = j

            y.itemset((i),idCluster)

        return y

    def fit(self, X:np.ndarray):
        """Apprentissage des centroides
        """
        # Récupère le nombre de données
        n_data = X.shape[0]

        # Sauvegarde tous les calculs de la somme des distances euclidiennes pour l'affichage
        if self.display:
            shutil.rmtree('./img_training', ignore_errors=True)
            metric = []

        # 2 cas à traiter : 
        #   - soit le nombre de clusters est supérieur ou égale au nombre de données
        #   - soit le nombre de clusters est inférieur au nombre de données
        if self.n_clusters >= n_data:
            # Initialisation des centroides : chacune des données est le centre d'un clusteur
            self.cluster_centers = np.zeros(self.n_clusters, X.shape[1])
            self.cluster_centers[:n_data] = X
        else:
            # Initialisation des centroides

            self.cluster_centers = np.ndarray(shape=(self.n_clusters,X.shape[1]))

            # centroides pris aleatoirement

            for i in range(self.n_clusters):
                r = random.randint(0,n_data-1)

                self.cluster_centers[i] = X[r]

            # initialisation d'un paramètre permettant de stopper les itérations lors de la convergence
            stabilise = False

            # Exécution de l'algorithme sur plusieurs itérations
            for i in range(self.max_iter):

                # détermine le numéro du cluster pour chacune de nos données
                y = self.predict(X)

                # calcule de la somme des distances initialiser le paramètres
                # de la somme des distances
                if i == 0:
                    current_distance = self._compute_inertia(X, y)

                # mise à jour des centroides
                self._update_centers(X, y)

                # mise à jour de la somme des distances
                old_distance = current_distance
                current_distance = self._compute_inertia(X, y)

                # stoppe l'algorithme si la somme des distances quadratiques entre 
                # 2 itérations est inférieur au seuil de tolérance
                if self.early_stopping:

                    if abs((old_distance-current_distance) <= self.tol):
                        stabilise = True

                    if stabilise:
                        break

                # affichage des clusters
                if self.display:
                    print("on affiche")
                    diff = abs(old_distance - current_distance)
                    metric.append(diff)
                    plot_training(i, X, y, self.cluster_centers, metric)

        return y

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Calcule le score de pureté
        """
        n_data = X.shape[0]

        y_pred = self.predict(X)

        score = 0
        for i in range(self.n_clusters):
            _, counts = np.unique(y[y_pred == i], return_counts=True) 
            score += counts.max()

        score /= n_data

        return score
