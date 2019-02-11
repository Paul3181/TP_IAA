import numpy as np
import math
from typing import Union


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.empty(n_obs)

        det = np.zeros(n_classes)

        #calcul des determinants
        for k in range(n_classes):
            det[k] = (np.linalg.det(self.sigma[k]))

        for i in range(n_obs):
            maxVal = -100000

            for j in range(n_classes):

                val = -(0.5 * math.log(det[j])) - (0.5)*np.dot(np.dot(np.transpose(X[i] - self.mu[j]), (np.linalg.inv(self.sigma[k]))), X[i] - self.mu[j])

                #rajout de la probabilité à priori
                #val = val * self.priors[j]

                #mise à jour classe
                if(maxVal<(val)):
                    maxVal = (val)
                    y[i] = val


        #print("predict",y)
        return y

    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))

        for i in range (n_classes):
            #calcul vecteur moyen
            self.mu[i] = np.average(X[y == i], 0)
            #calcul de la matrice
            self.sigma[i] = np.cov(X[y == i].T)

        print("vecteur moyen \n",self.mu)

        print("matrice covariance \n",self.sigma)



    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
