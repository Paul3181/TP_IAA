import csv
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def load_dataset(pathname:str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset in csv format.

    Each line of the csv file represents a data from our dataset and each
    column represents the parameters.
    The last column corresponds to the label associated with our data.

    Parameters
    ----------
    pathname : str
        The path of the csv file.

    Returns
    -------
    data : ndarray
        All data in the database.
    labels : ndarray
        Labels associated with the data.
    """
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile, delimiter=',')
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        # converts Python lists into NumPy matrices
        # in the case of the list of labels, generate an int id per class
        data = np.array(data, dtype=np.float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    # return data with the associated label
    return data, labels


def plot_training(data: np.ndarray, labels: np.ndarray) -> None:

    customPalette = ['#0D31FB', '#1BFB0D','#FB370D']
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title("Répartition en fonction de la couleur moyenne")
    for y in np.unique(labels):
        x = data[labels == y]
        # add data points
        ax.scatter(x=x[:, 0],
                   y=x[:, 1],
                   alpha=0.50,
                   color=customPalette[int(y)])


    # met à jour l'affichage
    plt.pause(0.25)
    plt.show()

def fun(x, y, z):
  return int(z[x])

def plot_vraissemblance(data: np.ndarray, labels: np.ndarray, z: np.ndarray) -> None:


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y, z) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()