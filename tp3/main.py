import numpy as np
import matplotlib
import six
from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist


def main():
    train_data, train_labels = load_dataset("./data/train.csv")
    test_data, test_labels = load_dataset("./data/test.csv")

    #affichage
    plot_scatter_hist(train_data, train_labels)


    # Instanciation de la classe GaussianB
    g = GaussianBayes(priors=[0.3,0.3,0.3], diag=True)

    # Apprentissage
    g.fit(train_data, train_labels)


    # Score
    score = g.score(test_data, test_labels)
    print("precision : {:.2f}".format(score))

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
