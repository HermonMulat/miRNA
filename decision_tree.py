"""
decision_tree.py

Build decision tree model on miRNA data. Uses randomized grid search to
determine best model.

Collin Epstein, Hermon Mulat
2/26/18
CSC371
Dr. Ramanujan
"""

# imports
#import sys, os
import numpy as np
import pandas as pd
import datetime
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

def read_data(data_set="training"):
    """
    Read miRNA data from CSV file. Return features and targets separately.
    """

    if data_set == "training" or data_set == "test":
        data = pd.read_csv(data_set+".csv", header=None)
        last_col = data.shape[-1] - 1
        targ = data[last_col]
        feat = data.drop([last_col], axis = 1)
        return feat, targ
    else:
        print "Must specifiy 'training' or 'testing' set."
        quit()

# Utility function to report best scores
def report(results, n_top=3):
    """
    Report best scores from generated models.
    Modified from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    """

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def main():

    print "Trial begin:", str(datetime.datetime.now())
    # read data
    features, targets = read_data("training")
    print "Data read from file"

    #toy = True
    toy = False
    if(toy):
        frac = 0.2
        print "Running on {0}% of data.".format(frac)
        features = features[:int(len(features)*frac)]
        targets = targets[:int(len(targets)*frac)]

    # standardize data
    scaler = StandardScaler()
    scaler.fit(features)
    std_features = scaler.transform(features)
    print "Data standardized"
    print("")

    # parameter distribution for randomized grid search
    param_dist = {"criterion" : ["gini", "entropy"],
                  "splitter" : ["best", "random"],
                  "max_depth" : range(2,21),
                  "min_samples_split" : range(2,21),
                  "min_samples_leaf" : range(1,21),
                  #"min_weight_fraction_leaf" : [],
                  #"max_features" : [],
                  #"random_state" : [],
                  #"max_leaf_nodes" : [],
                  "min_impurity_decrease" : [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
                  #"class_weight" : [],
                  #"presort" : [True, False]
                  }

    # set up trials
    dtc = DecisionTreeClassifier()
    iterations = 100
    rscv = RandomizedSearchCV(dtc, param_distributions=param_dist,
                                n_iter=iterations, scoring="f1_micro", cv=10)

    # run randomized hyperparamter search with cross validation on decision DecisionTreeClassifier
    start = time()
    rscv.fit(std_features, targets)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), iterations))
    report(rscv.cv_results_)

if __name__ == "__main__":
    main()
