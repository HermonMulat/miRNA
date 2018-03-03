"""
random_forest.py

Build random forest model on miRNA data. Uses randomized grid search to
determine best model. Includes some feature selection.

Collin Epstein, Hermon Mulat
2/28/18
CSC371
Dr. Ramanujan
"""

# imports
#import sys, os
import numpy as np
import pandas as pd
import datetime
from time import time
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from decision_tree import read_data, report, variance_feat_sel, chi2_feat_sel

def main():

        # mark trial transript beginning
        print "===================================================================="
        print "Trial begin:", str(datetime.datetime.now())
        print "===================================================================="
        print ""

        # read data
        features, targets = read_data("training")
        print "Data read from file"

        # trial settings
        toy = False
        variance_flag = False
        threshold = 0.0
        chi2_flag = False
        percentile = 10
        binarize_flag = False

        # quick trial on subset of training data
        if toy:
            frac = 0.2
            num = int(len(features)*frac)
            print "Running on {0}% of data, {1} features.".format(frac, num)
            features = features[:num]
            targets = targets[:num]

        # feature selection based on variance
        if variance_flag:
            features = variance_feat_sel(features, thresh=threshold)
            print "Features selected by variance with threshold = {0}".format(threshold)

        # feature selection based on chi2 score
        if chi2_flag:
            features = chi2_feat_sel(features, targets, percent=percentile)
            print "Features selected by chi2 with percentile = {0}%".format(percentile)

        # binarize all feature data
        if binarize_flag:
            features = binarize(features, threshold)
            print "Features binarized with threshold = {0}".format(threshold)

        # standardize data
        if not binarize_flag:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            print "Data standardized"
        print ""

        # parameter distribution for randomized hyperparameter search
        param_dist = {"n_estimators" : range(10,101)[::10],
                      "criterion" : ["gini", "entropy"],
                      "max_features" : ["sqrt", "log2", 0.75, 0.5, 0.25, None],
                      "max_depth" : range(2,21),
                      "min_samples_split" : range(2,21),
                      "min_samples_leaf" : range(1,21),
                      #"min_weight_fraction_leaf" : [], ?
                      #"random_state" : [], ?
                      "max_leaf_nodes" : range(10, 100)[::10] + [None],
                      "min_impurity_decrease" : [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
                      #"bootstrap" : [True, False], ?
                      #"oob_score" : [True, False], ?
                      "n_jobs" : [-1], # parallelize as much as possible
                      #"random_state" : [], ?
                      #"verbose" : [], ?
                      "warm_start" : [True, False],
                      #"class_weight" : [] ?
                      }

        # set up trials
        rfc = RandomForestClassifier()
        iterations = 100
        scoring = "f1_weighted"
        folds = 10

        rscv = RandomizedSearchCV(rfc, param_distributions=param_dist,
                                    n_iter=iterations, scoring=scoring, n_jobs=-1,
                                    cv=folds)

        # print trial parameters
        print "Using parameters:\nIterations = {0}\tScoring = {1}\t Folds = {2}\n".format(iterations, scoring, folds)

        # run randomized hyperparameter search with cross validation on decision tree
        start = time()
        rscv.fit(features, targets)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), iterations))
        report(rscv.cv_results_)
        print "Trial finished\n"

if __name__ == '__main__':
    main()
