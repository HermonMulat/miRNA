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
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def read_data(data_set="training"):
    """
    Read miRNA data from CSV file.
    """

    if data_set == "training" or data_set == "test":
        return pd.read_csv(data_set+".csv", header=None)
    else:
        print "Must specifiy 'training' or 'testing' set."
