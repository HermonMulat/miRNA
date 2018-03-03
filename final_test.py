"""
final_test.py

Trains final model on training data, evaluates on testing data. Writes results,
displays confusion matrix plot for examination.

Collin Epstein, Hermon Mulat
3/2/18
CSC371
Dr. Ramanujan
"""

# imports
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from decision_tree import read_data
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def main():

    # mark trial beginning
    print "===================================================================="
    print "Trial begin:",str(datetime.datetime.now())
    print "===================================================================="
    print ""

    # import data
    print "Importing data"
    train_feat, train_targ = read_data("training")
    test_feat, test_targ = read_data("test")
    print "Data imported\n"

    # standardize datasets
    print "Standardizing data"
    scaler = StandardScaler()
    scaler.fit(train_feat) # fit on training data
    train_feat = scaler.transform(train_feat)
    test_feat = scaler.transform(test_feat) # transform based on training data
    print "Data standardized\n"

    # select features
    print "Selecting features"
    selector = SelectKBest(chi2, 1577) # optimal feature selection  = 1577 best
    selector.fit(train_feat, train_targ) # fit on training data
    train_feat = selector.transform(train_feat)
    test_feat = selector.transform(test_feat) # transform based on training data
    print "Features selected\n"

    # train model
    print "Training model"
    lrc = LogisticRegression(penalty="l1", n_jobs=-1) # best performing models
    lrc.fit(train_feat, train_targ)
    print "Model trained\n"

    # evaluate model
    print "Testing model"
    predictions = lrc.predict(test_feat)
    f1 = f1_score(test_targ, predictions, average = "weighted")
    accuracy = accuracy_score(test_targ, predictpredictionsed)
    cm = confusion_matrix(test_targ, predictions, labels)
    print "Model tested\n"

    # report Results
    print "Printing results\n"
    labels = ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma",
                "Pancreatic Adenocarcinoma", "Breast Invasive Carcinoma",
                "Kidney Renal Clear Cell Carcinoma", "Uveal Melanoma"]
    print "Model Parameters:\nLogistic Regression, Features = 1557, Penalty = L1"
    print "Tolerance = 10E-4, C = 1.0, Maximum Iterations = 100, Fit Intercept = True"
    print "Weighted F1 Score =", f1
    print "Accuracy Score =", accuracy
    print "Confusion Matrix:"
    print cm
    print "Results printed\n"

    # display confusion matrix graph
    print "Plotting confusing matrix"
    plt.figure()
    plot_confusion_matrix(cm, classes, False, "miRNA Unnormalized Confusion Matrix")
    plt.show()

    plot_confusion_matrix(cm, classes, True, "miRNA Normalized Confusion Matrix")
    plt.show()
    print "Confusion matrix plotted\n"

    print "Trial finished\n"



if __name__ == "__main__":
    main()
