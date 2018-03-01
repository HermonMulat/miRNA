import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
# for feature selection using variance
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import pandas as pd
import sys

def fs_variance(k,X_train,X_test):
    # Pick k features with the highest variance
    selector = VarianceThreshold()
    selector.fit(X_train)
    col_count = X_train.shape[-1]
    feature_var = [(i,selector.variances_[i]) for i in xrange(col_count)]
    feature_var.sort(key = lambda x:x[-1]) # sort by variance
    picked_featuresIndex  = [i[0] for i in feature_var[-1*k:]]
    X_train,X_test = X_train[picked_featuresIndex], X_test[picked_featuresIndex]

    return X_train,X_test

def fs_chisquared(k,X_train,X_test,y_train):
    # feature selection
    selector = SelectKBest(score_func = chi2, k=k)
    selector.fit(X_train,y_train)
    X_train, X_test = selector.transform(X_train),selector.transform(X_test)

    return X_train, X_test

def run(k,fs_choice):
    train_set = pd.read_csv('training.csv', header=None)
    last_col = train_set.shape[-1] - 1
    Y = train_set[last_col]
    X = train_set.drop([last_col], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(
                X, Y,test_size=0.2, random_state=42)

    # Feature Selection + Standardization
    if (fs_choice == 1):
        X_train, X_test = fs_variance(k,X_train,X_test)
    else:
        X_train, X_test = fs_chisquared(k,X_train,X_test,y_train)

    # Stadardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train,X_test = scaler.transform(X_train), scaler.transform(X_test)

    # pick and train model
    clf = KNeighborsClassifier(n_neighbors = 3, p = 1, n_jobs = 8)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 = f1_score(y_test,predicted, average = "weighted")
    acc = accuracy_score(y_test,predicted)

    print "%d, %.5f, %.5f" % (k,f1,acc)

def main():
    choice = int(sys.argv[1])
    feature_count = [1]
    while(feature_count[-1]<1881):
        feature_count.append(feature_count[-1]+4)
    for i in feature_count[]:
        run(i,choice)

if __name__ == "__main__":
    main()
