import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
# for feature selection using variance
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import pandas as pd
import sys

def main():
    train_set = pd.read_csv('training.csv', header=None)
    last_col = train_set.shape[-1] - 1
    Y = train_set[last_col]
    X = train_set.drop([last_col], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(
                X, Y,test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train,X_test = scaler.transform(X_train), scaler.transform(X_test)
    C = 1
    #clf_l2_LR_br = LogisticRegression(C=C, penalty='l2', tol=0.01, solver="sag", multi_class="ovr")
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    #clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver="sag", multi_class="multinomial")

    clf_l1_LR.fit(X_train, y_train)
    print("C=%.2f" % C)
    # print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
    predicted = clf_l1_LR.predict(X_test)
    print"F1 score with L1 penalty: ",
    for i in f1_score(y_test,predicted, average = None):
        print ("%.4f"%i),
    print
    print("Accuracy score with L1 penalty: %.4f" % accuracy_score(y_test,predicted))

if __name__ == "__main__":
    main()
