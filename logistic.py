import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import pandas as pd
from scipy import stats
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

def data_prep(fs_choice,k):
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

    return X_train, X_test, y_train, y_test

def run_fs(k,fs_choice,params):
    X_train, X_test, y_train, y_test = data_prep(fs_choice,k)

    # pick and train model
    clf = LogisticRegression()
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 = f1_score(y_test,predicted, average = "weighted")
    acc = accuracy_score(y_test,predicted)

    return k,f1,acc

def tune_params(k,fs_choice):
    X_train, X_test, y_train, y_test = data_prep(fs_choice,k)

    # pick and train model
    model = LogisticRegression()
    # params to tune
    params = {"penalty": ["l1", "l2"],
               "tol": stats.uniform(0.00000001,0.001),
               "C": stats.uniform(0.01,100) }

    # run randomized search
    n_iter_search = 10
    clf = RandomizedSearchCV(model, param_distributions=params,
                                       n_iter=n_iter_search, n_jobs = 16)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 = f1_score(y_test,predicted, average = "weighted")
    acc = accuracy_score(y_test,predicted)

    print "\tBest result from Tunning: %d, %.5f, %.5f" % (k,f1,acc)
    return clf.best_params_

def select_features(params,fn):
    choice = int(sys.argv[1])
    # Generate feature counts to try
    feature_count = [1]
    while(feature_count[-1]<1881):
        feature_count.append(feature_count[-1]+4)

    best_k,best_f1,best_acc = 0,0,0
    with open(fn,"w") as results:
        for i in feature_count[::50]:
            k,f1,acc = run_fs(i,choice,params)
            results.write("%d, %.5f, %.5f\n" % (k,f1,acc))
            if (best_f1+best_acc < f1+acc):
                best_k,best_f1,best_acc = k,f1,acc
    print "\tBest Result from FS: %d, %.5f, %.5f" % (best_k,best_f1,best_acc)
    return best_k,best_f1,best_acc

def read_data(fn):
    data = []
    with open(fn,"r") as fl:
        for line in fl:
            data.append([float(i.strip()) for i in line.split(",")])
    return data

def main():
    fs_choice = int(sys.argv[-2])
    base_fn = sys.argv[-1]
    iters = 0
    params = {}
    while (iters < 3):
        curr_fn = base_fn+str(iters+1)+".txt"
        print "Iteration #%d - writting results to %s" % (iters+1, curr_fn)
        k,f1,acc = select_features(params, curr_fn)
        params = tune_params(k, fs_choice)
        iters += 1

if __name__ == "__main__":
    main()
