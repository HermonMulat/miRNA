from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# for feature selection using variance
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
import sys

FEATURE = 0
LABEL = 1

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

    # pick features
    # threshold = float(sys.argv[1])
    # df_train,df_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    # index_choice = sf.pick_features(threshold,X_train)
    # X_train,X_test = df_train[index_choice], df_test[index_choice]

    # feature selection based on variance
    # threshold = float(sys.argv[1])
    # selector = VarianceThreshold(threshold=threshold)
    # selector.fit(X_train)
    # X_train, X_test = selector.transform(X_train), selector.transform(X_test)
    # print "Selected Feature count:",len(X_train[0])

    # selector = SelectKBest(score_func = chi2, k=int(sys.argv[1]))
    # selector.fit(X_train,y_train)
    # X_train, X_test = selector.transform(X_train),selector.transform(X_test)
    # print "Selected feature count:", len(picked_features[0])

    '''
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors([X_train[0]])
    print indices
    print distances
    '''
    n_neighbors = 5

    clf_neigh = KNeighborsClassifier(n_neighbors, weights="distance", p = 1)
    clf_neigh.fit(X_train, y_train)
    #print(y_train[0])

    #print(clf_neigh.predict([X_train[0]]))
    #print(clf_neigh.predict_proba([X_train[0]]))
    predicted = clf_neigh.predict(X_test)
    print("F1 score with %d neighbors" % (n_neighbors)),
    for i in f1_score(y_test,predicted, average = None):
        print ("%.4f" % i),
    print

    print("Accuracy score with %d neighbors: %.4f" % (n_neighbors,accuracy_score(y_test,predicted)))

if __name__ == "__main__":
    main()
