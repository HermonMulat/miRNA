import matplotlib.pyplot as plt
import numpy as np
import sys

def read_data(fn):
    data = []
    with open(fn,"r") as fl:
        for line in fl:
            data.append([float(i.strip()) for i in line.split(",")])
    return data

def main():
    data1 = read_data(sys.argv[1])  # Logistic
    data2 = read_data(sys.argv[2])  # KNN
    data3 = read_data(sys.argv[3])  # Decision Tree

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    #plot accuracy vs feature count
    x = [row[0] for row in data1]

    f1_1, f1_2, f1_3 = [row[1] for row in data1],[row[1] for row in data2], [row[1] for row in data3]
    ax1.set_title('Weighted F1-Score vs Feature Count')
    ax1.scatter(x,f1_1, s=10, color = 'green', label = "Logistic Regression")
    ax1.scatter(x,f1_2, s=10, color = 'blue', label = "KNN")
    ax1.scatter(x,f1_3, s=10, color = 'red', label = "Decision Tree")
    ax1.legend()

    #plot F1-score
    acc_1, acc_2, acc_3 = [row[2] for row in data1],[row[2] for row in data2],[row[2] for row in data3]
    ax2.set_title('Accuracy vs Feature Count')
    ax2.scatter(x,acc_1, s=10, color = 'green', label = "Logistic Regression")
    ax2.scatter(x,acc_2, s=10, color = 'blue', label = "KNN")
    ax2.scatter(x,acc_3, s=10, color = 'red', label = "Decision Tree")
    ax2.legend()

    f.subplots_adjust(hspace=0.3)
    plt.show()

if __name__ == '__main__':
    main()
