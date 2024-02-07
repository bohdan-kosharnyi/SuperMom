import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


os.environ['PATH'] = r"C:\Program Files (x86)\Graphviz\bin"


def buildSVMModel(data):
    X_cols = ['Self-realization', 'Parenting']
    X = data[X_cols]
    y = data['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    svm_ = svm.SVC(kernel='linear')
    svm_.fit(X_train, y_train)
    y_pred = svm_.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    visualizeSVMModel(svm_, X_train, y_train)


def visualizeSVMModel(model, X, y):
    plt.figure(figsize=(8, 6))

    h = .02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('SVM Decision Boundary')
    plt.show()

