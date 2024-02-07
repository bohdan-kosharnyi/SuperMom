import pandas as pd

from ProcessData import readImages

from ClassificationModels.DecisionTree import buildDecisionTreeModel
from ClassificationModels.RandomForest import buildRandomForestModel
from ClassificationModels.SVM import buildSVMModel
from ClassificationModels.NeuralNetwork import buildNNModel


if __name__ == '__main__':
    # scores = readImages()
    # scores.to_csv('scores.csv', sep=';', index=False)
    scores = pd.read_csv('scores.csv', sep=';')
    buildDecisionTreeModel(scores)
    buildRandomForestModel(scores)
    buildSVMModel(scores)
    buildNNModel(scores)


