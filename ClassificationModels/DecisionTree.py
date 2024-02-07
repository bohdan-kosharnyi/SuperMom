import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import graphviz

os.environ['PATH'] = r"C:\Program Files (x86)\Graphviz\bin"


def buildDecisionTreeModel(data):
    X_cols = ['Parenting', 'Self-realization']
    X = data[X_cols]
    y = data['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    tree = DecisionTreeClassifier()
    tree = tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    visualizeDecisionTreeModel(tree, X_cols)


def visualizeDecisionTreeModel(tree, X_cols):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=X_cols, class_names=[str(i) for i in range(1, 3)])
    plot = pydotplus.graph_from_dot_data(dot_data.getvalue())
    plot.write_png('ModelsPlots/supermom_tree.png')
    Image(plot.create_png())
