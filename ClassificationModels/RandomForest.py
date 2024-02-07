import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import graphviz


os.environ['PATH'] = r"C:\Program Files (x86)\Graphviz\bin"


def buildRandomForestModel(data):
    X_cols = ['Parenting', 'Self-realization']
    X = data[X_cols]
    y = data['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    param_dist = {'n_estimators': [20, 50, 100, 200, 500], 'max_depth': [1, 5, 10, 15, 20]}
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_
    print('Best hyperparameters:', rand_search.best_params_)

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    visualizeRandomForestModel(best_rf, X_cols)
    showParametersImportance(best_rf, X_cols)


def visualizeRandomForestModel(rf, X_cols):
    for i in range(2):
        tree = rf.estimators_[i]
        dot_data = StringIO()
        export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                        feature_names=X_cols, class_names=[str(i) for i in range(1, 3)])
        plot = pydotplus.graph_from_dot_data(dot_data.getvalue())
        plot.write_png(f'ModelsPlots/supermom_forest_{i+1}.png')
        Image(plot.create_png())


def showParametersImportance(best_rf, X_cols):
    feature_importances = pd.Series(best_rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    print(feature_importances)
