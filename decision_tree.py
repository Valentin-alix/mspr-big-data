from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def decision_tree(X_train, y_train, X_test, y_test):
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train, y_train)
    print(decision_tree_model.score(X_test, y_test))