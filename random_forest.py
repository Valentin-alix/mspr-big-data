from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd

def random_forest(X_train, y_train, X_test, y_test: pd.DataFrame):
    knn_model = RandomForestRegressor()
    knn_model.fit(X_train, y_train.values.ravel())

    y_pred = knn_model.predict(X_test)

    print(knn_model.score(X_test, y_test))