from sklearn.ensemble import RandomForestRegressor

def random_forest(X_train, y_train, X_test, y_test):
    knn_model = RandomForestRegressor()
    knn_model.fit(X_train, y_train.values.ravel())
    print(knn_model.score(X_test, y_test))