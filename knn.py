from sklearn.neighbors import KNeighborsRegressor

def knn(X_train, y_train, X_test, y_test):
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    print(knn_model.score(X_test, y_test))