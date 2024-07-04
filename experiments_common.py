from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def evaluate_on_full_data(X_train, Y_train, X_test, Y_test, param_grid):
    ''' Perform grid search on full data to find the best max_leaf_nodes,
        returns best parameters and mean squared error on the testing set after training
        on the full training dataset '''
    model = RandomForestRegressor(n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(mean_squared_error), cv=5)
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    return grid_search.best_params_, mean_squared_error(Y_test, Y_pred)


def evaluate_on_coreset(coreset, X_train, Y_train, X_test, Y_test, param_grid):
    ''' Perform grid search on coreset to find the best max_leaf_nodes,
        returns best parameters and mean squared error on the testing set after training
        on a subset of the training dataset '''
    X_coreset, Y_coreset, weights = coreset.X, coreset.Y, coreset.weights
    model_coreset = RandomForestRegressor(n_jobs=-1)
    grid_search = GridSearchCV(estimator=model_coreset, param_grid=param_grid, scoring=make_scorer(mean_squared_error),
                               cv=5)
    grid_search.fit(X_coreset, Y_coreset, sample_weight=weights)

    best_model_coreset = grid_search.best_estimator_
    Y_pred_coreset = best_model_coreset.predict(X_test)
    return grid_search.best_params_, mean_squared_error(Y_test, Y_pred_coreset)


# Define your parameter grid

# Example usage:
# best_params_full, mse_full = evaluate_on_full_data(X_train, Y_train, X_test, Y_test, param_grid)
# best_params_coreset, mse_coreset = evaluate_on_coreset(coreset, X_train, Y_train, X_test, Y_test, param_grid)
