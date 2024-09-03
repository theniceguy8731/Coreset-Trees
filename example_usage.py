'''
    This is an example code that shows how to construct the coreset,
    train a model on coreset data, and evaluate the results.

*******************************************************************************
MIT License

Copyright (c) 2021 Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda,
                   Ilan Newman, Dan Feldman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************
'''

####################################### NOTES #################################
# - Please cite our paper when using the code:
#             "Coresets for Decision Trees of Signals" (NeurIPS'21)
#             Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda,
#             Ilan Newman, Dan Feldman
#
###############################################################################


from coreset.decision_tree import dt_coreset
from coreset.utils.formats import SparseData
from data.datasets import scale_data, get_circles, get_air_quality,get_moons,get_gesture_phase
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
def evaluate_on_full_data(X_train, Y_train, X_test, Y_test, k):
    ''' Model training on full data to compare with coreset results,
        returns mean squared error on the testing set after training
        on the full training dataset '''
    model = RandomForestRegressor(max_leaf_nodes=k)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return mean_squared_error(Y_test, Y_pred)

def evaluate_on_coreset(coreset, X_train, Y_train, X_test, Y_test, k):
    ''' Model training on coreset, returns mean squared error on the
        testing set after training on a subset of the training dataset '''
    X_coreset, Y_coreset, weights = coreset.X, coreset.Y, coreset.weights
    model_coreset = RandomForestRegressor(max_leaf_nodes=k)
    model_coreset.fit(X_coreset, Y_coreset, sample_weight=weights)
    Y_pred_coreset = model_coreset.predict(X_test)
    return mean_squared_error(Y_test, Y_pred_coreset)

if __name__ == "__main__":
    # Data
    # X, Y = get_circles(50000, 300)
    X, Y = get_gesture_phase(99000)
    # X, Y = get_air_quality(9358)
    X, Y = scale_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    data_train = SparseData(X_train, Y_train)

    # Coreset construction for different epsilons.
    # Note: in practice we get much smaller error than the given epsilon.
    # Tune the epsilon to get the desired coreset size and check for practical
    # error on the validation set.
    epsilons = [0.04, 0.07, 0.1, 0.15]
    k = 20
    coreset_verbose = True  # True for printing additional information

    # Use_exact_bicriteria = True will result in training a single decision
    # tree on full data for evaluating problem complexity (approximating
    # the optimal cost, called OPT, on given data).
    # Otherwise, OPT is evaluated using (alpha, beta)-bicriteria approximation.
    # In practice when using coreset for hyperparameters tuning, where
    # hundreds of hyperparameters are checked, training a single decision
    # tree doesn't increase much the computational time, but allows to obtain
    # better coreset, increasing overall accuracy.
    use_exact_bicriteria_values = [True]
    for use_exact_bicriteria in use_exact_bicriteria_values:
        print("\nConstructing coresets using exact bicriteria: {}\n".format(
            use_exact_bicriteria))
        for epsilon in epsilons:
            print("\nConstructing coreset for epsilon = {}\n".format(epsilon))

            # coreset construction
            coreset, coreset_smoothed = dt_coreset(
                data_train, k, epsilon, verbose=coreset_verbose,
                use_exact_bicriteria=use_exact_bicriteria)

            # evaluation: training the models and calculating errors
            '''
            To obtain theoretically proven error, either one of two conditions
            must be met:
                1. Smoothed coreset (with duplicated data points) is used with
                   the original DecisionTree model. Any existing model, such as
                   DecisionTreeRegressor from sklearn, or LGBRegressor from
                   LightGBM, can be used without modifying its cost function.
                2. Original coreset is used, but the DecisionTree model is
                   modified to work with the Fitting-Loss algorithm as the
                   cost function, as described in the supplementary materials
                   of the paper. In this case, any model such as
                   DecisionTreeRegressor or LightGBM regressor can be used,
                   but the class must be modified to work with an updated
                   Fitting-Loss cost function.
            Because we do not modify the sklearn's DecisionTreeRegressor class,
            smoothed coreset must be used to obtain better results. However,
            in practice the original coreset also obtains good approximation
            for smaller values of epsilon.
            '''
            error_full = evaluate_on_full_data(
                X_train, Y_train, X_test, Y_test, k)
            error_coreset = evaluate_on_coreset(
                coreset, X_train, Y_train, X_test, Y_test, k)
            error_coreset_smoothed = evaluate_on_coreset(
                coreset_smoothed, X_train, Y_train, X_test, Y_test, k)

            # printing the results
            print(("Using 100% of the training set ({} examples):\n" +
                   "\tTesting error (full data):\t\t{:.5f}").format(
                len(X_train), error_full))
            print(("Using {:.2f}% of the training set " +
                   " (coreset of {} examples):").format(
                coreset.size / float(len(X_train)) * 100,
                coreset.size))
            print("sum of the weights of the coreset: ",sum(coreset.weights))
            print("\tTesting error (original coreset):\t{:.5f}".format(
                error_coreset))
            print("\tTesting error (smoothed coreset):\t{:.5f}".format(
                error_coreset_smoothed))