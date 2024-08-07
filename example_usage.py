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
from data.datasets import scale_data, get_circles, get_air_quality,get_moons
from sklearn.model_selection import train_test_split
from experiments_common import evaluate_on_coreset, evaluate_on_full_data

if __name__ == "__main__":
    # Data
    # save resutls csv
    with open("results.csv", "w") as f:
        f.write("epsilon,coreset_size,use_exact_bicriteria,full_data_error,uniform_sample_error,coreset_error,smoothed_coreset_error\n")
    X, Y = get_air_quality(5000)
    X, Y = scale_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    data_train = SparseData(X_train, Y_train)

    # Coreset construction for different epsilons.
    # Note: in practice we get much smaller error than the given epsilon.
    # Tune the epsilon to get the desired coreset size and check for practical
    # error on the validation set.
    # epsilons = [0.04, 0.07, 0.1]
    # 0.0213
    epsilons = [0.0213,0.0172]
    k = 100
    coreset_verbose = True # True for printing additional information

    # Use_exact_bicriteria = True will result in training a single decision
    # tree on full data for evaluating problem complexity (approximating
    # the optimal cost, called OPT, on given data).
    # Otherwise, OPT is evaluated using (alpha, beta)-bicriteria approximation.
    # In practice when using coreset for hyperparameters tuning, where
    # hundreds of hyperparameters are checked, training a single decision
    # tree doesn't increase much the computational time, but allows to obtain
    # better coreset, increasing overall accuracy.
    use_exact_bicriteria_values=[True]
    for use_exact_bicriteria in use_exact_bicriteria_values:
        print("\nConstructing coresets using exact bicriteria: {}\n".format(
            use_exact_bicriteria))
        for epsilon in epsilons:
            print("\nConstructing coreset for epsilon = {}\n".format(epsilon))

            # coreset construction
            coreset, coreset_smoothed = dt_coreset(
                data_train, k, epsilon, verbose=coreset_verbose,
                use_exact_bicriteria=use_exact_bicriteria)
            # pick a uniform random sample from the full train data of the same size as corese
            uniform_sample = data_train.get_sample(coreset.size)
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
            param_grid = {'max_leaf_nodes': [1000, 5000, 10000,50000]}
            best_k_unif,error_unif = evaluate_on_coreset(
                uniform_sample, X_train, Y_train, X_test, Y_test, param_grid)
            best_k_full,error_full = evaluate_on_full_data(
                X_train, Y_train, X_test, Y_test, param_grid)
            best_k_coreset,error_coreset = evaluate_on_coreset(
                coreset, X_train, Y_train, X_test, Y_test, param_grid)
            best_k_smooth_coreset,error_coreset_smoothed = evaluate_on_coreset(
                coreset_smoothed, X_train, Y_train, X_test, Y_test, param_grid)
            # error_full = 2
            # error_coreset = 2
            # error_coreset_smoothed = 2
            #
            # for param in range(1000, 10001, 1000):
            #     error_full = min(evaluate_on_full_data(X_train, Y_train, X_test, Y_test, param), error_full)
            # for param in range(1000, 10001, 1000):
            #     error_coreset = min(evaluate_on_coreset(coreset, X_train, Y_train, X_test, Y_test, param),
            #                         error_coreset)
            # for param in range(1000, 10001, 1000):
            #     error_coreset_smoothed = min(
            #         evaluate_on_coreset(coreset_smoothed, X_train, Y_train, X_test, Y_test, param),
            #         error_coreset_smoothed)

            # printing the results
            print(("Using 100% of the training set ({} examples):\n" +
                   "\tTesting error (full data):\t\t{:.5f}").format(
                      len(X_train), error_full))
            print(("Using of the training set " +
                  " (uniform sample of examples):\n" +
                  "\tTesting error (uniform sample):\t{:.5f}").format(
                       error_unif))
            print(("Using {:.2f}% of the training set " +
                  " (coreset of {} examples):").format(
                      coreset.size / float(len(X_train)) * 100,
                      coreset.size))
            print("\tTesting error (original coreset):\t{:.5f}".format(
                  error_coreset))
            print("\tTesting error (smoothed coreset):\t{:.5f}".format(
                  error_coreset_smoothed))
            # add the results to a csv file
            with open("results.csv", "a") as f:
                f.write("{},{},{},{},{},{}\n".format(epsilon,use_exact_bicriteria,error_full,error_unif,error_coreset,error_coreset_smoothed))
