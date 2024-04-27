import os
import pprint
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from copy import deepcopy
import itertools
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)

from MeLting.training_helpers import (
    supervised_cv,
    supervised_test,
    hyperparameter_optimization,
    compute_errors,
)
from MeLting.utils import construct_features_dictionary


def random_results(model, model_name, param_grid, n_iter=1000):
    """Hyperparameter and feature optimization using grid search

    Parameters
    ----------
    model: scikit-learn model
        Scikit-learn model to be used for training
    model_name: str
        Name of the model
    param_grid : dict
        Dictionary of parameter grid to be supplied for optimization
    n_iter: int
        Number of iterations for paramater choice. Used only for RandomizedSearchCV.

    Returns
    -------
    DataFrame
        DataFrame containing all predictions, true values, errors, and decorators.

    """

    print("/////////////////////////////////////////////////////////////////////////")
    start_time = time.time()
    scores = {}  # change the variable name ONLY
    for averaging_method in tqdm(list(features.keys())):
        print(f"Averaging method: {averaging_method}")
        input_cols = features[averaging_method]
        best_model = hyperparameter_optimization(
            train_data,
            input_cols,
            output_cols,
            model,
            model_name,
            param_grid,
            scale_x=scale_x,
            scale_y=scale_y,
            cv=5,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring=scoring,
            grid="random",
            n_iter=n_iter,
        )

        best_model_estimator = best_model.best_estimator_["model"]
        print(f"Best Estimator Model: {best_model.best_estimator_}")
        print(f"Best Model Parameters: {best_model.best_params_}")
        print("<---      --->")
        model = deepcopy(best_model_estimator)

        loocv_result_db = supervised_cv(
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            cv="loocv",
            verbose=1,
        )
        loocv_scores = compute_errors(loocv_result_db)

        cv_result_db = supervised_cv(
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            cv=5,
            verbose=1,
        )
        cv_scores = compute_errors(cv_result_db)

        test_result_db = supervised_test(
            test_data,
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            verbose=1,
        )
        test_scores = compute_errors(test_result_db)

        scores[averaging_method] = [loocv_scores, cv_scores, test_scores]
        print("")
    pprint.pprint(scores)
    print("My optimization took:", time.time() - start_time)
    print("/////////////////////////////////////////////////////////////////////////")


def grid_results(model, model_name, param_grid):
    """Hyperparameter and feature optimization using random search
    Note that this is separated from the grid search in case manual tweeks are needed to be applied here

    Parameters
    ----------
    model: scikit-learn model
        Scikit-learn model to be used for training
    model_name: str
        Name of the model
    param_grid : dict
        Dictionary of parameter grid to be supplied for optimization

    Returns
    -------
    DataFrame
        DataFrame containing all predictions, true values, errors, and decorators.

    """

    print("/////////////////////////////////////////////////////////////////////////")
    start_time = time.time()

    scores = {}  # change the variable name ONLY
    for averaging_method in tqdm(list(features.keys())):
        print(f"Averaging method: {averaging_method}")
        input_cols = features[averaging_method]
        best_model = hyperparameter_optimization(
            train_data,
            input_cols,
            output_cols,
            model,
            model_name,
            param_grid,
            scale_x=scale_x,
            scale_y=scale_y,
            cv=5,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring=scoring,
            grid="grid",
        )

        best_model_estimator = best_model.best_estimator_["model"]
        print(f"Best Estimator Model: {best_model_estimator}")
        print(f"Best Model Parameters: {best_model.best_params_}")
        print("<---      --->")
        model = deepcopy(best_model_estimator)

        loocv_result_db = supervised_cv(
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            cv="loocv",
            verbose=1,
        )
        loocv_scores = compute_errors(loocv_result_db)

        cv_result_db = supervised_cv(
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            cv=5,
            verbose=1,
        )
        cv_scores = compute_errors(cv_result_db)

        test_result_db = supervised_test(
            test_data,
            train_data,
            input_cols,
            output_cols,
            descriptor_cols,
            model,
            model_name,
            scale_x=scale_x,
            scale_y=scale_y,
            output_transform=None,
            output_inverse_transform=output_inverse_transform,
            include_errors=True,
            verbose=1,
        )
        test_scores = compute_errors(test_result_db)

        scores[averaging_method] = [loocv_scores, cv_scores, test_scores]
        print("")
    pprint.pprint(scores)
    print("My optimization took:", time.time() - start_time)
    print("/////////////////////////////////////////////////////////////////////////")


if __name__ == "__main__":

    src = "../data/"
    data_file_train = "melting_temperature_and_features_training_set.csv"  # one cause use cluster-specific data here as well
    data_file_test = "melting_temperature_and_features_test_set.csv"
    train_data = pd.read_csv(os.path.join(path_data, data_file_train))
    test_data = pd.read_csv(os.path.join(path_data, data_file_test))

    # If needed, restrict the melting temperatures to below some threshold, e.g. 2000 K
    # data = data[(data['melt_temp_C']>=0) & (data['melt_temp_K']<2000)]
    # test_data = test_data[(test_data['melt_temp_C']>=0) & (test_data['melt_temp_K']<2000)]

    # this parameter set is described in utils
    train_data = train_data
    test_data = test_data
    output_cols = ["log10_melt_temp_K"]
    descriptor_cols = ["Compound"]
    features = construct_features_dictionary()
    scale_x = True
    scale_y = False
    output_inverse_transform = True

    cv = 5
    n_jobs = -1
    verbose = 1
    scoring = "neg_root_mean_squared_error"

    # complete hyperparameter optimization for different models considered

    # /////////////////////////////////Linear models////////////////////////////////////////
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    model_name = "Linear_Regression"
    param_grid = {"fit_intercept": [True, False], "positive": [True, False]}
    grid_results(model, model_name, param_grid)

    model = linear_model.Ridge()
    model_name = "Ridge_Regression"
    param_grid = {
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "fit_intercept": [True, False],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
    }
    grid_results(model, model_name, param_grid)

    model = linear_model.Lasso()
    model_name = "LASSO"
    param_grid = {
        "fit_intercept": [True, False],
        "positive": [True, False],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
    }
    grid_results(model, model_name, param_grid)

    model = linear_model.ElasticNet()
    model_name = "ElasticNet"
    param_grid = {
        "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fit_intercept": [True, False],
        "positive": [True, False],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
    }
    random_results(model, model_name, param_grid, n_iter=500)

    # /////////////////////////////////Non-linear kernel-based models////////////////////////////////////////

    model = GaussianProcessRegressor(random_state=42)
    model_name = "Gaussian Processes"
    ker_rbf = kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * kernels.RBF(
        1.0, length_scale_bounds="fixed"
    )
    ker_rq = kernels.ConstantKernel(
        1.0, constant_value_bounds="fixed"
    ) * kernels.RationalQuadratic(alpha=0.1, length_scale=1)
    ker_expsine = kernels.ConstantKernel(
        1.0, constant_value_bounds="fixed"
    ) * kernels.ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
    kernel_list = [ker_rbf, ker_rq, ker_expsine]
    param_grid = {
        "kernel": kernel_list,
        "alpha": [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-3, 1e-2],
        "normalize_y": [False, True],
    }
    random_results(model, model_name, param_grid, n_iter=500)

    model = KernelRidge()
    model_name = "Kernel Ridge Regression"
    param_grid = {
        "kernel": ["chi2", "linear", "poly", "rbf", "laplacian", "sigmoid", "cosine"],
        "gamma": [1e-4, 1e-3, 1e-2, 0.05, 1e-1, 0.5, 1, 5, 1e1, 50, 1e2, 1e3, 1e4],
        "alpha": [1e-4, 1e-3, 1e-2, 0.05, 1e-1, 0.5, 1, 5, 1e1, 50, 1e2, 1e3, 1e4],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = SVR()
    model_name = "SVR"
    param_grid = {
        "kernel": ["chi2", "linear", "poly", "rbf", "laplacian", "sigmoid", "cosine"],
        "gamma": ["scale", "auto"],
        "C": [1e-3, 1e-2, 1e-1, 0.5, 1, 5, 10, 25, 50, 75, 1e2, 1e3],
        "epsilon": [0.05, 0.1, 0.2],
    }
    random_results(model, model_name, param_grid, n_iter=500)

    # /////////////////////////////////Ensemble models////////////////////////////////////////

    model = RandomForestRegressor(n_jobs=n_jobs, random_state=42)
    model_name = "Random Forest"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=10, stop=1.25 * len(train_data), num=10)
        ],
        "max_depth": [int(x) for x in np.linspace(2, 70, num=7)] + [None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"],
        "bootstrap": [True, False],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = GradientBoostingRegressor(n_jobs=n_jobs, random_state=42)
    model_name = "Gradient Boosting Regression"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=10, stop=1.25 * len(train_data), num=10)
        ],
        "learning_rate": [x for x in np.linspace(0.02, 1.72, num=7)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [int(x) for x in np.linspace(2, 70, num=7)] + [None],
        "max_features": ["auto", "sqrt"],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = AdaBoostRegressor()
    model_name = "AdaBoost Regression"
    param_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=10, stop=len(data), num=10)],
        "learning_rate": [x for x in np.linspace(0.5, 1.2, num=4)] + [None],
        "loss": ["linear", "square"],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = ExtraTreesRegressor(n_jobs=n_jobs, random_state=42)
    model_name = "Extra Trees Regression"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=10, stop=1.25 * len(train_data), num=10)
        ],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [int(x) for x in np.linspace(2, 70, num=7)] + [None],
        "max_features": ["auto", "sqrt"],
    }
    random_results(model, model_name, param_grid, n_iter=500)

    model = HistGradientBoostingRegressor(random_state=42)
    model_name = "Histogram Gradient Boosting Regression"
    param_grid = {
        "max_iter": [
            int(x) for x in np.linspace(start=10, stop=1.25 * len(train_data), num=10)
        ],
        "learning_rate": [x for x in np.linspace(0.02, 1.72, num=7)],
        "l2_regularization": [0, 0.33, 0.66, 1],
        "max_depth": [int(x) for x in np.linspace(2, 70, num=7)] + [None],
        "min_samples_leaf": [1, 2, 4],
    }
    random_results(model, model_name, param_grid, n_iter=500)

    model = lightgbm.LGBMRegressor(boosting_type="gbdt", n_jobs=n_jobs, random_state=42)
    model_name = "LGBMRegressor/GBDT"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=5, stop=1.25 * len(train_data), num=10)
        ],
        "max_depth": [int(x) for x in np.linspace(1, 20, num=7)] + [None],
        #'min_child_weight':[4,5],
        "learning_rate": [i / 10.0 for i in range(3, 6)],
        "subsample": [i / 10.0 for i in range(8, 11)],
        "colsample_bytree": [i / 10.0 for i in range(8, 11)],
        "reg_alpha": [0, 0.25],
        "reg_lambda": [0, 0.25],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = lightgbm.LGBMRegressor(boosting_type="rf", n_jobs=n_jobs, random_state=42)
    model_name = "LGBMRegressor/RF"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=5, stop=1.25 * len(train_data), num=10)
        ],
        "max_depth": [int(x) for x in np.linspace(1, 20, num=7)] + [None],
        #'min_child_weight':[4,5],
        "learning_rate": [i / 10.0 for i in range(3, 6)],
        "subsample": [i / 10.0 for i in range(8, 11)],
        "colsample_bytree": [i / 10.0 for i in range(8, 11)],
        "reg_alpha": [0, 0.25],
        "reg_lambda": [0, 0.25],
        "bagging_freq": [1],
        "bagging_fraction": [0.9],
    }
    random_results(model, model_name, param_grid, n_iter=1000)

    model = xgb.XGBRegressor(n_jobs=n_jobs)
    model_name = "XGBRegressor"
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(start=5, stop=1.25 * len(train_data), num=10)
        ],
        "max_depth": [int(x) for x in np.linspace(1, 20, num=7)] + [None],
        #'min_child_weight':[4,5],
        "gamma": [i / 10.0 for i in range(3, 6)],
        "subsample": [i / 10.0 for i in range(6, 11)],
        "colsample_bytree": [i / 10.0 for i in range(6, 11)],
        "eta": [0.3, 0.2, 0.1, 0.05, 0.01, 0.005],
    }
    random_results(model, model_name, param_grid, n_iter=1000)
