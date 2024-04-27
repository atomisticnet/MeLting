from copy import deepcopy
import numpy as np
import pandas as pd
from math import sqrt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


def supervised_cv(
    data,
    input_cols,
    output_cols,
    descriptor_cols,
    model,
    model_name,
    scale_x=True,
    scale_y=False,
    output_transform=None,
    output_inverse_transform=None,
    include_errors=True,
    cv=10,
    scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error"],
    verbose=1,
    n_jobs=-1,
):
    """Supervised learning regression scores of training data using cross-validation

    Parameters
    ----------
    data : DataFrame
        The DataFrame object containing all of the required data
    input_cols : list
        List of feature columns
    output_cols : list
        List of output columns (e.g., ['log10_melt_temp_K'])
    descriptor_cols: list
        List of decorators to be added to the final prediction DataFrame. These do no participate in the training.
    model: scikit-learn model
        Scikit-learn model to be used for training
    model_name: str
        Name of the model
    scale_x: bool, optionaal
        Whether to scale feaatures with StandardScaler from scikit-learn (default is True)
    scale_y: bool, optional
        Whether to scale outputs with StandardScaler from scikit-learn (default is False)
    output_transform: function
        How to transform the output (default is None). We are directly using log10 tranformed output.
    output_inverse_transform: function
        How to inverse transform the output for reporting and error quantification (default is None)
    include_errors: bool
        Whether to include absolute and relative errors in the final reported DataFrame (default is True)
    cv: int or str
        How many cross-validation splits is required (default is 10)
    scoring: list or str
        List or str of scoring method used for hyperparameter optimization
    verbose: int
        Whether to print error metrics. If verbose <=0, then don't, and do if verbose >0. (default is True)
    n_jobs: int
        The number of jobs to run in parallel. None means 1. -1 means using all processors. (default is -1)

    Returns
    -------
    DataFrame
        DataFrame containing all predictions, true values, errors, and decorators.
    """

    output_predictions = []

    # special treatment is required for loocv
    if cv == "loocv":
        for idx in range(data.shape[0]):

            train_db = data.loc[data.index != idx, :]
            test_db = data.iloc[[idx]]

            if scale_x:
                x_scaler = StandardScaler().fit(train_db[input_cols])
                X_train = x_scaler.transform(train_db[input_cols])
                X_test = x_scaler.transform(test_db[input_cols])
            else:
                X_train = train_db[input_cols]
                X_test = test_db[input_cols]

            if scale_y:
                y_scaler = StandardScaler().fit(train_db[output_cols])
                Y_train = y_scaler.transform(train_db[output_cols])
            else:
                Y_train = train_db[output_cols].values.flatten()
            Y_test = test_db[output_cols].values.flatten()

            descriptor_test = test_db[descriptor_cols].values.flatten()

            regr = deepcopy(model)
            regr = regr.fit(X_train, np.ravel(Y_train))
            prediction = regr.predict(X_test)

            if scale_y:
                prediction = y_scaler.inverse_transform(prediction)

            if output_inverse_transform:
                prediction = 10**prediction
                Y_test = 10**Y_test

            output_predictions.append(
                [Y_test[0], round(prediction[0], 2), descriptor_test[0]]
            )

        predictions_db = pd.DataFrame(
            output_predictions, columns=["exp", "ml", *descriptor_cols]
        )

        if include_errors:
            predictions_db["error"] = round(
                abs(predictions_db["exp"] - predictions_db["ml"]), 2
            )
            predictions_db["rel_error"] = round(
                abs(
                    (predictions_db["exp"] - predictions_db["ml"])
                    / predictions_db["exp"]
                ),
                2,
            )

        output_true = predictions_db["exp"].values
        output_ml = predictions_db["ml"].values
        RMSE = sqrt(mean_squared_error(output_true, output_ml))
        MAE = mean_absolute_error(output_true, output_ml)

    elif isinstance(cv, int):

        X, y = data[input_cols].values, data[output_cols].values.flatten()
        descriptor = data[descriptor_cols].values

        if scale_x and scale_y:
            estimator = TransformedTargetRegressor(
                regressor=model, transformer=StandardScaler()
            )
            pipeline = Pipeline([("scale_x", StandardScaler()), ("model", estimator)])
        elif scale_x and not scale_y:
            pipeline = Pipeline([("scale_x", StandardScaler()), ("model", model)])
        elif not scale_x and scale_y:
            estimator = TransformedTargetRegressor(
                regressor=model, transformer=StandardScaler()
            )
            pipeline = Pipeline([("model", estimator)])
        else:
            pipeline = Pipeline([("model", model)])

        predictions = cross_val_predict(
            pipeline, X, y, cv=cv, n_jobs=n_jobs, verbose=verbose
        )

        if output_inverse_transform:
            predictions = [10**i for i in predictions.flatten()]
            y = [10**i for i in y]

        predictions_db = pd.DataFrame(columns=["exp", "ml", *descriptor_cols])
        predictions_db["ml"] = [round(i, 2) for i in predictions]
        predictions_db["exp"] = y
        predictions_db[descriptor_cols] = descriptor

        if include_errors:
            predictions_db["error"] = round(
                abs(predictions_db["exp"] - predictions_db["ml"]), 2
            )
            predictions_db["rel_error"] = round(
                abs(
                    (predictions_db["exp"] - predictions_db["ml"])
                    / predictions_db["exp"]
                ),
                2,
            )

    else:
        raise ValueError("Please choose between LOO-CV and integer-CV.")

    RMSE, MAE, MAPE, R2 = compute_errors(predictions_db, "exp", "ml")
    if verbose > 0:
        print("{}: RMSE of {:.0f} using {}-CV.".format(model_name, RMSE, cv))
        print("{}: MAE of {:.0f} using {}-CV.".format(model_name, MAE, cv))
        print("{}: MAPE of {:.0f}% using {}-CV.".format(model_name, MAPE, cv))
        print("{}: R2 of {:.2f} using {}-CV.".format(model_name, R2, cv))

    return predictions_db


def supervised_test(
    test_data,
    train_data,
    input_cols,
    output_cols,
    descriptor_cols,
    model,
    model_name,
    scale_x=True,
    scale_y=False,
    output_transform=None,
    output_inverse_transform=None,
    include_errors=True,
    verbose=1,
):
    """Supervised learning regression scores on test data

    Parameters
    ----------
    test_data : DataFrame
        The DataFrame object containing all of the required test set data
    train_data : DataFrame
        The DataFrame object containing all of the required training set data

    Returns
    -------
    predictions_db: DataFrame
        DataFrame containing all predictions, true values, errors, and decorators.
    """

    if scale_x:
        x_scaler = StandardScaler().fit(train_data[input_cols])
        X_train = x_scaler.transform(train_data[input_cols])
        X_test = x_scaler.transform(test_data[input_cols])
    else:
        X_train = train_data[input_cols]
        X_test = test_data[input_cols]

    if scale_y:
        y_scaler = StandardScaler().fit(train_data[output_cols])
        Y_train = y_scaler.transform(train_data[output_cols])
    else:
        Y_train = train_data[output_cols].values.flatten()
    Y_test = test_data[output_cols].values.flatten()

    descriptor_test = test_data[descriptor_cols].values.flatten()

    regr = deepcopy(model)
    regr = regr.fit(X_train, np.ravel(Y_train))
    prediction = regr.predict(X_test)

    if scale_y:
        prediction = y_scaler.inverse_transform(prediction)

    if output_inverse_transform:
        prediction = [10**i for i in prediction]
        Y_test = [10**i for i in Y_test]

    predictions_db = pd.DataFrame(
        {"exp": Y_test, "ml": prediction, "Compound": descriptor_test}
    )

    if include_errors:
        predictions_db["error"] = round(
            abs(predictions_db["exp"] - predictions_db["ml"]), 2
        )
        predictions_db["rel_error"] = round(
            abs((predictions_db["exp"] - predictions_db["ml"]) / predictions_db["exp"]),
            2,
        )

    RMSE, MAE, MAPE, R2 = compute_errors(predictions_db, "exp", "ml")
    if verbose > 0:
        print("{}: RMSE of {:.0f} on test data.".format(model_name, RMSE))
        print("{}: MAE of {:.0f} on test data.".format(model_name, MAE))
        print("{}: MAPE of {:.0f}% on test data.".format(model_name, MAPE))
        print("{}: R2 of {:.2f} on test data.".format(model_name, R2))

    return predictions_db


def hyperparameter_optimization(
    data,
    input_cols,
    output_cols,
    model,
    model_name,
    param_grid,
    scale_x=True,
    scale_y=False,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring="neg_mean_absolute_error",
    search_type="grid",
    n_iter=100,
):
    """Hyperparameter Tuning using CV

    Parameters
    ----------
    param_grid : dict
        Dictionary of parameter grid to be supplied for optimization
    grid : str
        If "grid", use GridSearchCV. If "random", use RandomizedSearchCV.
    n_iter: int
        Number of iterations for paramater choice. Used only for RandomizedSearchCV.

    Returns
    -------
    DataFrame
        DataFrame containing all predictions, true values, errors, and decorators.
    """

    X, Y = data[input_cols].values, data[output_cols].values

    if scale_x and scale_y:
        estimator = TransformedTargetRegressor(regressor=model, func=StandardScaler())
        pipeline = Pipeline([("scale_x", StandardScaler()), ("model", estimator)])
        param_grid = {"model__regressor__" + k: v for k, v in param_grid.items()}
    elif scale_x and not scale_y:
        pipeline = Pipeline([("scale_x", StandardScaler()), ("model", model)])
        param_grid = {"model__" + k: v for k, v in param_grid.items()}
    elif not scale_x and scale_y:
        estimator = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        pipeline = Pipeline([("model", estimator)])
        param_grid = {"regressor__" + k: v for k, v in param_grid.items()}
    else:
        pipeline = Pipeline([("model", model)])
        param_grid = {"model__" + k: v for k, v in param_grid.items()}

    if search_type == "grid":
        regr = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            cv=cv,
        )

    elif search_type == "random":
        regr = RandomizedSearchCV(
            estimator=pipeline,
            n_iter=n_iter,
            param_distributions=param_grid,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            cv=cv,
            random_state=1,
        )
    else:
        raise ValueError("Please choose between GridSearchCV and RandomizedSearchCV.")

    regr = regr.fit(X, np.ravel(Y))

    if verbose > 0:
        print(f"Best score: {abs(regr.best_score_)}")
        print("Best parameters set: ")
        best_parameters = regr.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print(f"\t{param_name}: {best_parameters[param_name]}")

    return regr


def compute_errors(db, true_col="exp", pred_col="ml"):
    """Computing errors between experimental and predicted values

    Parameters
    ----------
    db : DataFrame
        The DataFrame object containing all of the required data
    true_col : str
        The name of the experimental data (default is "exp")
    pred_col : str
        The name of the column containing predicted data (default is "ml")

    Returns
    -------
    list
        A list containing all error metrics
    """

    output_true = db[true_col].values
    output_ml = db[pred_col].values
    RMSE = int(sqrt(mean_squared_error(output_true, output_ml)))
    MAE = int(mean_absolute_error(output_true, output_ml))
    MAPE = int(mean_absolute_percentage_error(output_true, output_ml) * 100)
    R2 = round(r2_score(output_true, output_ml), 2)
    return [RMSE, MAE, MAPE, R2]
