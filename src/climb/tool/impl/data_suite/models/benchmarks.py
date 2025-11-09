import logging
import random

import numpy as np
import pandas as pd
import torch
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc, RegressorNc, RegressorNormalizer
from sklearn import gaussian_process, linear_model
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from climb.tool.impl.data_suite.third_party.uq360.algorithms.quantile_regression import QuantileRegression
from climb.tool.impl.data_suite.utils.uncertainty_metrics import (
    compute_uncertainty_metrics,
    process_results,
    test_ood,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def sample_copula(x_train, y_train, x_test, y_test, inlier_ids):
    from climb.tool.impl.data_suite.third_party.copulas.multivariate import GaussianMultivariate

    if len(y_train.shape) == 1:
        y_train = np.expand_dims(y_train, axis=1)

    if len(y_test.shape) == 1:
        y_test = np.expand_dims(y_test, axis=1)

    data_train = np.hstack((x_train, y_train))

    columns = [f"x{i + 1}" for i in range(data_train.shape[1])]
    df = pd.DataFrame(data=data_train, columns=columns)
    dist = GaussianMultivariate()

    dist.fit(df)

    data = np.hstack((x_test, y_test))
    data = data[inlier_ids, :]

    means = []
    sigmas = []
    n_samples = 100

    for idx in tqdm(range(data.shape[0])):
        conditions_dict = {}

        for i in range(x_train.shape[1]):
            conditions_dict[f"x{i + 2}"] = data[idx, i + 1]

        values = dist.sample(n_samples, conditions=conditions_dict).x1.values

        sigmas.append(np.std(values))
        means.append(np.mean(values))

    return sigmas, means


def comparison_methods(
    x_train,
    y_train,
    x_test,
    y_test,
    inlier_ids,
    df_inlier,
    model_type,
    return_ids=True,
    seed=42,
):
    """
    > This function takes in the training and testing data, the inlier ids, the dataframe of inliers, the model type,
    and a seed. It then fits the specific comparison model and returns the uncertainty scores

    Args:
      x_train: training data
      y_train: the training data
      x_test: the test data
      y_test: the true values of the test set
      inlier_ids: the indices of the inliers
      df_inlier: the dataframe of inliers
      model_type: the type of model to use. Can be one of the following:
      return_ids: If True, returns the ids of the most uncertain and least uncertain samples. If False,
    returns the uncertainty score. Defaults to True
      seed: random seed.

    Returns:
      The uncertainty score
    """

    ids = inlier_ids
    if model_type == "gp":
        gp = gaussian_process.GaussianProcessRegressor(random_state=seed)

        # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        #             noise_level=1, noise_level_bounds=(1e-10, 1e1)
        #         )

        # gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=0.0)

        gp.fit(x_train, y_train)
        y_pred, sigma = gp.predict(x_test[ids, :], return_std=True)
        uncertainty_score = sigma

    if model_type == "qr":
        config = {
            "n_estimators": 10,
            "max_depth": 5,
            "learning_rate": 0.01,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "random_state": seed,
        }

        qr = QuantileRegression(model_type="gbr", config=config)
        qr.fit(x_train, y_train)
        output = qr.predict(x_test[ids, :])
        uncertainty_score = output[2] - output[1]

    if model_type == "bnn":
        from climb.tool.impl.data_suite.third_party.uq360.algorithms.variational_bayesian_neural_networks.bnn import (
            BnnRegression,
        )

        config = {
            "ip_dim": x_train.shape[1],
            "op_dim": 1,
            "num_nodes": 8,
            "num_layers": 5,
            "step_size": 3e-5,
            "num_epochs": 100,
            "random_state": seed,
        }

        bnn = BnnRegression(config=config)
        bnn.fit(
            torch.tensor(x_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float),
        )
        output = bnn.predict(torch.tensor(x_test[ids, :], dtype=torch.float))
        uncertainty_score = output[2] - output[1]

    if model_type == "conformal":
        # uncertainty_score = conformal(x_train, y_train, x_test, y_test, ids)
        from climb.tool.impl.data_suite.models.conformal import conformal_class

        conf = conformal_class(seed=seed)
        conf.fit(x_train, y_train)
        uncertainty_score = conf.predict(
            x_test[ids, :],
            y_test[ids],
            just_conf=True,
        )

    if model_type == "copula":
        uncertainty_score, _ = sample_copula(
            x_train,
            y_train,
            x_test,
            y_test,
            ids,
        )

    if model_type == "mcd":
        from climb.tool.impl.data_suite.models.mcd import mc_dropout

        mcd = mc_dropout(epochs=10, lr=0.01, batch_size=5, device=device)
        mcd.fit(x_train=x_train, y_train=y_train)
        _, uncertainty_score = mcd.predict(
            x_test=x_test[ids, :],
            y_test=y_test[ids],
            mc_samples=10,
        )

    if model_type == "ensemble":
        from climb.tool.impl.data_suite.models.ensemble import ensemble

        ens = ensemble(
            epochs=10,
            lr=0.01,
            batch_size=128,
            device=device,
            n_models=5,
        )
        ens.fit(x_train=x_train, y_train=y_train)
        _, uncertainty_score = ens.predict(
            x_test=x_test[ids, :],
            y_test=y_test[ids],
            mc_samples=2,
        )

    if return_ids:
        from copy import deepcopy

        tmp_df = deepcopy(df_inlier)
        nsamples = 100

        tmp_df["conf_interval"] = uncertainty_score

        df_sorted = tmp_df.sort_values(by=["conf_interval"], ascending=True)

        non_noisy_ids = df_sorted.index.values[0:nsamples]

        df_sorted = tmp_df.sort_values(by=["conf_interval"], ascending=False)

        noisy_ids = df_sorted.index.values[0:nsamples]

        return non_noisy_ids, noisy_ids

    else:
        return uncertainty_score


def conformal(x_train, y_train, x_test, y_test, inlier_ids):
    normalize = True

    feat = int(0)
    logging.info(f"Running analysis for feature = {feat}")

    if not normalize:
        underlying_model = RegressorAdapter(linear_model.LinearRegression())
        nc = RegressorNc(underlying_model, AbsErrorErrFunc())

    else:
        underlying_model = RegressorAdapter(linear_model.LinearRegression())
        normalizing_model = RegressorAdapter(
            KNeighborsRegressor(n_neighbors=1),
        )
        normalizer = RegressorNormalizer(
            underlying_model,
            normalizing_model,
            AbsErrorErrFunc(),
        )
        nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)

    # -----------------------------------------------------------------------------
    # Fit

    x = x_train
    y = y_train

    train_indices = random.sample(range(x.shape[0]), int(x.shape[0] * 0.8))
    calibration_indices = []
    for i in range(x.shape[0]):
        if i not in train_indices:
            calibration_indices.append(i)

    icp = IcpRegressor(nc)
    icp.fit(x[train_indices, :], y[train_indices])
    icp.calibrate(x[calibration_indices, :], y[calibration_indices])

    # -----------------------------------------------------------------------------
    # Predict
    prediction = icp.predict(x_test[inlier_ids, :], significance=0.1)
    header = ["min_val", "max_val", "true_val", "conf_interval"]
    size = prediction[:, 1] - prediction[:, 0]

    table = np.vstack([prediction.T, y_test[inlier_ids], size.T]).T
    df_conf_model = pd.DataFrame(table, columns=header)
    return df_conf_model.min_val, df_conf_model.max_val


def uncertainty_benchmark(
    x_train,
    y_train,
    x_test,
    y_test,
    y_test_ids,
    ids,
    model_type,
    wandb_dict,
    conformal_dict=None,
):
    """
    > This function takes in a model type, trains the model.
    It peforms the full uncertainty benchmarking, hence besides training the model,
    it also computes uncertainty metrics and OOD metrics. That we would log for the synthetic experiment to wandb.

    Args:
      x_train: the training data
      y_train: the training labels
      x_test: the test data
      y_test: the true values of the test set
      y_test_ids: the true labels of the test set
      ids: the indices of the inliers in the test set
      model_type: the type of model to use. Can be one of the following:
      wandb_dict: a dictionary that will be used to store the results of the experiment.
      conformal_dict: a dictionary of dataframes containing the conformal predictions for each feature.

    Returns:
      the dictionary wandb_dict which contains the results of the uncertainty benchmark.
    """

    if model_type == "qr":
        from climb.tool.impl.data_suite.third_party.uq360.algorithms.quantile_regression import QuantileRegression

        config = {
            "n_estimators": 10,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
        }

        qr = QuantileRegression(model_type="gbr", config=config)
        qr.fit(x_train, y_train)
        output = qr.predict(x_test[ids, :])

        preds = output[0]  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = output[1]  # lower bound of the prediction interval
        ub = output[2]  # upper bound of the prediction interval

    if model_type == "bnn":
        import torch

        from climb.tool.impl.data_suite.third_party.uq360.algorithms.variational_bayesian_neural_networks.bnn import (
            BnnRegression,
        )

        config = {
            "ip_dim": x_train.shape[1],
            "op_dim": 1,
            "num_nodes": 8,
            "num_layers": 5,
            "step_size": 3e-2,
            "num_epochs": 100,
        }

        bnn = BnnRegression(config=config)
        bnn.fit(
            torch.tensor(x_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float),
        )
        output = bnn.predict(torch.tensor(x_test[ids, :], dtype=torch.float))

        preds = output[0]  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = output[1]  # lower bound of the prediction interval
        ub = output[2]  # upper bound of the prediction interval

    if model_type == "gp":
        from sklearn import gaussian_process
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        gp = gaussian_process.GaussianProcessRegressor()

        kernel = 1.0 * RBF(
            length_scale=100.0,
            length_scale_bounds=(1e-2, 1e3),
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))

        gp = gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
        )

        gp.fit(x_train, y_train)

        y_pred, sigma = gp.predict(x_test[ids, :], return_std=True)

        preds = y_pred  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = y_pred - sigma  # lower bound of the prediction interval
        ub = y_pred + sigma  # upper bound of the prediction interval

    if model_type == "copula":
        sigmas, means = sample_copula(x_train, y_train, x_test, y_test, ids)
        preds = np.array(means)  # target predictions
        true = y_test[ids]  # ground truth observations
        lb = np.array(means) - np.array(
            sigmas,
        )  # lower bound of the prediction interval
        ub = np.array(means) + np.array(
            sigmas,
        )  # upper bound of the prediction interval

    if model_type == "mcd":
        from climb.tool.impl.data_suite.models.mcd import mc_dropout

        mcd = mc_dropout(epochs=20, lr=0.01, batch_size=5, device=device)
        mcd.fit(x_train=x_train, y_train=y_train)
        y_pred, sigma = mcd.predict(
            x_test=x_test[ids, :],
            y_test=y_test[ids],
            mc_samples=50,
        )
        y_pred, sigma = np.array(y_pred), np.array(sigma)

        preds = y_pred  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = y_pred - (2 * sigma)  # lower bound of the prediction interval
        ub = y_pred + (2 * sigma)  # upper bound of the prediction interval

    if model_type == "ensemble":
        from climb.tool.impl.data_suite.models.ensemble import ensemble

        ens = ensemble(
            epochs=20,
            lr=0.01,
            batch_size=5,
            device=device,
            n_models=5,
        )
        ens.fit(x_train=x_train, y_train=y_train)
        y_pred, sigma = ens.predict(
            x_test=x_test[ids, :],
            y_test=y_test[ids],
            mc_samples=2,
        )

        preds = y_pred  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = y_pred - (2 * sigma)  # lower bound of the prediction interval
        ub = y_pred + (2 * sigma)  # upper bound of the prediction interval

    if model_type == "conformal":
        from climb.tool.impl.data_suite.models.conformal import conformal_class

        conf = conformal_class()
        conf.fit(x_train, y_train)
        df = conf.predict(x_test[ids, :], y_test[ids])

        preds = (df["max"] - df["min"]) / 2  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = df["min"]  # lower bound of the prediction interval
        ub = df["max"]  # upper bound of the prediction interval

    if model_type == "conformal_copula":
        feature = 0
        from copy import deepcopy

        df_conformal = conformal_dict[feature]

        dc = deepcopy(df_conformal)

        dc = dc.iloc[ids, :]

        dc["pred"] = dc["min"] + (dc["conf_interval"] / 2)

        preds = dc["pred"]  # target predictions
        true = y_test[ids]  # dc['true_val']  # ground truth observations
        lb = dc["min"]  # lower bound of the prediction interval
        ub = dc["max"]  # upper bound of the prediction interval

        min_vals, max_vals = conformal(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            inlier_ids=ids,
        )

        preds = (max_vals - min_vals) / 2  # target predictions
        true = y_test[ids]  # test[ids,0]  # ground truth observations
        lb = min_vals  # lower bound of the prediction interval
        ub = max_vals  # upper bound of the prediction interval

    (
        uncert_metrics,
        excess,
        deficet,
        excess_all,
        deficet_all,
    ) = compute_uncertainty_metrics(
        preds=preds,
        lower_bound=lb,
        upper_bound=ub,
        true=true,
    )

    idx_ordered = np.argsort(ub - lb)
    results, roc = test_ood(np.array(y_test_ids)[ids], idx_ordered)

    wandb_dict = process_results(
        wandb_dict,
        results,
        roc,
        uncert_metrics,
        excess,
        deficet,
        excess_all,
        deficet_all,
        name=model_type,
    )

    return wandb_dict
