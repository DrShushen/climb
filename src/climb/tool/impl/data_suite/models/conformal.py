import random

import numpy as np
import pandas as pd
import torch
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import (
    AbsErrorErrFunc,
    RegressorNc,
    RegressorNormalizer,
    SignErrorErrFunc,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from climb.tool.impl.data_suite.models import nn_conformal as nnc

# Parameters if using a pytorch NN for the base learner
# desired miscoverage level
alpha = 0.1
# pytorch's optimizer object
nn_learn_func = torch.optim.SGD
# number of epochs
epochs = 1000
# learning rate
lr = 3e-3
# mini-batch size
batch_size = 32
# hidden dimension of the network
hidden_size = 2
# dropout regularization rate
dropout = 0.1
# weight decay regularization
wd = 0
# seed for splitting the data in cross-validation.
cv_test_ratio = 0.1
# ratio of held-out data, used in cross-validation
cv_random_state = 1


# model_dict={

#     "rf": RegressorAdapter(RandomForestRegressor(min_samples_leaf=5, random_state=42)),
#     "tree": RegressorAdapter(DecisionTreeRegressor(min_samples_leaf=5,random_state=42)),
#     "knn": RegressorAdapter(KNeighborsRegressor(n_neighbors=1,random_state=42)),
#     "nn": nnc.MSENet_RegressorAdapter(model=None,
#                                        fit_params=None,
#                                        in_shape = 3,
#                                        hidden_size = hidden_size,
#                                        learn_func = nn_learn_func,
#                                        epochs = epochs,
#                                        batch_size=batch_size,
#                                        dropout=dropout,
#                                        lr=lr,
#                                        wd=wd,
#                                        test_ratio=cv_test_ratio,
#                                        random_state=42)
# }

conformity_dict = {"abs": AbsErrorErrFunc(), "sign": SignErrorErrFunc()}


# This class is a wrapper for the conformal prediction library. It takes in a base learner, a
# normalizer, a conformity score, and a boolean for normalization. It also takes in the input
# dimension and a seed.
class conformal_class:
    def __init__(
        self,
        base_name="rf",
        norm_name="knn",
        conformity_score="abs",
        normalize=True,
        input_dim=2,
        seed=42,
    ):
        input_dim = input_dim
        if not normalize:
            # TODO: define model_dict here
            underlying_model = model_dict[base_name]  # noqa F821
            nc = RegressorNc(
                underlying_model,
                conformity_dict[conformity_score],
            )

        else:
            if base_name == "rf":
                underlying_model = RegressorAdapter(
                    RandomForestRegressor(
                        min_samples_leaf=5,
                        random_state=seed,
                    ),
                )

            if base_name == "mlp":
                underlying_model = RegressorAdapter(
                    MLPRegressor(random_state=seed),
                )

            elif base_name == "tree":
                underlying_model = RegressorAdapter(
                    DecisionTreeRegressor(
                        min_samples_leaf=5,
                        random_state=seed,
                    ),
                )

            elif base_name == "knn":
                underlying_model = RegressorAdapter(
                    KNeighborsRegressor(n_neighbors=1),
                )

            elif base_name == "nn":
                underlying_model = nnc.MSENet_RegressorAdapter(
                    model=None,
                    fit_params=None,
                    in_shape=input_dim,
                    hidden_size=hidden_size,
                    learn_func=nn_learn_func,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    lr=lr,
                    wd=wd,
                    test_ratio=cv_test_ratio,
                    random_state=seed,
                )

            if norm_name == "rf":
                normalizing_model = RegressorAdapter(
                    RandomForestRegressor(
                        min_samples_leaf=5,
                        random_state=seed,
                    ),
                )

            elif norm_name == "tree":
                normalizing_model = RegressorAdapter(
                    DecisionTreeRegressor(
                        min_samples_leaf=5,
                        random_state=seed,
                    ),
                )

            elif norm_name == "knn":
                normalizing_model = RegressorAdapter(
                    KNeighborsRegressor(n_neighbors=1),
                )

            elif norm_name == "nn":
                normalizing_model = nnc.MSENet_RegressorAdapter(
                    model=None,
                    fit_params=None,
                    in_shape=3,
                    hidden_size=hidden_size,
                    learn_func=nn_learn_func,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    lr=lr,
                    wd=wd,
                    test_ratio=cv_test_ratio,
                    random_state=seed,
                )

            normalizer = RegressorNormalizer(
                underlying_model,
                normalizing_model,
                conformity_dict[conformity_score],
            )
            nc = RegressorNc(
                underlying_model,
                conformity_dict[conformity_score],
                normalizer,
            )

        self.seed = seed
        self.icp = IcpRegressor(nc)

    def fit(self, x_train, y_train):
        """
        > This function takes in the training data and splits it into a training set and a calibration set.
        It is then used to fit the conformal predictor

        Args:
          x_train: The training data.
          y_train: The target variable
        """

        x_train, y_train = (
            np.array(x_train).astype(np.float32),
            np.array(
                y_train,
            ).astype(np.float32),
        )
        self.x_train, self.y_train = x_train, y_train
        random.seed(a=self.seed)
        np.random.seed(self.seed)
        self.icp.fit(x_train, y_train)
        train_indices = random.sample(
            range(x_train.shape[0]),
            int(x_train.shape[0] * 0.8),
        )
        calibration_indices = []
        for i in range(x_train.shape[0]):
            if i not in train_indices:
                calibration_indices.append(i)

        self.train_indices = train_indices

        self.icp.fit(x_train[train_indices, :], y_train[train_indices])
        self.icp.calibrate(
            x_train[calibration_indices, :],
            y_train[calibration_indices],
        )

    def predict(self, x_test, y_test, just_conf=False):
        """
        > This function takes in the test data, and returns a dataframe with the confidence intervals, the true
        values, and the normalized confidence intervals.

        Args:
          x_test: the test data
          y_test: the true values of the test set
          just_conf: If True, only return the confidence intervals. Defaults to False

        Returns:
          The prediction of the model.
        """

        x_test = x_test.astype(np.float32)
        prediction = self.icp.predict(x_test, significance=0.1)
        header = ["min", "max", "true_val", "conf_interval"]
        size = prediction[:, 1] - prediction[:, 0]

        table = np.vstack([prediction.T, y_test, size.T]).T
        df = pd.DataFrame(table, columns=header)

        feature_array = self.y_train[self.train_indices]
        feature_range = np.max(feature_array) - np.min(feature_array)
        df["norm_interval"] = df["conf_interval"] / feature_range

        if just_conf:
            return df.conf_interval.values

        return df
