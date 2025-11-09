import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_synthetic_large(num_samples=1000):
    """
    > This function generates a random multivariate normal distribution with
    the given mean and covariance matrix

    Args:
      num_samples: The number of samples to generate. Defaults to 1000

    Returns:
      A tuple of two numpy arrays for train and test
    """
    # The desired mean values of the sample.
    mu = np.array([5.0, 0.0, 10.0, 2.0, 7.0])

    # The desired covariance matrix.
    r = np.array(
        [
            [3.40, -2.75, -2.00, -5.75, -2.00],
            [-2.75, 5.50, 1.50, 2.75, 3.00],
            [-5.00, 1.50, 1.25, 0.75, -3.00],
            [2.00, 1.50, 3.25, 2.75, 1.00],
            [5.00, 3.50, 5.25, -0.75, 2.00],
        ],
    )

    # Generate the random samples.
    data = np.random.multivariate_normal(mu, r, size=num_samples)

    train, test = train_test_split(data, test_size=0.66, random_state=42)

    return train, test


def generate_synthetic_small(num_samples=1000):
    """
    > This function generates a random sample of data from a multivariate normal
    distribution with a specified mean
    and covariance matrix

    Args:
      num_samples: The number of samples to generate. Defaults to 1000

    Returns:
      A tuple of two numpy arrays for train and test
    """

    # The desired mean values of the sample.
    mu = np.array([5.0, 0.0, 10.0])

    # The desired covariance matrix.
    r = np.array(
        [[3.40, -2.75, -2.00], [-2.75, 5.50, 1.50], [-2.00, 1.50, 1.25]],
    )

    # Generate the random samples.
    data = np.random.multivariate_normal(mu, r, size=num_samples)

    train, test = train_test_split(data, test_size=0.66, random_state=42)

    return train, test


def corrupt_data_func(
    data,
    feat_list,
    mean=0,
    variance=1,
    proportion=0.5,
    dist="normal",
):
    """
    > This function takes in a dataframe, a list of features to corrupt, and a distribution to corrupt the
    data with. It then corrupts the data with the specified distribution and returns the corrupted data,
    the original data, a list of the corrupted data points, a list of the noise added to the data, and a
    list of the indices of the corrupted data points.

    Args:
      data: the data you want to corrupt
      feat_list: the list of features to corrupt
      mean: the mean of the distribution you want to sample from. Defaults to 0
      variance: the variance of the noise. Defaults to 1
      proportion: the proportion of data that will be corrupted
      dist: the distribution of the noise. Defaults to normal

    Returns:
      corrupt_data, data, corrupt_ids, noise, noise_id
    """

    data_corrupt = deepcopy(data)

    for feat_idx in feat_list:
        corrupt_ids = []
        corrupt_data = []
        noise = []
        noise_id = []

        for i in range(len(data_corrupt)):
            value = data_corrupt[i, feat_idx]

            if random.random() < proportion:
                if dist == "normal":
                    noisy = np.random.normal(mean, variance)
                if dist == "beta":
                    noisy = np.random.beta(8, 2)
                if dist == "weibull":
                    noisy = np.random.weibull(2)
                if dist == "gamma":
                    noisy = np.random.gamma(1, 2)

                noise.append(noisy)
                corrupt_data.append(value + noisy)
                corrupt_ids.append(1)
                noise_id.append(i)

            else:
                corrupt_ids.append(0)
                noise.append(0)
                corrupt_data.append(value)

    return corrupt_data, data, corrupt_ids, noise, noise_id


def load_synthetic_data(
    n_synthetic=1000,
    mean=0,
    noise_variance=0,
    dim="small",
    prop="0.5",
    dist="normal",
):
    """
    > This function generates a synthetic dataset with a specified number of samples, mean, noise
    variance, dimensionality, proportion of noise, and distribution of noise

    Args:
      n_synthetic: number of samples to generate. Defaults to 1000
      mean: mean of the noise distribution. Defaults to 0
      noise_variance: the variance of the noise distribution. Defaults to 0
      dim: "small" or "large". Defaults to small
      prop: proportion of data to corrupt. Defaults to 0.5
      dist: the distribution of the noise. Can be "normal" or "uniform". Defaults to normal
    """

    if dim == "small":
        train, test_clean = generate_synthetic_small(num_samples=n_synthetic)
    if dim == "large":
        train, test_clean = generate_synthetic_large(num_samples=n_synthetic)

    test_corrupted, orig_test, noise_bool, noise_values, noise_idx = corrupt_data_func(
        data=test_clean,
        feat_list=[0],
        mean=0,
        variance=noise_variance,
        proportion=prop,
        dist=dist,
    )

    y_test_ids = noise_bool
    test = deepcopy(orig_test)
    test[:, 0] = test_corrupted

    return train, test, orig_test, y_test_ids, noise_values, noise_idx


def load_adult_data(split_size=0.3):
    """
    > This function loads the adult dataset, removes all the rows with missing values, and then splits the data into
    a training and test set

    Args:
      split_size: The proportion of the dataset to include in the test split.

    Returns:
      X_train, X_test, y_train, y_test, X, y
    """

    def process_dataset(df):
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values

        Args:
          df: The dataframe to be processed

        Returns:
          a dataframe after the mapping
        """

        data = [df]

        salary_map = {" <=50K": 1, " >50K": 0}
        df["salary"] = df["salary"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)

        df["country"] = df["country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        df.dropna(how="any", inplace=True)

        for dataset in data:
            dataset.loc[
                dataset["country"] != " United-States",
                "country",
            ] = "Non-US"
            dataset.loc[
                dataset["country"] == " United-States",
                "country",
            ] = "US"

        df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            [
                " Divorced",
                " Married-spouse-absent",
                " Never-married",
                " Separated",
                " Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            [" Married-AF-spouse", " Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if x["workclass"] == " Federal-gov" or x["workclass"] == " Local-gov" or x["workclass"] == " State-gov":
                return "govt"
            elif x["workclass"] == " Private":
                return "private"
            elif x["workclass"] == " Self-emp-inc" or x["workclass"] == " Self-emp-not-inc":
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
            ],
            axis=1,
            inplace=True,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    try:
        df = pd.read_csv("data/adult.csv", delimiter=",")
    except BaseException:
        df = pd.read_csv("../data/adult.csv", delimiter=",")

    df = process_dataset(df)

    df_sex_1 = df.query("sex ==1")

    salary_1_idx = df.query("sex == 0 & salary == 1")
    salary_0_idx = df.query("sex == 0 & salary == 0")

    X = df_sex_1.drop(["salary"], axis=1)
    y = df_sex_1["salary"]

    # Creation of Train and Test dataset
    random.seed(a=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_size,
        random_state=42,
    )

    sample_ids = random.sample(range(len(X_train)), X_train.shape[0])
    X_train = X_train.iloc[sample_ids, :]
    y_train = y_train.iloc[sample_ids]

    X_train = np.vstack([X_train, salary_0_idx.drop(["salary"], axis=1)])
    X_test = np.vstack([X_test, salary_1_idx.drop(["salary"], axis=1)])

    y_train = np.hstack([y_train, salary_0_idx["salary"]])
    y_test = np.hstack([y_test, salary_1_idx["salary"]])

    return X_train, X_test, y_train, y_test, X, y


def load_electric(path="electricity.arff"):
    """
    > This function loads the electric dataset from the file, encodes the class labels, and returns the training and test sets

    Args:
      path: the path to the dataset. Defaults to elecNormNew.arff

    Returns:
      X_train, X_test, y_train, y_test
    """
    from scipy.io.arff import loadarff
    from sklearn.preprocessing import OrdinalEncoder

    try:
        raw_data = loadarff(f"data/{path}")
    except BaseException:
        raw_data = loadarff(f"../data/{path}")
    df = pd.DataFrame(raw_data[0])

    ord_enc = OrdinalEncoder()
    df["class_encoded"] = ord_enc.fit_transform(df[["class"]])

    X_train = df.iloc[0:10000:, 2:8]
    y_train = df.iloc[0:10000:, -1]

    X_val = df.iloc[10000:15000:, 2:8]
    y_val = df.iloc[10000:15000:, -1]

    X_test_real = df.iloc[30000:40000:, 2:8]
    y_test_real = df.iloc[30000:40000:, -1]

    X_test = pd.concat([X_val, X_test_real])
    y_test = pd.concat([y_val, y_test_real])
    return X_train, X_test, y_train, y_test
