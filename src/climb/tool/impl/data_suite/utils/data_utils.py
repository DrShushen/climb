import pickle

import matplotlib.pyplot as plt
import numpy as np


def scaler(fg, bg, center=True):
    """
    > This function takes two arrays, one of foreground data and one of background data, and returns two arrays, one
    of foreground data and one of background data, where the foreground data is scaled to the background
    data

    Args:
      fg: foreground data
      bg: background data
      center: If True, the data will be centered before scaling. Defaults to True

    Returns:
      The transformed data.
    """
    if center:
        fg = fg - np.mean(fg, axis=0)
        bg = bg - np.mean(bg, axis=0)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(fg)
    return scaler.transform(fg), scaler.transform(bg)


def return_diagonal(array, just_diag=True):
    """
    > This function takes a 2D array and returns the diagonal of that array

    Args:
      array: the array you want to return the diagonal of
      just_diag: If True, the function will return the diagonal values of the array. If False, it will
    return the entire array. Defaults to True

    Returns:
      The diagonal values of the array.
    """
    diagonal_vals = []
    if just_diag:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if i != j:
                    array[i, j] = 0
                else:
                    diagonal_vals.append(array[i, j])
        return array, diagonal_vals
    else:
        return array, diagonal_vals


def covariance_comparison(clean_array, noisy_array):
    """
    > This function takes in two arrays, one clean and one noisy, and returns a list of the indices of the features
    that have a covariance that is greater than the covariance of the clean data

    Args:
      clean_array: The clean data array
      noisy_array: The array of noisy data

    Returns:
      a list of the indices of the features that have a covariance that is greater than 0.
    """

    clean_array, noisy_array = scaler(clean_array, noisy_array, center=True)

    noisy_array_cov = noisy_array.T.dot(
        noisy_array,
    ) / (noisy_array.shape[0] - 1)
    clean_array_cov = clean_array.T.dot(
        clean_array,
    ) / (clean_array.shape[0] - 1)

    plt.figure(figsize=(10, 10))
    data_matrix, diagonal_vals = return_diagonal(
        noisy_array_cov - clean_array_cov,
    )
    plt.matshow(data_matrix > 0, fignum=1, aspect="auto")
    plt.colorbar()
    plt.show()

    cov_suspects = np.argwhere(np.array(diagonal_vals) > 0)

    try:
        if len(cov_suspects) > 1:
            cov_suspects = list(cov_suspects.squeeze())
        else:
            cov_suspects = [int(cov_suspects.squeeze())]
    except BaseException:
        pass

    return cov_suspects


def get_suspect_features(clean_corpus, test_dataset, alpha=0.05):
    """
    > This function takes in a clean corpus and a test dataset, and returns a list of feature indices that are
    statistically different between the two

    Args:
      clean_corpus: the clean corpus
      test_dataset: the dataset you want to test for contamination
      alpha: the significance level for the KS test.

    Returns:
      The suspicious features are being returned.
    """

    from scipy.stats import ks_2samp

    np.random.seed(123456)
    suspicious_feat = []
    for feat_idx in range(clean_corpus.shape[1]):
        if ks_2samp(clean_corpus[:, feat_idx], test_dataset[:, feat_idx])[1] < alpha:
            suspicious_feat.append(feat_idx)

    return suspicious_feat


def write_to_file(contents, filename):
    """
    > This function takes in a variable and a filename, and writes the variable to the filename as a
    pickle file.

    Args:
      contents: the data to be written to the file
      filename: the name of the file to write to
    """
    # write contents to pickle file

    with open(filename, "wb") as handle:
        pickle.dump(contents, handle)


def read_from_file(filename):
    """
    > This function loads a file from a pickle

    Args:
      filename: the name of the file to read from

    Returns:
      the pickle file.
    """
    # load file from pickle

    return pickle.load(open(filename, "rb"))
