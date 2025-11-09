import logging

import numpy as np
import pandas as pd

from climb.tool.impl.data_suite.third_party.copulas.multivariate import GaussianMultivariate

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def fit_sample_copula(
    clean_corpus,
    copula="vine",
    copula_n_samples=10,
    columns=None,
    random_seed=42,
):
    """
    > The function takes a corpus of data, fits a copula to it, and then samples from the copula

    Args:
      clean_corpus: the corpus of data you want to fit the copula to.
      copula: the type of copula to use. Defaults to vine
      copula_n_samples: The number of samples to generate from the copula. Defaults to 10
      columns: The names of the columns in the dataframe.
      random_seed: The random seed. Defaults to 42
    """

    try:
        if copula == "vine":
            from climb.tool.impl.data_suite.third_party.copulas.multivariate import VineCopula

            logging.info("Vine...")
            # vine = VineCopula('center')
            # vine = VineCopula('regular')
            vine = VineCopula("direct", random_seed=random_seed)
            if columns is None:
                columns = [f"x{i + 1}" for i in range(clean_corpus.shape[1] - 1)] + ["y"]

            vine.fit(pd.DataFrame(data=clean_corpus))  # , columns=columns))
            logging.info(f"Copula Samples = {copula_n_samples}")
            samples = vine.sample(copula_n_samples)
    except BaseException:
        dist = GaussianMultivariate(random_seed=random_seed)
        dist.fit(clean_corpus)
        logging.info(f"Copula Samples = {copula_n_samples}")
        samples = dist.sample(copula_n_samples)

    if copula == "gauss":
        logging.info("Gaussian...")
        dist = GaussianMultivariate()
        dist.fit(clean_corpus)
        logging.info(f"Copula Samples = {copula_n_samples}")
        samples = dist.sample(copula_n_samples)

    copula_samples = np.array(samples)
    return copula_samples
