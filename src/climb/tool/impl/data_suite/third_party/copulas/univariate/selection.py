import numpy as np
from climb.tool.impl.data_suite.third_party.copulas import get_instance
from scipy.stats import kstest


def select_univariate(X, candidates):
    """Select the best univariate class for this data.

    Args:
        X (pandas.DataFrame):
            Data for which be best univariate must be found.
        candidates (list[Univariate]):
            List of Univariate subclasses (or instances of those) to choose from.

    Returns:
        Univariate:
            Instance of the selected candidate.
    """
    best_ks = np.inf
    best_model = None
    for model in candidates:
        try:
            instance = get_instance(model)
            instance.fit(X)
            ks, _ = kstest(X, instance.cdf)
            if ks < best_ks:
                best_ks = ks
                best_model = model
        except Exception:
            # Distribution not supported
            pass

    return get_instance(best_model)
