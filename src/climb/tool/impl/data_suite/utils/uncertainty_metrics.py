import numpy as np
from pyod.utils.data import evaluate_print
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from climb.tool.impl.data_suite.third_party.uq360.metrics.regression_metrics import compute_regression_metrics as crm


def compute_uncertainty_metrics(preds, lower_bound, upper_bound, true):
    """
    It computes the uncertainty metrics for a given set of predictions, lower bounds, upper bounds, and
    true values

    Args:
      preds: the predicted values
      lower_bound: the lower bound of the prediction interval
      upper_bound: The upper bound of the prediction interval.
      true: the true values
    """
    uncert_metrics = crm(
        y_true=true,
        y_mean=preds,
        y_lower=lower_bound,
        y_upper=upper_bound,
    )
    excess, excess_all = compute_excess(
        true=true,
        lb=lower_bound,
        ub=upper_bound,
    )
    deficet, deficet_all = compute_deficet(
        true=true,
        lb=lower_bound,
        ub=upper_bound,
    )
    return uncert_metrics, excess, deficet, excess_all, deficet_all


def compute_excess(true, lb, ub):
    """
    > This function computes the average excess of the true values over the lower and upper bounds

    Args:
      true: the true values of the data
      lb: lower bound
      ub: upper bound

    Returns:
      The mean and the proportion of excess
    """
    true, lb, ub = np.array(true), np.array(lb), np.array(ub)
    excess = []
    for i in range(true.shape[0]):
        if true[i] >= lb[i] and true[i] <= ub[i]:
            excess.append(np.min([true[i] - lb[i], ub[i] - true[i]]))

    return np.mean(excess), np.sum(excess) / true.shape[0]


def compute_deficet(true, lb, ub):
    """
    > This function computes the average and the proportion of the time that the true value is outside the confidence
    interval

    Args:
      true: the true values of the parameters
      lb: lower bound
      ub: upper bound

    Returns:
      The mean and the proportion of the deficet
    """

    true, lb, ub = np.array(true), np.array(lb), np.array(ub)
    deficet = []
    for i in range(true.shape[0]):
        if true[i] <= lb[i] or true[i] >= ub[i]:
            deficet.append(
                np.min([np.abs(true[i] - lb[i]), np.abs(true[i] - ub[i])]),
            )

    return np.mean(deficet), np.sum(deficet) / true.shape[0]


def perf_measure(y_actual, y_pred):
    """
    > This function takes two lists of the same length, and returns a tuple of four numbers:
    TN, FP, FN, TP

    Args:
      y_actual: the actual values of the target variable
      y_pred: The predicted values

    Returns:
      True Negative, False Positive, False Negative, True Positive
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return (TN, FP, FN, TP)


def test_ood(y_test_ids, idx_ordered):
    """
    > This functuib takes the ordered list of indices and the true labels, and then iterates through the indices,
    assigning the first x% of the indices to the "certain" class, and the remaining to the "uncertain"
    class.

    It then calculates performance metrics

    Args:
      y_test_ids: the true labels of the test set
      idx_ordered: the indices of the test set, ordered by the distance to the nearest neighbor.

    Returns:
        dictionary of metrics, ROC score
    """
    props = np.linspace(0, 1, 1000)
    results = {}
    recall = []
    precision = []
    roc = 0
    for prop in props:
        lim = int(prop * len(idx_ordered))
        # lim=threshold

        certain = idx_ordered[0:lim]
        uncertain = idx_ordered[lim:]

        y_pred = []

        for i in range(660):
            if i in certain:
                y_pred.append(0)
            if i in uncertain:
                y_pred.append(1)

        TN, FP, FN, TP = perf_measure(np.array(y_test_ids), np.array(y_pred))

        if round(TP / (TP + FN + 0.01), 2) != 0.95:
            continue

        # calculate recall and precision
        recall = recall_score(
            np.array(y_test_ids),
            np.array(y_pred),
            labels=[0, 1],
            average="binary",
        )
        precision = precision_score(
            np.array(y_test_ids),
            np.array(y_pred),
            labels=[0, 1],
            average="binary",
        )

        roc = evaluate_print(
            "KNNs",
            y_pred,
            y_test_ids,
        )
        roc = np.round(roc_auc_score(y_pred, y_test_ids), decimals=4)
        results[prop] = {
            "TPR": TP / (TP + FN + 0.01),
            "FPR": FP / (FP + TN + 0.01),
            "TNR": TN / (TN + FP + 0.01),
            "FNR": FN / (FN + TP + 0.01),
            "Recall": recall,
            "Precision": precision,
            "roc": roc,
        }

    return results, roc


def process_results(
    wandb_dict,
    results,
    roc,
    uncert_metrics,
    excess,
    deficet,
    excess_all,
    deficet_all,
    name,
):
    """
    > This function processes the results and stores it in a dict to log to wandb

    Args:
      wandb_dict: a dictionary that will be used to log the results to wandb
      results: a dictionary of dictionaries, where the keys are the names of the models and the values
                are dictionaries of the results of the model.
      roc: the ROC AUC score
      uncert_metrics: a dictionary of metrics that are calculated for the uncertainty
      excess: excess of interval for specific model
      deficet: deficet of interval for specific model
      excess_all: proportion excess
      deficet_all: proportion deficet
      name: The name of the model.
    """

    wandb_dict[f"excess_{name}"] = excess
    wandb_dict[f"deficet_{name}"] = deficet
    wandb_dict[f"excess_all_{name}"] = excess_all
    wandb_dict[f"deficet_all_{name}"] = deficet_all
    wandb_dict[f"roc_{name}"] = roc

    for key in uncert_metrics.keys():
        wandb_dict[f"{key}_{name}"] = uncert_metrics[key]

    if len(results) == 0:
        return wandb_dict

    tmp_results = list(results.values())[0]
    for key in tmp_results:
        wandb_dict[f"{key}_{name}"] = tmp_results[key]

    return wandb_dict
