import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
from autoprognosis.hooks import Hooks
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.studies.regression import RegressionStudy
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file
from autoprognosis.utils.tester import evaluate_estimator, evaluate_regression, evaluate_survival_estimator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ..tool_comms import BACKUP_OUTPUT_FILE, ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

APModelMode = Literal["linear", "all"]
APTask = Literal["classification", "regression", "survival"]


class BasicProgressReport(Hooks):
    def __init__(self, wd: str, task: APTask) -> None:
        self.called_times = 0
        self.wd = wd
        self.task = task

    def cancel(self) -> bool:
        # print("Study cancelled!")
        return False

    def _print_both(self, *args: Any) -> None:
        # Print to STDOUT as normal, and also print to the backup file (needed for reporting to the UI.)
        # It's not possible to capture the output of Autoprognosis STDOUT/STDERR streams as it uses multiprocessing.
        # We use the backup file to capture the output.
        print(*args)
        with open(os.path.join(self.wd, BACKUP_OUTPUT_FILE), "a") as f:  # pylint: disable=unspecified-encoding
            print(*args, file=f)

    def heartbeat(self, topic: str, subtopic: str, event_type: str, **kwargs: Any) -> None:
        investigating_model = kwargs["name"]
        self._print_both(f"AutoPrognosis 2.0: Duration of this iteration: {kwargs['duration']:.1f}s")
        self._print_both("AutoPrognosis 2.0: Investigating model:", investigating_model)

        # TODO: Log more metrics per step!
        metrics = {
            "classification": [
                {
                    "name": "aucroc",
                    "print_name": "AUCROC",
                },
            ],
            "regression": [
                {
                    "name": "r2",
                    "print_name": "R2",
                },
            ],
            "survival": [
                {
                    "name": "c_index",
                    "print_name": "C-index",
                },
                {
                    "name": "brier_score",
                    "print_name": "Brier score",
                },
            ],
        }

        for metric in metrics[self.task]:
            try:
                kwargs[metric["name"]]
            except KeyError:
                # Fallback:
                metric["name"] = metric["name"].replace("_", "")  # E.g. c_index -> cindex

            if isinstance(kwargs[metric["name"]], str):
                metric_val = kwargs[metric["name"]]
            else:
                metric_val = f"{kwargs[metric['name']]:.3f}"
            self._print_both(f"AutoPrognosis 2.0: {metric['print_name']} achieved: {metric_val}")

        self.called_times += 1

    def finish(self) -> None:
        self._print_both("AutoPrognosis 2.0: Study finished.")


def autoprognosis_classification(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    mode: APModelMode,
    workspace: str,
) -> None:
    SCORE_THRESHOLD = 0.2

    df = pd.read_csv(data_file_path)

    # NOTE: May wish to modify these for purposes of demos etc.
    classifiers = [
        "logistic_regression",
        "lda",
        # "linear_svm",
        # "ridge_classifier",
    ]
    num_iter = 1
    if mode == "all":
        classifiers += [
            "catboost",
            "neural_nets",
            # "decision_trees",
            # "gradient_boosting",
            # "random_forest",
            # "xgboost",
            # "gaussian_process",
            # "tabnet",
            # "knn",
        ]
        num_iter = 3  # 20
    num_study_iter = 1
    num_ensemble_iter = 1  # 3
    timeout = 5  # 60
    # --- --- ---

    tc.print("Setting up the classification study...")
    tc.print("Trying classifiers:")
    tc.print(json.dumps(classifiers, indent=2))
    study_name = f"my_study_{mode}"
    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,
        target=target_variable,
        num_iter=num_iter,
        # ^ Number of automl iterations (for each classifier pipeline). Increase this number to get better results.
        num_study_iter=num_study_iter,
        num_ensemble_iter=num_ensemble_iter,
        timeout=timeout,
        feature_selection=["nop"],  # NOTE.
        # imputers=["mean"],
        classifiers=classifiers,
        # ^ Consider these classifiers. Leave this parameter out to consider all classifiers.
        workspace=Path(workspace),
        hooks=BasicProgressReport(workspace, task="classification"),
        score_threshold=SCORE_THRESHOLD,
    )

    tc.print("Running the classification study, this may take several minutes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = study.fit()
        except AttributeError as e:
            if "no attribute 'fit'" in str(e):
                # TODO: Find a way to print more detail in this case.
                tc.set_returns(
                    f"Classification study failed to achieve minimum performance (below threshold {SCORE_THRESHOLD}). "
                    "No model saved."
                )
                return
    tc.print("Classification study completed!")

    tc.print(f"Final model summary:\n\n{out}\n\n")

    model_path = Path(workspace) / f"model_{mode}.p"
    save_model_to_file(model_path, out)
    # shutil.copy(str(model_path), str(new_model_path))
    tc.print(f"Model saved to: `{model_path}`")

    tc.print("Evaluating the model...")
    model = load_model_from_file(model_path)
    n_folds = 3
    ev = evaluate_estimator(
        [model] * n_folds,
        df[[c for c in df.columns if c != target_variable]],
        df[target_variable],
        n_folds=n_folds,
        pretrained=True,
    )
    ev_readable = json.dumps(ev["str"], indent=2)
    # tc.print(f"Evaluation results:\n\n{ev_readable}")

    # Make predictions.
    predictions = model.predict(df[[c for c in df.columns if c != target_variable]])
    df_with_pred = pd.concat([df, pd.DataFrame(predictions.to_numpy(), columns=["predictions"])], axis=1)
    pred_path = Path(workspace) / f"model_{mode}__predictions.csv"
    df_with_pred.to_csv(pred_path, index=False)
    pred_saved_msg = (
        f"Dataset with predictions was saved to: `{pred_path}`. The prediction column is named `predictions`."
    )
    tc.print(pred_saved_msg)
    tool_return = f"{pred_saved_msg}\n\n{ev_readable}"

    # Compute number of target classes:
    targets = df[target_variable]

    cm = confusion_matrix(targets, predictions)
    tc.print(f"Confusion matrix:\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    cm_fig = plt.gcf()

    # Save the confusion matrix plot.
    cm_plot_path = Path(workspace) / f"model_{mode}__confusion_matrix_plot.png"
    cm_fig.savefig(str(cm_plot_path))

    user_report: List[Any] = [
        """
#### Explanation of classification performance metrics:
- **AUC-ROC**: Measures how well the model separates classes (like "has a disease", vs "doesn't have a disease"). It ranges from `0` to `1`, with `1` being perfect separation and `0.5` meaning random guessing. A higher value (closer to `1`) is better.
- **AUC-PRC**: Focuses more on how well the model identifies the positive class (like "has a disease"). It also ranges from `0` to `1`, where `1` is ideal and `0.5` is random guessing. Higher values are better.
- **Accuracy**: The percentage of total correct predictions. It ranges from `0` to `1`, with `1` meaning perfect predictions. Higher accuracy is better, but it can be misleading in imbalanced datasets.
- **F1 Score (Micro, Macro, Weighted)**: Balances correctness (precision) and completeness (recall) in predictions. Ranges from `0` to `1`, with `1` being the best.
- **Kappa**: Compares model predictions to actual outcomes, accounting for chance. It ranges from `-1` to `1`, where `1` is perfect, `0` is random, and negative values indicate worse-than-random predictions. Higher values are better.
- **Kappa (Quadratic)**: Like Kappa, but it penalizes larger errors more heavily. Values closer to `1` are better.
- **Precision (Micro, Macro, Weighted)**: Measures how many of the predicted positives (like flagged spam emails) were actually correct. It ranges from `0` to `1`, with higher values being better.
- **Recall (Micro, Macro, Weighted)**: Measures how many of the actual positives (like real spam emails) were correctly identified. It ranges from `0` to `1`, with `1` being the best.
- **MCC (Matthews Correlation Coefficient)**: A comprehensive score that accounts for all aspects of prediction quality (correct and incorrect). Ranges from `-1` to `1`, with `1` being perfect, `0` being random guessing, and negative values indicating poor performance. Higher values are better.

##### Note:
**Micro vs Macro vs Weighted**: These are different ways to calculate metrics when there are multiple classes.
- **Micro**: Treats all instances equally (good if class sizes are similar).
- **Macro**: Treats all classes equally, regardless of size (useful if class sizes are different).
- **Weighted**: Adjusts for class size (better for imbalanced datasets).
""",
        "#### Confusion Matrix:",
        cm_fig,
        """
#### Explanation of the confusion matrix:
A confusion matrix, shown as a heatmap, helps visualize how well a model's predictions match the actual results.
* The diagonal values represent correct predictions for each class—the higher these values, the better the model is
performing for that class.
* Values off the diagonal indicate mistakes, where the model confused one class for another.

Ideally, most values should be on the diagonal, meaning accurate predictions, with few off-diagonal values, meaning
fewer errors.

In our heatmap, we use blue darker blue shades to represent higher values, and lighter shades for lower values.
""",
    ]

    tc.set_returns(
        tool_return=tool_return,
        user_report=user_report,
    )


class AutoprognosisClassification(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target_variable = kwargs["target_variable"]
        mode = kwargs.get("mode", "all")
        thrd, out_stream = execute_tool(
            autoprognosis_classification,
            wd=self.working_directory,
            data_file_path=real_path,
            target_variable=target_variable,
            mode=mode,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_classification"

    @property
    def description(self) -> str:
        return """
        Perform a **classification** study on the user's data using the library AutoPrognosis 2.0.

        If `mode` is set to `linear`, only linear classifiers will be considered. If `mode` is set to `all`, \
        all classifiers will be considered.

        This tool will automatically:
        - Preprocess the data (e.g., impute missing values, encode categorical variables, etc.).
        - Handle cross-validation and stratified sampling.
        - Perform auto-ML (hyperparameter selection and pipeline selection) to find the best classification model \
        ensemble for the data.
        - Save the best model as `model_{all,linear}.p` file in the working directory.
        - Return a text summary of the study (metrics like accuracy, precision, recall, F1-score, etc.).
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "target_variable": {"type": "string", "description": "Name of the target variable."},
                        "mode": {
                            "type": "string",
                            "description": "Mode to use for the classification study.",
                            "enum": ["linear", "all"],
                        },
                    },
                    "required": ["data_file_path", "target_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "uses the **AutoPrognosis `2.0`** library to automatically run a classification study "
            "on your data and returns and evaluates the best model"
        )

    @property
    def logs_useful(self) -> bool:
        return True


def autoprognosis_regression(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    mode: APModelMode,
    workspace: str,
) -> None:
    SCORE_THRESHOLD = 0.2

    df = pd.read_csv(data_file_path)

    # NOTE: May wish to modify these for purposes of demos etc.
    regressors = [
        "linear_regression",
        "bayesian_ridge",
    ]
    num_iter = 1
    if mode == "all":
        regressors += [
            "random_forest_regressor",
            "mlp_regressor",
            # "xgboost_regressor",
            # ...,
        ]
        num_iter = 20
    num_study_iter = 1
    num_ensemble_iter = 3
    timeout = 60
    # --- --- ---

    tc.print("Setting up the regression study...")
    tc.print("Trying regressors:")
    tc.print(json.dumps(regressors, indent=2))
    study_name = f"my_study_{mode}"
    study = RegressionStudy(
        study_name=study_name,
        dataset=df,
        target=target_variable,
        num_iter=num_iter,
        # ^ Number of automl iterations (for each regressor pipeline). Increase this number to get better results.
        num_study_iter=num_study_iter,
        num_ensemble_iter=num_ensemble_iter,
        timeout=timeout,
        feature_selection=["nop"],  # NOTE.
        # imputers=["mean"],
        regressors=regressors,
        # ^ Consider these regressors. Leave this parameter out to consider all regressors.
        workspace=Path(workspace),
        hooks=BasicProgressReport(workspace, task="regression"),
        score_threshold=SCORE_THRESHOLD,
    )

    tc.print("Running the regression study, this may take several minutes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = study.fit()
        except AttributeError as e:
            if "no attribute 'fit'" in str(e):
                # TODO: Find a way to print more detail in this case.
                tc.set_returns(
                    f"Regression study failed to achieve minimum performance (below threshold {SCORE_THRESHOLD}). "
                    "No model saved."
                )
                return
    tc.print("Regression study completed!")

    tc.print(f"Final model summary:\n\n{out}\n\n")

    model_path = Path(workspace) / f"model_{mode}.p"
    save_model_to_file(model_path, out)
    # shutil.copy(str(model_path), str(new_model_path))
    tc.print(f"Model saved to: `{model_path}`")

    tc.print("Evaluating the model...")
    model = load_model_from_file(model_path)
    n_folds = 3
    ev = evaluate_regression(
        [model] * n_folds,
        df[[c for c in df.columns if c != target_variable]],
        df[target_variable],
        n_folds=n_folds,
        pretrained=True,
    )
    ev_readable = json.dumps(ev["str"], indent=2)
    tc.print(f"Evaluation results:\n\n{ev_readable}")

    # Make predictions.
    predictions = model.predict(df[[c for c in df.columns if c != target_variable]])
    df_with_pred = pd.concat([df, pd.DataFrame(predictions.to_numpy(), columns=["predictions"])], axis=1)
    pred_path = Path(workspace) / f"model_{mode}__predictions.csv"
    df_with_pred.to_csv(pred_path, index=False)
    pred_saved_msg = (
        f"Dataset with predictions was saved to: `{pred_path}`. The prediction column is named `predictions`."
    )
    tc.print(pred_saved_msg)
    tool_return = f"{pred_saved_msg}\n\n{ev_readable}"

    tc.set_returns(
        tool_return=tool_return,
        user_report=[
            """
#### Explanation of regression performance metrics:
- **Mean Squared Error (MSE)** is a way to measure how well a model predicts outcomes in data science and statistics. \
It calculates the average of the squares of the errors. The "error" here is the difference between what the model \
predicts and the actual values. Imagine you're trying to hit a target with darts. MSE would measure how far each \
dart landed from the target and then average the squared distances. A smaller MSE means your darts are consistently \
closer to the target, indicating a better prediction model.
- **Mean Absolute Error (MAE)** is another method to measure prediction accuracy, similar to MSE. Instead of \
squaring the errors, MAE simply takes the absolute values of these differences. Using the dart analogy again, \
MAE measures the average distance each dart lands from the target, regardless of the direction. \
A lower MAE indicates that the predictions are closer to the actual outcomes on average.
- **R-squared**, also known as the coefficient of determination, is a statistical measure that represents \
the proportion of the variance in the dependent variable that is predictable from the independent variables. \
In simpler terms, it tells you how much of the change in your target variable (like sales or temperature) can be \
explained by the changes in your predictor variables (like advertising spend or time of day). An R-squared value \
closer to 1.0 indicates that the model explains a large portion of the variance, while a value closer to 0 \
indicates less explanation power.
"""
        ],
    )


class AutoprognosisRegression(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target_variable = kwargs["target_variable"]
        mode = kwargs.get("mode", "all")
        thrd, out_stream = execute_tool(
            autoprognosis_regression,  # TODO,
            wd=self.working_directory,
            data_file_path=real_path,
            target_variable=target_variable,
            mode=mode,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_regression"

    @property
    def description(self) -> str:
        return """
        Perform a **regression** study on the user's data using the library AutoPrognosis 2.0.
        
        If `mode` is set to `linear`, only linear regressors will be considered. If `mode` is set to `all`, \
        all regressors will be considered.

        This tool will automatically:
        - Preprocess the data (e.g., impute missing values, encode categorical variables, etc.).
        - Handle cross-validation and stratified sampling.
        - Perform auto-ML (hyperparameter selection and pipeline selection) to find the best regression model \
        ensemble for the data.
        - Save the best model as `model_{all,linear}.p` file in the working directory.
        - Return a text summary of the study (metrics like MSE, R^2, etc.).
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "target_variable": {"type": "string", "description": "Name of the target variable."},
                        "mode": {
                            "type": "string",
                            "description": "Mode to use for the regression study.",
                            "enum": ["linear", "all"],
                        },
                    },
                    "required": ["data_file_path", "target_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "uses the **AutoPrognosis `2.0`** library to automatically run a regression study "
            "on your data and returns and evaluates the best model"
        )

    @property
    def logs_useful(self) -> bool:
        return True


# TODO: Test this - as yet untested.
def autoprognosis_survival(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    time_variable: str,
    mode: APModelMode,
    workspace: str,
) -> None:
    SCORE_THRESHOLD = 0.2

    df = pd.read_csv(data_file_path)
    X = df.drop([time_variable, target_variable], axis=1)
    Y = df[target_variable]
    T = df[time_variable]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    # NOTE: May wish to modify these for purposes of demos etc.
    risk_estimators = [
        "cox_ph",
    ]
    num_iter = 1
    if mode == "all":
        risk_estimators += [
            "deephit",
            # "survival_xgboost",
            # ...,
        ]
        num_iter = 3  # 20
    num_study_iter = 1
    num_ensemble_iter = 1  # 3
    timeout = 5  # 60
    # --- --- ---

    tc.print("Setting up the survival analysis study...")
    tc.print("Will evaluate over the following time horizons:")
    tc.print(eval_time_horizons)
    tc.print("Trying models:")
    tc.print(json.dumps(risk_estimators, indent=2))
    study_name = f"my_study_{mode}"
    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=df,
        target=target_variable,
        time_to_event=time_variable,
        time_horizons=eval_time_horizons,
        num_iter=num_iter,
        # ^ Number of automl iterations (for each models pipeline). Increase this number to get better results.
        num_study_iter=num_study_iter,
        num_ensemble_iter=num_ensemble_iter,
        timeout=timeout,
        feature_selection=["nop"],  # NOTE.
        # imputers=["mean"],
        risk_estimators=risk_estimators,
        # ^ Consider these models. Leave this parameter out to consider all models.
        score_threshold=SCORE_THRESHOLD,
        # ^ NOTE.
        workspace=Path(workspace),
        hooks=BasicProgressReport(workspace, task="survival"),
    )

    tc.print("Running the survival analysis study, this may take several minutes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = study.fit()
        except AttributeError as e:
            if "no attribute 'fit'" in str(e):
                # TODO: Find a way to print more detail in this case.
                tc.set_returns(
                    "Survival analysis study failed to achieve minimum performance "
                    f"(below threshold {SCORE_THRESHOLD}). No model saved."
                    f"\n\n{e}."
                    f"{study}."
                    f"{type(study)}."
                )
                return
    tc.print("Survival analysis study completed!")

    tc.print(f"Final model summary:\n\n{out}\n\n")

    model_path = Path(workspace) / f"model_{mode}.p"
    save_model_to_file(model_path, out)
    # shutil.copy(str(model_path), str(new_model_path))
    tc.print(f"Model saved to: `{model_path}`")

    tc.print("Evaluating the model...")
    model = load_model_from_file(model_path)
    n_folds = 3
    ev = evaluate_survival_estimator(
        [model] * n_folds,
        X,
        T,
        Y,
        eval_time_horizons,  # type: ignore
        n_folds=n_folds,
        pretrained=True,
    )
    extra_explanation_of_metrics = """
Note:
- All metrics are the mean of metrics at each evaluation time horizon specified.
- "predicted_cases", "aucroc", "sensitivity", "specificity", "PPV", "NPV" are classification metrics adapted for \
survival analysis. The target (Y) used to compute these is `predicted_risk_sore > 0.5`. "predicted_cases" is simply the \
proportion of `predicted_risk_sore > 0.5`

"""
    ev_readable = json.dumps(ev["str"], indent=2)
    ev_readable = extra_explanation_of_metrics + ev_readable
    tc.print(f"Evaluation results:\n\n{ev_readable}")

    # Make predictions.
    predictions = model.predict(df[[c for c in df.columns if c != target_variable]], eval_time_horizons)
    predictions = pd.DataFrame(predictions)
    prediction_columns = [f"risk score at {x}" for x in eval_time_horizons]
    predictions.columns = prediction_columns
    df_with_pred = pd.concat([df, predictions], axis=1)
    pred_path = Path(workspace) / f"model_{mode}__predictions.csv"
    df_with_pred.to_csv(pred_path, index=False)
    pred_saved_msg = f"Dataset with predictions was saved to: `{pred_path}`. The prediction columns at each time horizon are: {prediction_columns}."
    tc.print(pred_saved_msg)
    tool_return = f"{pred_saved_msg}\n\n{ev_readable}"

    tc.set_returns(
        tool_return=tool_return,
        user_report=[
            """
#### Explanation of survival analysis performance metrics:

### 1. **Survival Analysis Metrics**  
- **C-index**: Measures how well the model predicts the order in which events (like survival times) happen. It ranges from `0.5` to `1`, where `1` means perfect predictions and `0.5` is random guessing. Higher values are better.
- **Brier Score**: Measures the accuracy of predicted survival probabilities. It ranges from `0` to `1`, where `0` means perfect accuracy, and higher values mean less accurate predictions. Lower scores are better.

### 2. **Classification Metrics Adapted for Survival Analysis**  
These metrics are computed by converting the predicted risk score into a binary prediction (risk score > 0.5 = positive case).

- **Predicted Cases**: The proportion of predictions where the model considers the risk to be high (risk score > 0.5).
- **AUC-ROC**: Measures how well the model separates high-risk from low-risk cases. It ranges from `0.5` to `1`, with `1` being perfect and `0.5` meaning random guessing. Higher values are better.
- **Sensitivity**: The proportion of true high-risk cases correctly identified by the model. It ranges from `0` to `1`, with `1` being perfect. Higher sensitivity means the model catches more high-risk cases.
- **Specificity**: The proportion of true low-risk cases correctly identified. A value of `1` means all low-risk cases were identified perfectly. Higher specificity is better.
- **PPV (Positive Predictive Value)**: The proportion of predicted high-risk cases that are actually high-risk. It ranges from `0` to `1`, with higher values meaning better precision in identifying high-risk cases.
- **NPV (Negative Predictive Value)**: The proportion of predicted low-risk cases that are actually low-risk. It ranges from `0` to `1`, with higher values meaning the model is better at identifying true low-risk cases.

**Note:**
If more than one evaluation time horizon is used, the metrics are the mean of the metrics at each time horizon.
"""
        ],
    )


# TODO: This
class AutoprognosisSurvival(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target_variable = kwargs["target_variable"]
        time_variable = kwargs["time_variable"]
        mode = kwargs.get("mode", "all")
        thrd, out_stream = execute_tool(
            autoprognosis_survival,
            wd=self.working_directory,
            data_file_path=real_path,
            target_variable=target_variable,
            time_variable=time_variable,
            mode=mode,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_survival"

    @property
    def description(self) -> str:
        return """
        Perform a **survival analysis** study on the user's data using the library AutoPrognosis 2.0.
        
        If `mode` is set to `linear`, only linear models will be considered. If `mode` is set to `all`, \
        all models will be considered.

        BOTH target_variable and time_variable MUST be agreed upon and provided!

        This tool will automatically:
        - Preprocess the data (e.g., impute missing values, encode categorical variables, etc.).
        - Handle cross-validation and stratified sampling.
        - Perform auto-ML (hyperparameter selection and pipeline selection) to find the best model \
        ensemble for the data.
        - Save the best model as `model_{all,linear}.p` file in the working directory.
        - Return a text summary of the study (metrics like MSE, R^2, etc.).
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "target_variable": {"type": "string", "description": "Name of the target (event) variable."},
                        "time_variable": {"type": "string", "description": "Name of the time variable."},
                        "mode": {
                            "type": "string",
                            "description": "Mode to use for the survival analysis study.",
                            "enum": ["linear", "all"],
                        },
                    },
                    "required": ["data_file_path", "target_variable", "time_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "uses the **AutoPrognosis `2.0`** library to automatically run a survival analysis study "
            "on your data and returns and evaluates the best model"
        )

    @property
    def logs_useful(self) -> bool:
        return True


def autoprognosis_subgroup_evaluation(
    tc: ToolCommunicator,
    task: APTask,
    data_file_paths: List[str],
    target_variable: str,
    model_path: str,
    workspace: str,
    time_variable: Optional[str] = None,
) -> None:
    tc.print(f"Subgroup evaluation started. Task type: {task}.")

    if task == "survival" and time_variable is None:
        raise ValueError('`time_variable` must be specified when `task` is "survival".')

    try:
        model = load_model_from_file(model_path)
    except Exception as e:
        raise TypeError(
            "Model file is not a valid AutoPrognosis 2.0 file. This tool only supports AutoPrognosis 2.0 models."
        ) from e

    final_results = dict()

    for data_file_path in data_file_paths:
        tc.print(f"Subgroup data file path: {data_file_path}.")
        df = pd.read_csv(data_file_path)

        if task == "classification":
            n_folds = 3
            ev = evaluate_estimator(
                [model] * n_folds,
                df[[c for c in df.columns if c != target_variable]],
                df[target_variable],
                n_folds=n_folds,
                pretrained=True,
            )
            final_results[data_file_path] = ev["str"]

        elif task == "regression":
            n_folds = 3
            ev = evaluate_regression(
                [model] * n_folds,
                df[[c for c in df.columns if c != target_variable]],
                df[target_variable],
                n_folds=n_folds,
                pretrained=True,
            )
            final_results[data_file_path] = ev["str"]

        else:
            X = df.drop([time_variable, target_variable], axis=1)
            Y = df[target_variable]
            T = df[time_variable]
            # NOTE: Using separate horizons per group is questionable, but keep it like this for simplicity.
            eval_time_horizons = [
                int(T[Y.iloc[:] == 1].quantile(0.25)),
                int(T[Y.iloc[:] == 1].quantile(0.50)),
                int(T[Y.iloc[:] == 1].quantile(0.75)),
            ]

            n_folds = 3
            ev = evaluate_survival_estimator(
                [model] * n_folds,
                X,
                T,
                Y,
                eval_time_horizons,  # type: ignore
                n_folds=n_folds,
                pretrained=True,
            )
            final_results[data_file_path] = ev["str"]

    # Save results as a JSON file.
    results_path = os.path.join(workspace, "subgroup_evaluation_results.json")
    with open(results_path, "w") as f:  # pylint: disable=unspecified-encoding
        json.dump(final_results, f, indent=2)

    tc.set_returns(json.dumps(final_results, indent=2))


class AutoprognosisSubgroupEvaluation(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_paths = [os.path.join(self.working_directory, p) for p in kwargs["data_file_paths"]]
        task = kwargs["task"]
        target_variable = kwargs["target_variable"]
        time_variable = kwargs.get("time_variable", None)
        model_path = os.path.join(self.working_directory, kwargs["model_path"])
        thrd, out_stream = execute_tool(
            autoprognosis_subgroup_evaluation,
            wd=self.working_directory,
            task=task,
            data_file_paths=real_paths,
            target_variable=target_variable,
            model_path=model_path,
            workspace=self.working_directory,
            time_variable=time_variable,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_subgroup_evaluation"

    @property
    def description(self) -> str:
        return """
        Perform a subgroup evaluation of the user's data using the library AutoPrognosis 2.0.
        
        This tool will automatically:
        - Load the model from the given path.
        - Evaluate the model on each of the given data files.
        - Return a text summary (JSON format) of the evaluation results.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Type of task to evaluate.",
                            "enum": ["classification", "regression", "survival"],
                        },
                        "data_file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of paths to the data files.",
                        },
                        "target_variable": {
                            "type": "string",
                            "description": (
                                "Name of the target variable. If task is survival, this is the **event** variable."
                            ),
                        },
                        "time_variable": {
                            "type": "string",
                            "description": "Name of the time variable. Required **only** if task is survival.",
                        },
                        "model_path": {"type": "string", "description": "Path to the model file."},
                    },
                    "required": ["task", "data_file_paths", "target_variable", "model_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "uses the **AutoPrognosis `2.0`** library to automatically run a subgroup evaluation "
            "of your data and returns the evaluation results"
        )

    @property
    def logs_useful(self) -> bool:
        return True
