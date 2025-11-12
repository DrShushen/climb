# 🛠️ Tools Reference

### `AutoprognosisClassification`
Uses the **AutoPrognosis `2.0`** library to automatically run a classification study on your data and returns and evaluates the best model

### `AutoprognosisClassificationTrainTest`
Uses the **AutoPrognosis `2.0`** library to automatically run a classification study on your data and returns and evaluates the best model

### `AutoprognosisExplainerInvase`
Uses the INVASE algorithm to generate feature importance explanations for your AutoPrognosis model. For risk estimation, the evaluation time horizons (25th, 50th, and 75th percentiles) are computed automatically from the provided time variable.

### `AutoprognosisExplainerSymbolicPursuit`
Uses the Symbolic Pursuit algorithm to generate an interpretable, symbolic explanation for your AutoPrognosis model. For risk estimation tasks, the evaluation time horizons (25th, 50th, and 75th percentiles) are computed automatically from the provided time variable.

### `AutoprognosisRegression`
Uses the **AutoPrognosis `2.0`** library to automatically run a regression study on your data and returns and evaluates the best model

### `AutoprognosisRegressionTrainTest`
Uses the **AutoPrognosis `2.0`** library to automatically run a regression study on your data and returns and evaluates the best model

### `AutoprognosisSubgroupEvaluation`
Uses the **AutoPrognosis `2.0`** library to automatically run a subgroup evaluation of your data and returns the evaluation results

### `AutoprognosisSurvival`
Uses the **AutoPrognosis `2.0`** library to automatically run a survival analysis study on your data and returns and evaluates the best model

### `AutoprognosisSurvivalTrainTest`
Uses the **AutoPrognosis `2.0`** library to automatically run a survival analysis study on your data and returns and evaluates the best model

### `BalanceData`
Uses the `balance_data` tool to rebalance target distribution of the data.

### `BorutaFeatureSelection`
Uses the **automated feature selection** library to automatically find the most important features in your data.

### `CleanlabOutlierDetection`
Identifies and removes outliers from a dataset

### `ConformalPrediction`
Applies conformal prediction on a pre-trained model and returns prediction intervals or sets.

### `DataIQInsights`
Provides insights for your classification task - which samples were 'easy', 'hard' or 'ambiguous' for classification.

### `DataSuiteInsights`

Uses the data_suite_insights tool to gain insights regions of the dataset that the model may perform poorly on.
The tool provides exemplar records that the user may want to collect more records similar to in order to improve the model's performance.


### `DescriptiveStatistics`
Produce medical paper -style descriptive statistics table for the dataset.

### `ExploratoryDataAnalysis`
Performs exploratory data analysis on your data, providing a summary of its characteristics.

### `FeatureExtractionFromText`
Uses an LLM to extract the features from free text fields.

### `HardwareInfo`
Gather information about your hardware (CPU, RAM, and GPU).

### `HyperImputeImputation`
Uses the **HyperImpute** library to automatically impute missing values in your data.

### `HyperImputeImputationTrainTest`
Uses the **HyperImpute** library to automatically impute missing values in your data.

### `KNNShapleyValuation`
This is a data valuation tool. It uses the KNN algorithm to compute Shapley values for each feature in the dataset. The Shapley value of a feature is a measure of its importance in predicting the target variable. The tool returns a list of features with positive Shapley values, which are considered good predictors, and a list of features with negative Shapley values, which are considered bad predictors.You may want to exclude the bad predictors from your model to improve its performance.

### `PermutationExplainer`
Performs permutation feature importance analysis with the prediction model and your data

### `ShapExplainer`
Performs SHAP feature importance analysis with the prediction model and your data

### `SmartTesting`

Uses the smart_testing tool to find subgroups of the dataset that the model may perform poorly on.
The tool provides a descriptive summary of the subgroups.


### `UploadAndSummarizeExamplePaper`
Allows you to upload an example paper in PDF format.

### `UploadDataFile`
Allows you to upload your data file in CSV format.

### `UploadDataMultipleFiles`
Allows you to upload your data files (training dataset, and if you have it, a test/evaluation dataset) in CSV format.
