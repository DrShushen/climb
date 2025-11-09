import ast
import hashlib
import itertools
import re
import warnings
from typing import Any, Dict, Optional

import pandas as pd
from openai import AzureOpenAI
from pydantic import BaseModel, PrivateAttr
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Import calculate_group_statistics function
from .utils import calculate_group_statistics, calculate_group_statistics_string


def generate_combinations_for_variable(var_values):
    # Single value combinations
    single_value_combos = [(val,) for val in var_values]
    # Pair value combinations
    pair_value_combos = list(itertools.combinations(var_values, 2))
    return single_value_combos + pair_value_combos


def clean_query_string(query):
    # Replace or remove unwanted characters
    query = query.replace("\\", "")  # Remove backslashes
    query = query.replace("'", "")  # Remove single quotes if necessary

    return query


def convert_to_string_condition(query):
    # Regex pattern to extract the column name and the condition
    pattern = r"\((\w+)\s*==\s*([\d\.]+\s*-\s*[\d\.]+)\)"
    match = re.search(pattern, query)

    if match:
        column_name = match.group(1)
        condition = match.group(2)

        # Constructing the new query
        new_query = f"({column_name} == '{condition}')"
        return new_query
    else:
        # Returning the old query
        return query


class SMART(BaseModel):
    llm: AzureOpenAI
    config: dict
    verbose: bool = True

    _subgroups: Optional[Dict] = PrivateAttr(default=None)
    # Hypotheses are stored as a string
    _hypotheses: Optional[str] = PrivateAttr(default=None)
    _updated_hypotheses: Optional[str] = PrivateAttr(default=None)

    # Get context and context targets as strings
    context: Optional[str] = None
    context_target: Optional[str] = None
    optimal_queries: Optional[Dict] = None
    task: Optional[str] = None
    _selfrefine_steps: Dict[int, str] = PrivateAttr(default_factory=dict)
    _subgroup_cache: Dict[str, Dict] = PrivateAttr(default_factory=dict)
    _unique_values: Optional[Dict] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _get_llm_response(self, input_text, system_message=None, metadata_output=False, modelid=None):
        if self.verbose:
            print("----------INPUT TEXT --------------")
            print(input_text)

        if system_message is None:
            # LLM response with/without a system message
            response = self.llm.chat.completions.create(
                model=self.config["engine"],
                messages=[{"role": "user", "content": input_text}],
                temperature=self.config["temperature"],
                # seed=self.config['seed'],
            )
        else:
            # Get the response from the LLM with a system message
            response = self.llm.chat.completions.create(
                model=self.config["engine"],
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": input_text}],
                temperature=self.config["temperature"],
                seed=self.config["seed"],
            )
        message = response.choices[0].message.content

        if self.verbose:
            print("----------LLM RESPONSE TEXT--------------")
            print(message)

        if metadata_output:
            metadata = {
                "tools": response.choices[0].message.tool_calls,
                "function calls": response.choices[0].message.function_call,
            }
            return message, metadata
        else:
            return message

    def _generate_cache_key(self, X: pd.DataFrame) -> str:
        """
        Generates a unique cache key based on the DataFrame columns.
        """
        column_string = ",".join(sorted(X.columns))
        return hashlib.sha256(column_string.encode()).hexdigest()

    def clear_cache(self):
        """Clears the subgroup cache."""
        self._subgroup_cache.clear()
        if self.verbose:
            print("Cache cleared.")

    def _get_unique_values(self, X, unique_threshold: int = 30) -> Dict[str, Any]:
        """
        Parses through the dataset and returns the unique values for each column.
        """
        unique_values = {}
        for col in X.columns:
            if len(X[col].unique()) <= unique_threshold:
                unique_values[col] = list(X[col].unique())
            else:
                if X[col].dtype in ["int64", "float64"]:
                    unique_values[col] = {"min": X[col].min(), "mean": X[col].mean(), "max": X[col].max()}
                else:
                    unique_values[col] = "Too many unique values"
                    if self.verbose:
                        warnings.warn(f"Column {col} has too many unique values.", UserWarning)
        return unique_values

    def fit(
        self,
        X: pd.DataFrame,
        context: Optional[str] = None,
        context_target: Optional[str] = None,
        n: int = 5,
        evaluate_feasibility=False,
    ):
        """Finds subgroups by generating hypotheses, operationalizing them, and summarizing the findings"""

        cache_key = self._generate_cache_key(X)

        # Check if the result is already cached
        if cache_key in self._subgroup_cache:
            self._subgroups = self._subgroup_cache[cache_key]
            print("Cached subgroups loaded.")
            return self

        unique_values = self._get_unique_values(X)
        # Save the unique values
        self._unique_values = unique_values
        task = self._construct_task(unique_values, context, context_target, n)

        # Update the context and context target
        self.context = context
        self.context_target = context_target
        self.task = task
        # Evaluate feasibility
        if evaluate_feasibility:
            feasibility_response = self._feasibility_check(unique_values, context, context_target)
            if feasibility_response.lower().strip() == "yes":  # pyright: ignore
                print("Group discovery is possible. Discovering subgroups...")
            elif feasibility_response.lower().strip() == "no":  # pyright: ignore
                print("No groups discovered")
                self._subgroups = {}
                self._subgroup_cache[cache_key] = self._subgroups

                return self
            else:
                print(f"The response from the feasibility status is: {feasibility_response.lower().strip()}")  # pyright: ignore

        # Assuming that the task is feasible, generating hypotheses
        hypotheses = self._get_llm_response(task)
        self._hypotheses = hypotheses  # pyright: ignore

        # Operationalizing the hypotheses
        operationalization_prompt = self._construct_operationalization_prompt(
            hypotheses, unique_values, context, context_target
        )
        operationalizations = self._get_llm_response(operationalization_prompt)

        # Summarizing the findings
        summarization_prompt = self._construct_summarization_prompt(operationalizations, unique_values)
        summary_dict = self._get_llm_response(summarization_prompt)

        # Set regex pattern
        pattern = r"\{.*?\}"

        try:
            summary_dict = re.findall(pattern, summary_dict, re.DOTALL)[0]  # pyright: ignore
            self._subgroups = ast.literal_eval(summary_dict)

        except Exception:
            correction_prompt = f"""The following is a dictionary that contains the subgroups. Return ONLY the dictionary with no additional text before or after. {summary_dict}"""
            if self.verbose:
                print(correction_prompt)
            response_correction = self._get_llm_response(correction_prompt)
            summary_dict = re.findall(pattern, response_correction, re.DOTALL)[0]  # pyright: ignore
            self._subgroups = ast.literal_eval(summary_dict)

        # Adjust the subgroup queries
        self._adjust_subgroup_queries(X)

        # Cache the subgroup findings
        self._subgroup_cache[cache_key] = self._subgroups  # pyright: ignore
        return self

    def find_subgroup_variables(
        self, X: pd.DataFrame, context: Optional[str] = None, context_target: Optional[str] = None, n: int = 30
    ):
        """Finds subgroups by generating hypotheses, operationalizing them, and summarizing the findings"""

        cache_key = self._generate_cache_key(X)

        # Check if the result is already cached
        if cache_key in self._subgroup_cache:
            self._subgroups = self._subgroup_cache[cache_key]
            print("Cached subgroups loaded.")
            return self

        unique_values = self._get_unique_values(X)
        # Save the unique values
        self._unique_values = unique_values
        task = self._construct_task_hypotheses(unique_values, context, context_target, n)

        # Update the context and context target
        self.context = context
        self.context_target = context_target
        self.task = task

        hypotheses = self._get_llm_response(task)
        self._hypotheses = hypotheses  # pyright: ignore

        # Operationalizing the hypotheses
        operationalization_prompt = self._construct_operationalization_subgroups(
            hypotheses, unique_values, context, context_target
        )
        operationalizations = self._get_llm_response(operationalization_prompt)

        # Set regex pattern
        pattern = r"\{.*?\}"

        try:
            summary_dict = re.findall(pattern, operationalizations, re.DOTALL)[0]  # pyright: ignore
            self._subgroups = ast.literal_eval(summary_dict)
            # Loop and ensure all of the subgroups are lists. If not, convert to lists.
            for key, value in self._subgroups.items():  # pyright: ignore
                if not isinstance(value, list):
                    self._subgroups[key] = [value]  # pyright: ignore

        except Exception:
            correction_prompt = f"""The following is a dictionary that contains the subgroups. Return ONLY the dictionary with no additional text before or after. {operationalizations}"""
            if self.verbose:
                print(correction_prompt)
            response_correction = self._get_llm_response(correction_prompt)
            summary_dict = re.findall(pattern, response_correction, re.DOTALL)[0]  # pyright: ignore
            self._subgroups = ast.literal_eval(summary_dict)
            # Loop and ensure all of the subgroups are lists. If not, convert to lists.
            for key, value in self._subgroups.items():  # pyright: ignore
                if not isinstance(value, list):
                    self._subgroups[key] = [value]  # pyright: ignore
        # Cache findings if not cached
        self._subgroup_cache[cache_key] = self._subgroups  # pyright: ignore
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts group membership for each observation in the DataFrame.

        :param X: DataFrame containing the observations.
        :return: DataFrame with additional boolean columns indicating group membership.
        """
        # Check if all column names in the dictionary conditions are valid
        valid_columns = set(X.columns)
        for group, condition in self._subgroups.items():  # pyright: ignore
            # Extract column names from the condition
            columns_in_condition = [word for word in condition.split() if word in valid_columns]
            if not columns_in_condition:
                warnings.warn(f"No valid columns found in condition for group {group}: {condition}", UserWarning)

        # TODO: Check if values in the dictionary exist in the DataFrame

        # Create group columns
        for group, condition in self._subgroups.items():  # pyright: ignore
            try:
                indices_condition = X.query(condition).index
                bool_condition = X.index.isin(indices_condition)
                X[f"group_{group}"] = bool_condition
            except Exception as e:
                warnings.warn(f"Failed to apply condition for group {group}: {condition}. Error: {e}", UserWarning)

        return X

    @property
    def subgroups(self):
        """Return the identified subgroups"""
        return self._subgroups

    @property
    def hypotheses(self):
        """Return the hypotheses"""
        return self._hypotheses

    def _self_refine(self, unique_values, context, context_target, previous_response, system_message, n=3):
        """Self-refines an answer multiple times"""

        for iter_ in range(n):
            selfrefine_task = f"""
            The context is: {context} and the target variable is {context_target} with the following columns: {", ".join(unique_values.keys())}. \nPrevious answer: {previous_response}. \n\nTASK: Critically evaluate the answer below and then re-write it. Make sure to follow the instructions provided before.  \n\n
            """
            previous_response = self._get_llm_response(selfrefine_task, system_message=system_message)

            self._selfrefine_steps[iter_] = previous_response  # pyright: ignore

        return previous_response

    def _feasibility_check(self, unique_values, context, context_target, system_message=None, n_refine=1):
        """Checks if the task is feasible. Logic: (1) perform a feasibility query; (2) self-refine the answer; (3) convert to boolean"""

        feasibility_task = f"""Your task is to evaluate whether it is reasonable to to search for subgroups where a predictive model which perform suboptimally. Given the following context about the dataset: {context} and the target variable: {context_target}, and the following columns: {", ".join(unique_values.keys())}, is it reasonable to search for societally meaningful subgroups? Write the reasons why yes, then why no, and provide an overall summary. """

        if system_message is None:
            system_message = """You are an expert at clearly evaluating whether there is a direct connection between the covariates and the outcome variable. Your goal is to determine whether such a connection exists in academic literature or other sources. Avoid making ridiculous connections that are unlikely to hold in reality. Be critical. Focus on avoiding false positives (i.e. relationships that might not exist) because it is costly to test these assumptions and we might overfit the results. Avoid speculative or weak connections. Prioritize false negatives (missing connections) than false positives (offering weak connections that might not hold)"""

        # Evaluating the feasibility of the response
        if self.verbose:
            print("Evaluating feasibility of the response...")
        feasibility_response = self._get_llm_response(feasibility_task, system_message=system_message)

        if self.verbose:
            print("Self-refining answer...")

        # Refining the answer
        feasibility_response = self._self_refine(
            unique_values, context, context_target, feasibility_response, system_message, n=n_refine
        )

        # Convert to boolean
        boolean_task = f"""Your task is to return an answer 'yes' or 'no' on whether it is worthwile to inspect subgroups, based on the response provided below. Answer: {feasibility_response} \n\nTASK: Answer whether it is worthwile to inspect subgroups, based on the response provided above. Answer: 'yes' or 'no'."""
        feasibility_boolean_response = self._get_llm_response(boolean_task)

        return feasibility_boolean_response

    def _construct_task_hypotheses(self, unique_values, context, context_target, n):
        """Constructs the task description for the LLM."""
        task = f"""Your task is to propose possible hypotheses as to which subgroups within the dataset might have worse predictive performance than on average because of societal bias in the dataset, insufficient data, other relationships, or others. The subgroups might be based on any of the provided characteristics, as well as on any combination of such characteristics. 
        
        Dataset information: {context}. {context_target}
        
        The dataset contains {len(unique_values)} columns. The columns are {", ".join(unique_values.keys())}. 
        
        Task: Create {n} hypotheses as to which subgroups within the dataset the model will perform worse than on average because of societal biases or other reasons. Important: Your hypothesis can contain either one variable or two variables in the condition. Therefore, your goal is to find discrepancies in the model's performance, not the underlying data outcomes. Justify why you think that for each of the {n} hypotheses. You must use this format: Hypothesis: <>; Justification: <>, with the hypothesis and justification on the same line separated by a ';'.
        e.g.
        Hypothesis 1: <Hypothesis>; Justification: <Justification>
        Hypothesis 2: <Hypothesis>; Justification: <Justification>
        
        """
        return task

    def _construct_task(self, unique_values, context, context_target, n):
        """Constructs the task description for the LLM."""
        task = f"""Your task is to propose possible hypotheses as to which subgroups within the dataset might have worse predictive performance than on average because of societal bias in the dataset, insufficient data, other relationships, or others. The subgroups might be based on any of the provided characteristics, as well as on any combination of such characteristics. 
        
        Dataset information: {context}. {context_target}
        
        The dataset contains {len(unique_values)} columns. The columns are {", ".join(unique_values.keys())}. The values are {str(unique_values.items())}
        
        Task: Create {n} hypotheses as to which subgroups within the dataset the model will perform worse than on average because of societal biases or other reasons. Therefore, your goal is to find discrepancies in the model's performance, not the underlying data outcomes. Justify why you think that. You must use this format of the output: Hypothesis: <>; Justification: <>, with the hypothesis, justification, and operationalization on the same line separated by a ';'.
        e.g.
        Hypothesis 1: <Hypothesis>; Justification: <Justification>
        Hypothesis 2: <Hypothesis>; Justification: <Justification>
        """
        return task

    def _construct_operationalization_prompt(self, hypotheses, unique_values, context, context_target):
        """Constructs the operationalization prompt for the LLM."""
        operationalization_prompt = f"""
        The following are hypotheses about which people within a dataset the model might underperform on. 
        Propose specific ranges for each hypothesis. Hypotheses: {hypotheses}.

        Dataset information: {context}. {context_target}

        The dataset contains {len(unique_values)} columns. The columns are {", ".join(unique_values.keys())}. The values are {str(unique_values.items())}

        TASK: Propose specific variable ranges for each hypothesis such that they are clearly operationalizable and defined. **Use the exact column names with the correct casing as they appear in the dataset**. Ensure that each Operationalization is a single-line expression without line breaks. You must use this format: Hypothesis: <>; Operationalization: <>, with the hypothesis and operationalization on the same line separated by a ';'.
        e.g.
        Hypothesis 1: <Hypothesis>; Operationalization: <Operationalization>
        Hypothesis 2: <Hypothesis>; Operationalization: <Operationalization>
        """
        return operationalization_prompt

    def _construct_operationalization_subgroups(self, hypotheses, unique_values, context, context_target):
        """Constructs the operationalization prompt for the LLM."""
        operationalization_prompt = f"""
        The following are hypotheses about which people within a dataset the model might underperform on. 
        Propose specific ranges for each hypothesis. Hypotheses: {hypotheses}.

        TASK: return a dictionary that contains an index number as the key and the column value as the value. If there are multiple columns in that hypothesis, return them in a list. There are the column names: {", ".join(unique_values.keys())}.
        """
        return operationalization_prompt

    def _construct_revised_operationalization_prompt(self, new_context, unique_values, context, context_target):
        """Constructs the operationalization prompt for the LLM."""
        operationalization_prompt = f"""
        You have access to the following information.

        Dataset information: {context}. {context_target}

        The dataset contains {len(unique_values)} columns. The columns are {", ".join(unique_values.keys())}. The values are {str(unique_values.items())}

        However, you are no longer working with the same data as just described. Rather, this is the context: {new_context}.

        These are the hypotheses: {self._updated_hypotheses}.

        TASK: Propose specific variable ranges for each hypothesis such that they are clearly operationalizable and defined. You must use this format: Hypothesis: <>; Operationalization: <>, with the hypothesis and operationalization on the same line separated by a ';'.
        e.g.
        Hypothesis 1: <Hypothesis>; Operationalization: <Operationalization>
        Hypothesis 2: <Hypothesis>; Operationalization: <Operationalization>
        """
        return operationalization_prompt

    def _construct_summarization_prompt(self, operationalizations, unique_values):
        """Constructs the summarization prompt for the LLM."""

        summarization_prompt = f"""
        The following are groups that are defined based on the dataset. Convert them into a Python dictionary format. Each group should be represented as a key-value pair in the dictionary, where the key is an index (0 to 4), and the value is a string representing the group using Python syntax and logical operators. For multiple conditions, use Python's logical 'and' ('&&') or 'or' ('||'). Ensure the format is a valid Python dictionary. 

        Examples:
        - Single Condition: {{0: 'X > 45'}}
        - Multiple Conditions: {{1: '(X > 45) and (Y < 20)'}}
        - String conditions: {{2: 'X == '45 - 60'}}

        Groups to summarize: {operationalizations}
        Column names: {", ".join(unique_values.keys())}
        Column values: {str(unique_values.items())}
        """
        return summarization_prompt

    def _adjust_subgroup_queries(self, X: pd.DataFrame, n_subgroups=1):
        """Adjusts the subgroup queries if they do not exist in the dataset."""

        unique_values = self._unique_values

        for group, condition in list(self._subgroups.items())[:n_subgroups]:  # pyright: ignore
            try:
                # Check if the condition yields any rows
                if len(X.query(condition)) == 0:
                    # Call LLM to adjust the condition
                    adjustment_prompt = self._construct_adjustment_prompt(condition, unique_values)
                    adjusted_condition = self._get_llm_response(adjustment_prompt)

                    # Update the condition
                    self._subgroups[group] = adjusted_condition  # pyright: ignore
                    print("Primary condition: ", condition)
                    print("Adjusted condition: ", adjusted_condition)
            except Exception as e:
                warnings.warn(f"Error evaluating condition for group {group}: {condition}. Error: {e}", UserWarning)

                # Call LLM to adjust the condition
                print("Adjusting condition...")
                unique_values_adj = {k: v for k, v in unique_values.items() if k in condition}  # pyright: ignore

                adjustment_prompt = self._construct_adjustment_prompt(condition, unique_values_adj)
                adjusted_condition = self._get_llm_response(adjustment_prompt)
                # TODO - make this part more robust.
                try:
                    X.query(adjusted_condition)  # pyright: ignore
                except Exception as e:
                    adjusted_condition = clean_query_string(adjusted_condition)
                    # Check if the condition has any strings assuming it is a single condition
                    if "and" not in adjusted_condition:
                        adjusted_condition = convert_to_string_condition(adjusted_condition)

                print("Primary condition: ", condition)
                print("Adjusted condition: ", adjusted_condition)
                # Update the condition
                self._subgroups[group] = adjusted_condition  # pyright: ignore

    def _construct_adjustment_prompt(self, condition, unique_values):
        """
        Constructs the prompt for adjusting a subgroup condition.
        """

        adjustment_prompt = f"""
        The following pandas query does not match any rows in the dataset: '{condition}'. This condition uses a column and a value to filter values, but the data types are incorrect.
        Adjust the condition using the datasets values and columns, such that the condition would work on the unique values in the dataset, and the condition would be as close as possible to the original one.
        Unique values/statistics of relevant columns: {str(unique_values)}
        
        TASK: Provide an adjusted dataframe query that uses the specific unique values in the dataset which would not throw an error and would closely match the original condition.
        Provide ONLY the query. Query: 
        """
        return adjustment_prompt

    def extract_hypotheses_and_justifications(self):
        """
        Extracts hypotheses and their justifications from the provided text and
        organizes them into a pandas DataFrame. Each hypothesis and its justification
        are in separate columns. The number of rows corresponds to the number of hypotheses.

        Returns:
            pd.DataFrame: A DataFrame with 'Hypothesis', 'Justification', and 'Operationalization' columns.
        """
        text = self._hypotheses
        subgroups = self.subgroups

        if not text:
            print("No hypotheses found in '_hypotheses'. Ensure that 'fit' has been called.")
            return pd.DataFrame(columns=["Hypothesis", "Justification", "Operationalization"])

        # Split the text into lines
        lines = text.split("\n")

        # Lists to store hypotheses, justifications, and operationalizations
        hypotheses = []
        justifications = []
        operationalizations = []

        # Initialize a counter for operationalizations
        subgroup_keys = list(subgroups.keys())  # pyright: ignore
        subgroup_index = 0

        # Process each line to extract hypothesis and justification
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if line.startswith("Hypothesis"):
                hypothesis_part, justification_part = line.split("; ", 1)
                # Extract text after 'Hypothesis: ' and 'Justification: ' if present
                hypothesis = hypothesis_part.split(": ", 1)[1] if ": " in hypothesis_part else hypothesis_part
                justification = (
                    justification_part.split(": ", 1)[1] if ": " in justification_part else justification_part
                )
                hypotheses.append(hypothesis)
                justifications.append(justification)

                # Assign operationalization if available
                if subgroup_index < len(subgroup_keys):
                    subgroup_key = subgroup_keys[subgroup_index]
                    operationalization = subgroups[subgroup_key]  # pyright: ignore
                    operationalizations.append(operationalization)
                    subgroup_index += 1
                else:
                    # If there are more hypotheses than subgroups, assign None or a default
                    operationalizations.append(None)

        # Create a DataFrame
        df = pd.DataFrame(
            {"Hypothesis": hypotheses, "Justification": justifications, "Operationalization": operationalizations}
        )

        # Optional: Check for alignment between hypotheses and operationalizations
        if len(df) != len(subgroup_keys):
            print("Number of hypotheses does not match number of subgroups.")

        return df

    def generate_model_report(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        keys_calculate=[
            "group_size",
            "support",
            "p_value_bootstrap",
            "num_criteria",
            "outcome_diff",
            "accuracy_diff",
            "odds_ratio_outcome",
            "odds_ratio_acc",
            "lift_outcome",
            "lift_acc",
            "weighted_relative_outcome",
            "weighted_relative_accuracy",
        ],
    ):
        """Currenty supported only for the subgroup_finder without the self-falsification mechanism"""
        table_summary = self.extract_hypotheses_and_justifications()
        table_summary_train = table_summary.copy()
        for oper in table_summary_train["Operationalization"]:
            for key in keys_calculate:
                table_summary_train.loc[table_summary_train["Operationalization"] == oper, key] = (
                    calculate_group_statistics(X_train, y_train, model, oper)[key]
                )

        table_summary_test = table_summary.copy()
        for oper in table_summary_test["Operationalization"]:
            for key in keys_calculate:
                table_summary_test.loc[table_summary_test["Operationalization"] == oper, key] = (
                    calculate_group_statistics(X_test, y_test, model, oper)[key]
                )

        input_text = f"""
        The following is the context: {self.context}. The following is the target context: {self.context_target}. The following is a table summarizing the information about the results on the training dataset: {table_summary_train}. The following is a table summarizing the information about the results on the test dataset: {table_summary_test}.

        TASK: Write recommendations to the user based on the results. Answer these questions: \n 1. When does the model fail and when it is reliable? \n 2. What should the end user be aware of before deploying the model? Keep your recommendations short, brief, and actionable. Avoid repeating information which has already been said.
        """
        response_recommendations = self._get_llm_response(input_text)
        return response_recommendations

    def revise_hypotheses(self, new_context: str) -> str:
        """
        Revises the existing hypotheses based on a new context.

        :param new_context: A string representing the new context to consider for revising hypotheses.
        :return: A string containing the set of new hypotheses.
        """
        if self._hypotheses is None:
            raise ValueError("No existing hypotheses to revise. Please run 'fit' method first.")

        # Constructing the new prompt
        revise_prompt = f"""
            The original context for the dataset was: {self.context} with the target variable {self.context_target}. Based on this, the following hypotheses were generated: {self._hypotheses}.

            The subgroups identified were: {str(self._subgroups)}

            Now, a new context has emerged: {new_context}. 

            TASK: Considering both the original and the new context, revise the earlier hypotheses. Generate a new set of hypotheses instead of the old hypotheses that take into account any changes or additional information provided by the new context. Ensure that these hypotheses are relevant and applicable to the updated scenario. Assume access to the same data as before.
            """

        # Getting the response from LLM
        new_hypotheses = self._get_llm_response(revise_prompt)

        # Revise the hypotheses
        if self.verbose:
            print("Revising hypotheses...")
        new_hypotheses = self._self_refine(
            self._unique_values, new_context, self.context_target, new_hypotheses, system_message=revise_prompt, n=2
        )

        # Updating the hypotheses attribute
        self._updated_hypotheses = new_hypotheses  # pyright: ignore

        return new_hypotheses  # pyright: ignore

    def revise_fit(self, new_context, X):
        unique_values = self._unique_values
        # Operationalizing the hypotheses
        operationalization_prompt = self._construct_revised_operationalization_prompt(
            new_context, unique_values, self.context, self.context_target
        )
        operationalizations = self._get_llm_response(operationalization_prompt)

        # Summarizing the findings

        summarization_prompt = self._construct_summarization_prompt(operationalizations, unique_values)
        summary_dict = self._get_llm_response(summarization_prompt)

        # Set regex pattern
        pattern = r"\{.*?\}"

        try:
            summary_dict = re.findall(pattern, summary_dict, re.DOTALL)[0]  # pyright: ignore
            subgroups = ast.literal_eval(summary_dict)

        except Exception:
            correction_prompt = f"""The following is a dictionary that contains the subgroups. Return ONLY the dictionary with no additional text before or after. Return an empty dictionary if none exists. \n {summary_dict}"""
            if self.verbose:
                print(correction_prompt)
            response_correction = self._get_llm_response(correction_prompt)
            summary_dict = re.findall(pattern, response_correction, re.DOTALL)[0]  # pyright: ignore
            subgroups = ast.literal_eval(summary_dict)

        return subgroups

    def get_optimal_split_query(
        self, dataframe, features, outcome, min_group_size=10, test_for_min=True, max_group_size=float("inf")
    ):
        """
        Generates a query string for splitting the dataframe into two subgroups
        where the difference in the outcome variable is maximized, based on up to three features.

        :param dataframe: A pandas DataFrame containing the data.
        :param features: A list of feature variable names (up to 3 features).
        :param outcome: The name of the outcome variable.
        :param min_group_size: The minimum size of each group.
        :return: A query string for the subgroup where the outcome is minimized.
        """

        if not all(feature in dataframe.columns for feature in features):
            raise ValueError("All features must be present in the dataframe")
        if outcome not in dataframe.columns:
            raise ValueError("Outcome variable must be present in the dataframe")

        # Determine if the outcome variable is continuous or categorical
        if pd.api.types.is_numeric_dtype(dataframe[outcome]):
            tree_model = DecisionTreeRegressor(max_depth=len(features))
        else:
            tree_model = DecisionTreeClassifier(max_depth=len(features))

        # Fit the model
        tree_model.fit(dataframe[features], dataframe[outcome])

        # Function to recursively traverse the tree and find the optimal split
        def traverse_tree(node=0, depth=0, conditions=[]):
            if (
                tree_model.tree_.children_left[node] == tree_model.tree_.children_right[node]  # pyright: ignore
            ):  # Leaf node
                if not conditions:  # Check for empty conditions
                    return None, float("-inf")
                # Evaluate split
                left_indices = dataframe.query(" and ".join(conditions)).index
                right_indices = dataframe.index.difference(left_indices)

                if test_for_min:
                    if len(left_indices) < min_group_size or len(right_indices) < min_group_size:
                        return None, float("-inf")
                else:
                    if len(left_indices) < max_group_size and len(left_indices) > min_group_size:
                        left_mean = dataframe.loc[left_indices, outcome].mean()
                        right_mean = dataframe.loc[right_indices, outcome].mean()
                        discrepancy = abs(left_mean - right_mean)
                        if left_mean >= right_mean:
                            return " and ".join(conditions), discrepancy
                        else:
                            conditions = [
                                cond.replace("<=", ">") if "<=" in cond else cond.replace(">", "<=")
                                for cond in conditions
                            ]
                            return " and ".join(conditions), discrepancy
                    elif len(right_indices) < max_group_size and len(right_indices) > min_group_size:
                        right_mean = dataframe.loc[right_indices, outcome].mean()
                        left_mean = dataframe.loc[left_indices, outcome].mean()
                        discrepancy = abs(left_mean - right_mean)
                        if right_mean <= left_mean:
                            conditions = [
                                cond.replace("<=", ">") if "<=" in cond else cond.replace(">", "<=")
                                for cond in conditions
                            ]
                            return " and ".join(conditions), discrepancy
                        else:
                            return " and ".join(conditions), discrepancy
                    else:
                        return None, float("-inf")

                left_mean = dataframe.loc[left_indices, outcome].mean()
                right_mean = dataframe.loc[right_indices, outcome].mean()
                discrepancy = abs(left_mean - right_mean)

                if left_mean >= right_mean:
                    return " and ".join(conditions), discrepancy
                else:
                    conditions = [
                        cond.replace("<=", ">") if "<=" in cond else cond.replace(">", "<=") for cond in conditions
                    ]
                    return " and ".join(conditions), discrepancy

            # Not a leaf node, continue splitting
            feature = features[tree_model.tree_.feature[node]]  # pyright: ignore
            threshold = tree_model.tree_.threshold[node]  # pyright: ignore

            left_condition = f"{feature} <= {threshold}"
            right_condition = f"{feature} > {threshold}"

            # Traverse left and right
            left_query, left_discrepancy = traverse_tree(
                tree_model.tree_.children_left[node],  # pyright: ignore
                depth + 1,
                conditions + [left_condition],
            )
            right_query, right_discrepancy = traverse_tree(
                tree_model.tree_.children_right[node],  # pyright: ignore
                depth + 1,
                conditions + [right_condition],
            )

            if left_discrepancy == right_discrepancy:
                return left_query, left_discrepancy
            else:
                return (
                    (left_query, left_discrepancy)
                    if left_discrepancy > right_discrepancy
                    else (right_query, right_discrepancy)
                )

        return traverse_tree()[0]

    def get_optimal_queries_strings(self, X, y, model, min_group_size=10, n_groups=10, alpha=0.05):
        optimal_queries = {}

        # Fit the model
        ohe = OneHotEncoder(sparse=False)
        X_dummies = pd.DataFrame(ohe.fit_transform(X), columns=ohe.get_feature_names_out())
        X_dummies.index = X.index

        model.fit(X_dummies, y)
        for group_id, variables in self.subgroups.items():  # pyright: ignore
            max_difference = float("-inf")
            optimal_query = None

            # Check if there is only one variable
            if len(variables) == 1:
                var = variables[0]
                # Generate all non-empty combinations of unique values for the single variable
                value_combinations = itertools.chain.from_iterable(
                    itertools.combinations(X[var].unique(), r) for r in range(1, len(X[var].unique()) + 1)
                )
            else:
                # Generate all relevant combinations for each variable
                variable_combinations = [generate_combinations_for_variable(X[var].unique()) for var in variables]

                # Generate combinations of these combinations across variables
                value_combinations = itertools.product(*variable_combinations)

            # Iterate over each combination
            for combo in value_combinations:
                query_parts = []
                if len(variables) == 1:
                    var = variables[0]
                    if len(combo) == 1:
                        query_parts.append(f"{var} == '{combo[0]}'")
                    else:
                        query_parts.append(f"{var} in {combo}")
                else:
                    for var, vals in zip(variables, combo):
                        if len(vals) == 1:
                            query_parts.append(f"{var} == '{vals[0]}'")
                        else:
                            query_parts.append(f"{var} in {vals}")
                query = " and ".join(query_parts)

                # Filter the dataframe based on the query and calculate accuracy difference
                subgroup_X = X.query(query)
                if len(subgroup_X) >= min_group_size:
                    subgroup_y = y[subgroup_X.index]
                    subgroup_X_ohe = pd.DataFrame(ohe.transform(subgroup_X), columns=ohe.get_feature_names_out())  # pyright: ignore
                    subgroup_X_ohe.index = subgroup_X.index
                    accuracy_diff = self.calculate_accuracy_difference(X_dummies, y, model, subgroup_X_ohe, subgroup_y)

                    if accuracy_diff > max_difference:
                        max_difference = accuracy_diff
                        optimal_query = query

            if optimal_query:
                optimal_queries[group_id] = optimal_query

        # Filter top
        subgroup_results = {}
        for group, condition in optimal_queries.items():
            try:
                results_group = calculate_group_statistics_string(X, y, model, condition, ohe)
                significant_result = results_group["p_value_bootstrap"] < alpha
                subgroup_results[group] = {"results": results_group, "significant": significant_result}
            except Exception as e:
                print(f"Error with group {group}: {condition}", e)
                continue
        # Order subgroups by p-value
        subgroup_results = sorted(
            subgroup_results.items(), key=lambda x: x[1]["results"]["accuracy_diff"], reverse=True
        )
        # Get the top n subgroups

        # Filter based on significant to only include when it is significant
        # subgroup_results = [subgroup for subgroup in subgroup_results if subgroup[1]['significant']]

        top_subgroups = subgroup_results[:n_groups]
        # Get only the queries of the top n subgroups
        top_queries = [query[1]["results"]["query"] for query in top_subgroups]
        # Convert to dictionary
        top_queries = {i: query for i, query in enumerate(top_queries)}
        return top_queries

    def calculate_outcome_difference(self, y, full_y):
        """
        Calculates the difference in the proportion of the most common outcome
        between the subgroup and the full dataset.

        :param y: The outcome variable for the subgroup.
        :param full_y: The outcome variable for the full dataset.
        :return: The difference in proportions.
        """
        if y.empty or full_y.empty:
            return 0

        # Get the most common outcome in the full dataset
        most_common_outcome = full_y.mode()[0]

        # Calculate the proportion of this outcome in both the subgroup and the full dataset
        subgroup_proportion = (y == most_common_outcome).mean()
        full_dataset_proportion = (full_y == most_common_outcome).mean()

        # Calculate the difference in proportions
        difference = abs(subgroup_proportion - full_dataset_proportion)

        return difference

    def get_optimal_queries(
        self,
        X,
        y,
        model,
        outcome="y_failures",
        min_group_size=10,
        alpha=0.1,
        n_groups=10,
        test_for_min=True,
        max_group_size=float("inf"),
    ):
        """
        Generates a list of query strings for splitting the dataframe into two subgroups
        where the difference in the outcome variable is maximized, based on up to three features.

        :param dataframe: A pandas DataFrame containing the data.
        :param features: A list of feature variable names (up to 3 features).
        :param outcome: The name of the outcome variable.
        :param min_group_size: The minimum size of each group.
        :param n_queries: The number of queries to generate.
        :return: A list of query strings for the subgroup where the outcome is maximized.
        """
        dataframe = X.copy()
        # Calculate model failures
        y_pred = model.predict(X)
        y_failures = (y_pred != y).astype(int)

        dataframe[outcome] = y_failures
        optimal_queries = {}
        # Get groups
        group_variables = self.subgroups

        # Loop through the groups
        for group, condition in group_variables.items():  # pyright: ignore
            # Get the optimal query for each group
            optimal_query = self.get_optimal_split_query(
                dataframe, condition, outcome, min_group_size, test_for_min, max_group_size
            )
            optimal_queries[group] = optimal_query

        self.optimal_queries = optimal_queries

        # Remove values which are None and which repeat themselves
        optimal_queries = {k: v for k, v in optimal_queries.items() if v is not None}

        # Loop over each query and if the query already exists, skip. Otherwise, add it to a new dictionary
        optimal_queries_unique = {}
        for group, query in optimal_queries.items():
            if query not in optimal_queries_unique.values():
                optimal_queries_unique[group] = query

        optimal_queries = optimal_queries_unique

        subgroup_results = {}
        for group, condition in optimal_queries.items():
            try:
                results_group = calculate_group_statistics(X, y, model, condition)
                significant_result = results_group["p_value_bootstrap"] < alpha
                subgroup_results[group] = {"results": results_group, "significant": significant_result}
            except Exception as e:
                print(f"Error with group {group}: {condition}", e)
                continue
        # Order subgroups by p-value
        subgroup_results = sorted(
            subgroup_results.items(), key=lambda x: x[1]["results"]["accuracy_diff"], reverse=True
        )
        # Get the top n subgroups

        top_subgroups = subgroup_results[:n_groups]
        # Get only the queries of the top n subgroups
        top_queries = [query[1]["results"]["query"] for query in top_subgroups]
        # Convert to dictionary
        top_queries = {i: query for i, query in enumerate(top_queries)}
        return top_queries

    def calculate_accuracy_difference(self, X_tr, y, model, subgroup_X_tr, subgroup_y):
        """
        Calculates the accuracy difference between the model's predictions on the full dataset
        and a specific subgroup.

        :param X_tr: Transformed predictor variables of the full dataset.
        :param y: Outcome variable of the full dataset.
        :param model: Trained model to make predictions.
        :param subgroup_X_tr: Transformed predictor variables of the subgroup.
        :param subgroup_y: Outcome variable of the subgroup.
        :return: The accuracy difference.
        """
        try:
            # Calculate accuracy on the full dataset and the subgroup
            accuracy_dataset = accuracy_score(y, model.predict(pd.get_dummies(X_tr, drop_first=True)))
            accuracy_subgroup = accuracy_score(subgroup_y, model.predict(subgroup_X_tr))
        except Exception as e:
            # Log the exception if needed
            print("Error in accuracy calculation:", e)
            return 0

        # Compute the accuracy difference
        accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)
        return accuracy_diff
