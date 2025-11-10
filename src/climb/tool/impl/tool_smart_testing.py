import os
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from autoprognosis.utils.serialization import load_model_from_file
from sklearn.model_selection import train_test_split

from climb.common import Session
from climb.tool.impl.smart_testing_helpers.SMART import SMART
from climb.tool.impl.sub_agents import create_llm_client

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def smart_testing(
    tc: ToolCommunicator,
    data_path: str,
    model_path: str,
    context: str,
    context_target: str,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
    workspace: str,
):
    """

    Args:
        tc (ToolCommunicator): The tool communicator object.
        data_path (str): The path to the input CSV file.
        workspace (str): The workspace directory path.
    """
    workspace = Path(workspace)  # pyright: ignore
    df = pd.read_csv(data_path)

    # Define features and target
    X = df.drop(columns=[context_target])
    y = df[context_target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = Path(workspace) / model_path  # pyright: ignore
    # save_model_to_file(model_path, out)

    # Step 2: Instantiate and train a logistic regression model
    tc.print("Loading the model from file...")
    model = load_model_from_file(model_path)

    # Step 3: Initialize AzureOpenAI client (replace with your actual configuration)
    client = create_llm_client(session=session, additional_kwargs_required=additional_kwargs_required)
    pattern = r"openai/?$"
    base_url = re.sub(pattern, "", str(client._base_url))  # noqa: F841

    config_dict = {
        "api_type": "azure" if session.engine_name in ("azure_openai_v1",) else "openai_v1",
        "api_base": str(client._base_url),
        "api_version": client._api_version,
        "api_key": client.api_key,
        "engine": additional_kwargs_required["azure_openai_config"].deployment_name,
        "deployment_id": additional_kwargs_required["azure_openai_config"].deployment_name,
        "temperature": additional_kwargs_required["engine_params"]["temperature"],
        "seed": 0,
    }

    # Step 4: Create SMART instance
    subgroup_finder = SMART(llm=client, config=config_dict, verbose=False)

    subgroup_finder.fit(X_train, context=context, context_target=context_target, evaluate_feasibility=False)

    # Step 5: Display the identified subgroups
    tc.print("Identified Subgroups:")
    tc.print(subgroup_finder.subgroups)

    tc.print("Hypotheses generated about each subgroup")
    tc.print(subgroup_finder.hypotheses)

    recommendations = subgroup_finder.generate_model_report(X_train, y_train, X_test, y_test, model)

    tc.set_returns(
        tool_return=(recommendations),  # pyright: ignore
    )


class SmartTesting(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_path"])
        thrd, out_stream = execute_tool(
            smart_testing,
            wd=self.working_directory,
            data_path=real_path,
            model_path=kwargs["model_path"],
            context=kwargs["context"],
            context_target=kwargs["context_target"],
            workspace=self.working_directory,
            session=kwargs["session"],
            additional_kwargs_required=kwargs["additional_kwargs_required"],
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "smart_testing"

    @property
    def description(self) -> str:
        return """
Uses the smart_testing tool to find subgroups of the dataset that the model may perform poorly on.
The tool provides a descriptive summary of the subgroups.
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
                        "data_path": {"type": "string", "description": "Path to the data file."},
                        "model_path": {"type": "string", "description": "Path to the saved autoprognosis model."},
                        "context": {
                            "type": "string",
                            "description": "A description of the dataset, including the target column and features, in plain english.",
                        },
                        "context_target": {"type": "string", "description": "Name of the target column."},
                    },
                    "required": [
                        "data_path",
                        "model_path",
                        "context",
                        "context_target",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return """
Uses the smart_testing tool to find subgroups of the dataset that the model may perform poorly on.
The tool provides a descriptive summary of the subgroups.
"""
