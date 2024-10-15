from typing import Any, Dict, List, Optional

from .impl.tool_autoprognosis import (
    AutoprognosisClassification,
    AutoprognosisRegression,
    AutoprognosisSubgroupEvaluation,
    AutoprognosisSurvival,
)
from .impl.tool_data_centric import DataIQInsights  # ConfidentLearningInsights
from .impl.tool_descriptive_stats import DescriptiveStatistics
from .impl.tool_exploratory_data_analysis import ExploratoryDataAnalysis
from .impl.tool_feature_importance import PermutationExplainer, ShapExplainer
from .impl.tool_feature_selection import BorutaFeatureSelection
from .impl.tool_hardware import HardwareInfo
from .impl.tool_imputation import HyperImputeImputation
from .impl.tool_paper import UploadAndSummarizeExamplePaper
from .impl.tool_upload import UploadDataFile
from .tool_comms import ToolCommunicator, ToolOutput, ToolReturnIter
from .tools import ToolBase, UserInputRequest


def get_tool(tool_name: str) -> ToolBase:
    return AVAILABLE_TOOLS[tool_name]


def list_all_tool_specs(filter_tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tool_specs = [
        tool.specification
        for tool in _AVAILABLE_TOOLS_LIST
        if (filter_tool_names is None or tool.name in filter_tool_names)
    ]

    # Validate tool specs:
    for tool in tool_specs:
        if "function" not in tool:
            raise ValueError(f"Tool spec must have a 'function' key: {tool}")
        func = tool["function"]
        if "name" not in func:
            raise ValueError(f"Tool spec's function must have a 'name' key: {tool}")
        if "description" not in func:
            raise ValueError(f"Tool spec's function must have a 'description' key: {tool}")
        if len(func["description"]) > 1027:
            raise ValueError(f"Tool spec's function description must be 1027 characters or less: {tool}")
        # NOTE: Not meant to be extensive validation.

    return tool_specs


def list_all_tool_names(filter_tool_names: Optional[List[str]] = None) -> List[str]:
    names = [
        tool.name for tool in _AVAILABLE_TOOLS_LIST if (filter_tool_names is None or tool.name in filter_tool_names)
    ]
    return names


_AVAILABLE_TOOLS_LIST = [
    AutoprognosisClassification(),
    AutoprognosisRegression(),
    AutoprognosisSubgroupEvaluation(),
    AutoprognosisSurvival(),
    BorutaFeatureSelection(),
    # ConfidentLearningInsights(),
    DataIQInsights(),
    DescriptiveStatistics(),
    ExploratoryDataAnalysis(),
    HardwareInfo(),
    HyperImputeImputation(),
    PermutationExplainer(),
    ShapExplainer(),
    UploadAndSummarizeExamplePaper(),
    UploadDataFile(),
]
AVAILABLE_TOOLS = {tool.name: tool for tool in _AVAILABLE_TOOLS_LIST}

__all__ = [
    "get_tool",
    "HardwareInfo",
    "HyperImputeImputation",
    "list_all_tool_specs",
    "ToolBase",
    "ToolCommunicator",
    "ToolOutput",
    "ToolReturnIter",
    "UserInputRequest",
]
