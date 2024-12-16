from difflib import SequenceMatcher

from utils import get_tool_output

from climb.tool.impl.tool_exploratory_data_analysis import exploratory_data_analysis
from climb.tool.tool_comms import ToolCommunicator


def test_exploratory_data_analysis(df_eda_path, eda_output):
    """High level test to make sure that the output from EDA function remains consistent."""

    mock_tc = ToolCommunicator()

    # Execute function with mock_tc
    exploratory_data_analysis(mock_tc, df_eda_path, "", ".")

    tool_return = get_tool_output(mock_tc).tool_return

    # The tool output is a string. Here, we assert that it's not too different from the expected output
    assert SequenceMatcher(None, tool_return, eda_output).ratio() > 0.9
