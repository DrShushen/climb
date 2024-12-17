import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest


@pytest.fixture
def tool_return():
    """ToolOutput.tool_return should be str"""
    return "Foobar"


@pytest.fixture
def user_report_outputs():
    """ToolOutput.user_report_outputs could be a str, matplotlib graph, or plotly graph"""
    return ["Foobar", plt.Figure(), go.Figure()]
