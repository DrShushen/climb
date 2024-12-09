import pytest

from climb.tool.tool_comms import ToolCommunicator, ToolOutput

BACKUP_OUTPUT_FILE = "progress.txt"


# Tests for ToolOutput
def test_tool_output_set(tool_return, user_report_outputs):
    """ToolOutput uses property with setter. This tests that
    the setter works with specific input types"""
    output = ToolOutput()

    output.tool_return = tool_return
    assert output.tool_return == tool_return

    output.user_report_outputs = user_report_outputs
    assert output.user_report_outputs == user_report_outputs


# Tests for ToolCommunicator
def test_tool_communicator_print_before_return_set():
    """Test that .print() adds messages to comm_queue"""
    tc = ToolCommunicator()
    test_message = "Foobar"
    tc.print(test_message)
    # Retrieve the message from the queue
    output = tc.comm_queue.get_nowait()
    assert output == f"{test_message}\n"


def test_tool_communicator_print_after_return_set():
    """Test that .print() raises ValueError after return is set."""
    tc = ToolCommunicator()
    tc.return_set = True
    with pytest.raises(ValueError) as exc_info:
        tc.print("This should raise an exception")
    assert "Cannot print after return value has been set" in str(exc_info.value)


def test_tool_communicator_set_returns():
    """Test that set_returns adds a ToolOutput to the comm_queue and sets return_set to True."""
    tc = ToolCommunicator()
    tool_return = "Result"
    user_report = ["Report line 1", "Report line 2"]
    files_in = ["input.txt"]
    files_out = ["output.txt"]
    tc.set_returns(tool_return, user_report, files_in, files_out)
    # Retrieve the ToolOutput from the queue
    output = tc.comm_queue.get_nowait()
    assert isinstance(output, ToolOutput)
    assert output.tool_return == tool_return
    assert output.user_report_outputs == user_report
    assert output.files_in == files_in
    assert output.files_out == files_out
    assert tc.return_set is True


# TODO: Tests for ToolThreads
