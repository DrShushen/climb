import os
import pathlib
import subprocess
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from climb.common.exc import ClimbConfigurationError
from climb.engine._code_execution import execute_code, get_conda_env_libraries


@pytest.fixture
def mock_popen(mocker: MockerFixture) -> MagicMock:
    mock_process = MagicMock()
    mock_process.stdout = ("mocked stdout",)
    mock_process.returncode = 0

    # Make the mock work with context manager:
    mock_process.__enter__.return_value = mock_process
    mock_process.__exit__.return_value = None

    # Create the Popen mock that returns our process mock.
    return mocker.patch("subprocess.Popen", return_value=mock_process)


def test_run_command(mock_popen: MagicMock, mocker: MockerFixture, tmp_path: pathlib.Path):
    # Mock the get_conda_env_libraries function, as it uses a subprocess conda call which we don't want to run.
    mocker.patch("climb.engine._code_execution.get_conda_env_libraries", return_value=[])

    working_directory = tmp_path / "working_directory"
    os.makedirs(working_directory, exist_ok=True)

    code_execution_stream = list(  # Call list() to consume the generator.
        execute_code(
            code_file_name="code_file_name.py",
            working_directory=tmp_path / "working_directory",
            generated_code="# Nothing, it's mocked",
            dependencies=[],
            conda_path=None,
        )
    )

    # Verify the expected commands were run.
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "climb-code",
        "python",
        "-u",
        os.path.join(working_directory, "code_file_name.py"),
    ]
    mock_popen.assert_called_once_with(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        cwd=str(working_directory),
        text=True,
        universal_newlines=True,
    )

    assert "mocked stdout" in code_execution_stream


def test_get_conda_env_libraries_error_no_conda_command(mocker: MockerFixture):
    nonexistent_command = "definitely-not-conda-aoiedhgaopqoi"
    with pytest.raises(ClimbConfigurationError, match=r".*definitely-not-conda.*"):
        get_conda_env_libraries(conda_env_name="dummy-env", conda_path=nonexistent_command)
