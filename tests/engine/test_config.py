import os
import pathlib

import pytest

from climb.common.exc import ClimbConfigurationError
from climb.engine._config import get_dotenv_config

DOTENV = """
OPENAI_API_KEY="foo"
AZURE_OPENAI_API_KEY__endpoint1="bar"
AZURE_OPENAI_API_KEY__endpoint2="baz"
"""


@pytest.fixture
def test_dotenv(tmp_path: pathlib.Path) -> pathlib.Path:
    test_file = tmp_path / ".env"
    test_file.write_text(DOTENV)
    return test_file


@pytest.fixture
def test_keysdotenv(tmp_path: pathlib.Path) -> pathlib.Path:
    test_file = tmp_path / "keys.env"
    test_file.write_text(DOTENV)
    return test_file


def test_get_dotenv_config_explicit_try_dotenv_paths(tmp_path, test_dotenv) -> None:
    # Set working directory to the directory containing the test .env file
    os.chdir(tmp_path)
    config = get_dotenv_config([".env"])
    assert config == {
        "OPENAI_API_KEY": "foo",
        "AZURE_OPENAI_API_KEY__endpoint1": "bar",
        "AZURE_OPENAI_API_KEY__endpoint2": "baz",
    }


def test_get_dotenv_config_default_try_dotenv_paths_dotenv(tmp_path, test_dotenv) -> None:
    # Set working directory to the directory containing the test .env file
    os.chdir(tmp_path)
    config = get_dotenv_config()
    assert config == {
        "OPENAI_API_KEY": "foo",
        "AZURE_OPENAI_API_KEY__endpoint1": "bar",
        "AZURE_OPENAI_API_KEY__endpoint2": "baz",
    }


def test_get_dotenv_config_default_try_dotenv_paths_keysdotenv(tmp_path, test_keysdotenv) -> None:
    # Set working directory to the directory containing the test .env file
    os.chdir(tmp_path)
    config = get_dotenv_config()
    assert config == {
        "OPENAI_API_KEY": "foo",
        "AZURE_OPENAI_API_KEY__endpoint1": "bar",
        "AZURE_OPENAI_API_KEY__endpoint2": "baz",
    }


def test_get_dotenv_config_no_dotenv_files(tmp_path) -> None:
    # Set working directory to the directory containing the test .env file
    os.chdir(tmp_path)
    with pytest.raises(ClimbConfigurationError, match=r".*\.env.*not found.*"):
        get_dotenv_config(["nonexistent.env"])
