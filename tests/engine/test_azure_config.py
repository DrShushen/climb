import pathlib

import pytest

from climb.common.exc import ClimbConfigurationError
from climb.engine._azure_config import (
    AzureOpenAIConfig,
    get_api_key_for_azure_openai,
    load_azure_openai_config_item,
    load_azure_openai_configs,
)

TEST_AZURE_OPENAI_CONFIG = """
models:
  - name: "model-A"
    endpoint: "https://testlocation1.openai.azure.com/"
    deployment_name: "test-deployment-A"
    api_version: "2024-02-01"
    model: "gpt-4-0125-preview"
  - name: "model-B"
    endpoint: "https://testlocation2.openai.azure.com/"
    deployment_name: "test-deployment-B"
    api_version: "2024-02-01"
    model: "gpt-4o-2024-05-13"
"""

TEST_AZURE_OPENAI_WRONG_FORMAT = """
models:
  - unknown: "model-A"
    endpoint: "https://testlocation1.openai.azure.com/"
    deployment_name: "test-deployment-A"
    api_version: "2024-02-01"
    model: "gpt-4-0125-preview"
"""


@pytest.fixture
def test_azure_openai_config(tmp_path: pathlib.Path) -> pathlib.Path:
    test_file = tmp_path / "az_openai_config.txt"
    test_file.write_text(TEST_AZURE_OPENAI_CONFIG)
    return test_file


@pytest.fixture
def test_azure_openai_wrong_format(tmp_path: pathlib.Path) -> pathlib.Path:
    test_file = tmp_path / "az_openai_config.txt"
    test_file.write_text(TEST_AZURE_OPENAI_WRONG_FORMAT)
    return test_file


def test_load_load_azure_openai_configs_not_found():
    with pytest.warns(UserWarning, match=r".*Azure OpenAI.*not found.*"):
        out = load_azure_openai_configs(config_path="./nonexistent.yml")
    assert out == []


def test_load_load_azure_openai_configs_loaded(test_azure_openai_config: pathlib.Path):
    out = load_azure_openai_configs(config_path=test_azure_openai_config)
    assert len(out) == 2
    assert out[0].name == "model-A"
    assert out[1].name == "model-B"
    assert isinstance(out[0], AzureOpenAIConfig)
    assert isinstance(out[1], AzureOpenAIConfig)


def test_load_load_azure_openai_configs_wrong_format(test_azure_openai_wrong_format: pathlib.Path):
    with pytest.raises(ClimbConfigurationError, match=r".*parsing.*"):
        load_azure_openai_configs(config_path=test_azure_openai_wrong_format)


def test_load_azure_openai_config_item_not_found(test_azure_openai_config: pathlib.Path):
    with pytest.raises(ClimbConfigurationError, match=r".*not found.*"):
        load_azure_openai_config_item(config_path=test_azure_openai_config, config_item_name="model-C")


def test_load_azure_openai_config_item_found(test_azure_openai_config: pathlib.Path):
    out = load_azure_openai_config_item(config_path=test_azure_openai_config, config_item_name="model-B")
    assert out.name == "model-B"
    assert out.model == "gpt-4o-2024-05-13"
    assert isinstance(out, AzureOpenAIConfig)


def test_get_api_key_for_azure_openai_found():
    az_config = AzureOpenAIConfig(
        name="model-A",
        endpoint="https://testlocation1.openai.azure.com/",
        deployment_name="test-deployment-A",
        api_version="2024-02-01",
        model="gpt-4-0125-preview",
    )
    dotenv = {"AZURE_OPENAI_API_KEY__testlocation1": "foo"}
    out = get_api_key_for_azure_openai(az_config, dotenv)
    assert out == "foo"


def test_get_api_key_for_azure_openai_not_found():
    az_config = AzureOpenAIConfig(
        name="model-A",
        endpoint="https://testlocation1.openai.azure.com/",
        deployment_name="test-deployment-A",
        api_version="2024-02-01",
        model="gpt-4-0125-preview",
    )
    dotenv = {}
    with pytest.raises(ClimbConfigurationError, match=r".*API key.*not found.*"):
        get_api_key_for_azure_openai(az_config, dotenv)
