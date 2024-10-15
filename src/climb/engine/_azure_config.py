import os
from typing import Dict, List

import pydantic
import yaml

from .const import ALLOWED_MODELS

AZURE_OPENAI_CONFIG_PATH = "az_openai_config.yml"  # Relative to app launch directory.


class AzureOpenAIConfig(pydantic.BaseModel):
    name: str
    endpoint: str
    deployment_name: str
    api_version: str
    model: str

    @pydantic.field_validator("model")
    @classmethod
    # pylint: disable-next=unused-argument
    def check_allowed_model(cls, v: str, info: pydantic.ValidationInfo) -> str:
        assert v in ALLOWED_MODELS, f"Model must be one of: {ALLOWED_MODELS}"
        return v


def load_azure_openai_configs(config_path: str) -> List[AzureOpenAIConfig]:
    if not os.path.exists(config_path):
        return []
    # pylint: disable-next=unspecified-encoding
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    azure_openai_config = []
    for item in yaml_config["models"]:
        azure_openai_config.append(AzureOpenAIConfig(**item))
    return azure_openai_config


def load_azure_openai_config_item(config_path: str, config_item_name: str) -> AzureOpenAIConfig:
    configs = load_azure_openai_configs(config_path)
    try:
        az_config = [x for x in configs if x.name == config_item_name][0]
    except IndexError as e:
        raise ValueError(f"Azure OpenAI config with name '{config_item_name}' not found in file {config_path}") from e
    return az_config


def get_api_key_for_azure_openai(azure_openai_config: AzureOpenAIConfig, dotenv: Dict) -> str:
    endpoint_id = (
        azure_openai_config.endpoint.replace("https://", "").replace("http://", "").replace("/", "").split(".")[0]
    )
    try:
        api_key = dotenv[f"AZURE_OPENAI_API_KEY__{endpoint_id}"]
    except KeyError as e:
        raise ValueError(
            f"API key not found in the .env file for Azure OpenAI endpoint: {azure_openai_config.endpoint}. "
            f"Check that an entry AZURE_OPENAI_API_KEY__{endpoint_id} exists in your .env file."
        ) from e
    return api_key
