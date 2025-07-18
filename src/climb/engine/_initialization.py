from typing import Any, Dict

from climb.common.data_structures import Session
from climb.common.utils import fix_windows_path_backslashes
from climb.db import DB

from ._azure_config import (
    AZURE_OPENAI_CONFIG_PATH,
    get_api_key_for_azure_openai,
    load_azure_openai_config_item,
)
from ._config import get_dotenv_config
from ._engine import EngineBase
from .engine_openai_dc import AzureOpenAIDCEngine, OpenAIDCEngine
from .engine_openai_min_baseline import AzureOpenAIMinBaselineEngine, OpenAIMinBaselineEngine
from .engine_openai_nextgen import AzureOpenAINextGenEngine, OpenAINextGenEngine
from .engine_openai_sim import AzureOpenAINextGenEngineSim, OpenAINextGenEngineSim
from .engine_openai_tool_baseline import AzureOpenAIToolBaselineEngine, OpenAIToolBaselineEngine

dotenv_config = get_dotenv_config()


ENGINE_MAP = {
    # Current engines:
    OpenAINextGenEngine.get_engine_name(): OpenAINextGenEngine,
    AzureOpenAINextGenEngine.get_engine_name(): AzureOpenAINextGenEngine,
    # CoT engines:
    OpenAIDCEngine.get_engine_name(): OpenAIDCEngine,
    AzureOpenAIDCEngine.get_engine_name(): AzureOpenAIDCEngine,
}

if dotenv_config.get("BASELINE_METHODS", "False") == "True":
    ENGINE_MAP.update(
        {
            # Minimal baseline versions:
            OpenAIMinBaselineEngine.get_engine_name(): OpenAIMinBaselineEngine,
            AzureOpenAIMinBaselineEngine.get_engine_name(): AzureOpenAIMinBaselineEngine,
            # Tool baseline versions:
            OpenAIToolBaselineEngine.get_engine_name(): OpenAIToolBaselineEngine,
            AzureOpenAIToolBaselineEngine.get_engine_name(): AzureOpenAIToolBaselineEngine,
        }
    )

if dotenv_config.get("SIMULATED_USER_METHODS", "False") == "True":
    ENGINE_MAP.update(
        {
            # Minimal baseline versions:
            OpenAINextGenEngineSim.get_engine_name(): OpenAINextGenEngineSim,
            AzureOpenAINextGenEngineSim.get_engine_name(): AzureOpenAINextGenEngineSim,
        }
    )

azure_engines = [engine_name for engine_name in ENGINE_MAP.keys() if "azure" in engine_name]
non_azure_engines = [engine_name for engine_name in ENGINE_MAP.keys() if "azure" not in engine_name]


def create_engine(db: DB, session: Session, config: Dict[str, Any]) -> EngineBase:
    EngineClass = ENGINE_MAP[session.engine_name]
    conda_path = config.get("CONDA_PATH", None)
    if conda_path is not None:
        conda_path = fix_windows_path_backslashes(conda_path)
    extra_kwargs: Dict[str, Any] = {
        "conda_path": conda_path,
    }
    if EngineClass.get_engine_name() in non_azure_engines:
        extra_kwargs["api_key"] = config["OPENAI_API_KEY"]
    elif EngineClass.get_engine_name() in azure_engines:
        extra_kwargs["azure_openai_config"] = load_azure_openai_config_item(
            AZURE_OPENAI_CONFIG_PATH,
            session.engine_params["config_item_name"],  # pyright: ignore
        )
        extra_kwargs["api_key"] = get_api_key_for_azure_openai(extra_kwargs["azure_openai_config"], config)
    return EngineClass(
        db=db,
        session=session,
        **extra_kwargs,
    )
