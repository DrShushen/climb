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
from .engine_openai_v1 import AzureOpenAIV1Engine, OpenV1Engine

dotenv_config = get_dotenv_config()


ENGINE_MAP = {
    # v1 engines:
    OpenV1Engine.get_engine_name(): OpenV1Engine,
    AzureOpenAIV1Engine.get_engine_name(): AzureOpenAIV1Engine,
}

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
