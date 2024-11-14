from pathlib import Path
from typing import Dict, List, Union

from dotenv import dotenv_values

from climb.common.exc import ClimbConfigurationError

TRY_DOTENV_PATHS = [
    ".env",
    "keys.env",
]


def get_dotenv_config(try_dotenv_files: List[str] = TRY_DOTENV_PATHS) -> Dict[str, Union[str, None]]:
    dotenv_found = False
    dotenv_found_error_msg = ""

    for path in try_dotenv_files:
        if not Path(path).exists():
            dotenv_found_error_msg += f"`.env` file not found: {Path(path).absolute()}.\n"
        else:
            # print(f"Using `.env` file: {Path(path).absolute()}.")
            dotenv_config = dotenv_values(path)
            dotenv_found = True
            break
    if not dotenv_found:
        raise ClimbConfigurationError(dotenv_found_error_msg + "No more acceptable .env paths to try.")

    return dotenv_config
