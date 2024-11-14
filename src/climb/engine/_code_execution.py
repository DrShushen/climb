import ast
import os
import re
import subprocess
from typing import Iterable, List, Literal, Optional, Tuple, Union

import pydantic

from climb.common.exc import EXC_DOCS_REFS, ClimbConfigurationError
from climb.common.utils import make_filename_path_safe

CodeExecStatus = Literal["success", "error"]


class CodeExecFinishedSentinel(pydantic.BaseModel):
    status: CodeExecStatus
    error_message: Optional[str] = None


CodeExecReturn = Iterable[Union[str, CodeExecFinishedSentinel]]

# TODO: Hacky, make this better.
_NO_DEPENDENCIES_STRINGS = ("", "none", "no", "nothing", "no dependencies")


def is_code_generated(input_text: str) -> bool:
    if "DEPENDENCIES:" in input_text and "CODE:" in input_text:
        return True
    return False


def code_extract(input_text: str) -> Tuple[List[str], str, List[str], List[str]]:
    def is_ast_valid(code: str) -> None:
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code (fails `ast.parse`): {str(e)}") from e

    # Regular expressions to match dependencies and code blocks
    dependencies_pattern = r"DEPENDENCIES:\n```(.*?)```"
    code_pattern = r"CODE:\n```python\n(.*?)```"

    files_in_pattern = r"FILES_IN:\n```(.*?)```"
    files_out_pattern = r"FILES_OUT:\n```(.*?)```"

    # Search for dependencies and code using the patterns
    dependencies_match = re.search(dependencies_pattern, input_text, re.DOTALL)
    code_match = re.search(code_pattern, input_text, re.DOTALL)

    files_in_match = re.search(files_in_pattern, input_text, re.DOTALL)
    files_out_match = re.search(files_out_pattern, input_text, re.DOTALL)

    # Raise ValueError if incorrect number of matches are found.
    if dependencies_match is None:
        raise ValueError("No DEPENDENCIES block found in input text.")
    if code_match is None:
        raise ValueError("No CODE block found in input text.")
    if files_in_match is None:
        raise ValueError("No FILES_IN block found in input text.")
    if files_out_match is None:
        raise ValueError("No FILES_OUT block found in input text.")
    if len(dependencies_match.groups()) > 1:
        raise ValueError("More than one DEPENDENCIES block found in input text.")
    if len(code_match.groups()) > 1:
        raise ValueError("More than one CODE block found in input text.")
    if len(files_in_match.groups()) > 1:
        raise ValueError("More than one FILES_IN block found in input text.")
    if len(files_out_match.groups()) > 1:
        raise ValueError("More than one FILES_OUT block found in input text.")

    # Extract dependencies and code if matches are found
    dependencies = dependencies_match.group(1).strip().split("\n") if dependencies_match else []
    dependencies = [d.replace("pip install", "").strip() for d in dependencies]
    dependencies = [d for d in dependencies if d]  # Remove empty strings
    # Validate dependencies:
    for dep in dependencies:
        # Check that it is alphanumeric and underscores or hyphens only.
        if not re.match(r"^[a-zA-Z0-9_-]+$", dep):
            raise ValueError(f"Invalid pip dependency name: {dep}")
    if len(dependencies) == 1 and dependencies[0].lower().strip() in _NO_DEPENDENCIES_STRINGS:
        dependencies = []
    code = code_match.group(1).strip() if code_match else ""
    # Validate code:
    if code == "":
        raise ValueError("Code parsed was an empty string, check code formatting.")
    is_ast_valid(code)

    files_in = files_in_match.group(1).strip().split("\n") if files_in_match else []
    files_in = [f.strip() for f in files_in]
    files_in = [f for f in files_in if f]

    files_out = files_out_match.group(1).strip().split("\n") if files_out_match else []
    files_out = [f.strip() for f in files_out]
    files_out = [f for f in files_out if f]

    # print("Dependencies:", dependencies)
    # print("Code:", code)

    return dependencies, code, files_in, files_out


def get_conda_env_libraries(
    conda_env_name: str,
    conda_path: str,
) -> List[str]:
    try:
        # Running the `conda list` command for the specified environment
        try:
            result = subprocess.run(
                [conda_path, "list", "--name", conda_env_name], capture_output=True, text=True, check=True
            )
        except FileNotFoundError as e:
            raise ClimbConfigurationError(
                f"Could not execute `conda` using the command {conda_path}. "
                "Make sure that the `conda` command is available. "
                "Please refer to the following documentation section for troubleshooting:\n"
                f"{EXC_DOCS_REFS['troubleshooting_conda_not_founc']}"
            ) from e

        # Parsing the output to extract package names
        packages = result.stdout.split("\n")
        library_names = []
        for package in packages:
            if package and not package.startswith("#"):
                library_name = package.split()[0]
                library_names.append(library_name)
        return library_names
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return []


def prepare_code_file_name(code_file_name: str) -> str:
    # Take off the .py extension if it exists:
    if code_file_name.endswith(".py"):
        code_file_name = code_file_name[:-3]

    code_file_name = make_filename_path_safe(code_file_name)

    # Add back .py:
    code_file_name = f"{code_file_name}.py"

    return code_file_name


def execute_code(
    code_file_name: str,
    working_directory: str,
    generated_code: str,
    dependencies: List[str],
    conda_path: Optional[str] = None,
) -> CodeExecReturn:
    conda_path = conda_path or "conda"

    code_file = prepare_code_file_name(code_file_name)
    code_file_path = os.path.realpath(os.path.join(working_directory, code_file))
    with open(code_file_path, "w", encoding="utf8") as f:
        f.write(generated_code)

    # TODO: Better management of environments.
    # TODO: Deleting code file at the end?

    env_name = "climb-code"

    needed_dependencies = set(dependencies)
    env_dependencies = set(
        get_conda_env_libraries(
            conda_env_name=env_name,
            conda_path=conda_path,
        )
    )
    missing_dependencies = needed_dependencies - env_dependencies

    commands = []
    if missing_dependencies:
        yield f"Missing dependencies will be installed: {missing_dependencies}\nPlease wait..."
        command_dependencies = (
            f"{conda_path} run --no-capture-output -n {env_name} pip install {' '.join(missing_dependencies)}".split(
                " "
            )
        )
        commands.append(command_dependencies)
    command_main = f"{conda_path} run --no-capture-output -n {env_name} python -u {code_file_path}".split(" ")
    commands.append(command_main)

    # Execute the code
    for command in commands:
        stdout = ""
        error_msg = ""
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            universal_newlines=True,
            cwd=os.path.realpath(working_directory),
        ) as proc:
            if proc.stdout is not None:
                # Yield each line of output
                for line in proc.stdout:
                    stdout += line
                    yield line
            # Wait for the subprocess to finish and check if there was an error.
            proc.wait()
            if proc.returncode == 0:
                yield CodeExecFinishedSentinel(status="success")
            if proc.returncode != 0 and proc.stderr is not None:
                # Read the error message from stderr
                error_msg = proc.stderr.read().strip()
                yield CodeExecFinishedSentinel(status="error", error_message=error_msg)
                break
