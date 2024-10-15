"""Module for estimating OpenAI tokens count in API calls.

Why is this needed?
- When `stream=True`, Azure OpenAI API does not return the number of tokens used.
- Note that *actual* OpenAI API *does* return the number of tokens used in the response, you just need to set \
`stream_options={"include_usage": True}`. So with actual OpenAI API, you don't need this estimation.
- Once Azure OpenAI API starts returning the number of tokens used, this can be replaced by the real values.
See:
- [When Azure will sync OpenAI ChatCompletion stream_options feature?](https://github.com/Azure/azure-rest-api-specs/issues/29157)

Other relevant links:
- https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573
- https://github.com/rubenselander/openai-function-tokens
"""

from typing import Any, Dict, List, Optional, Union

import tiktoken

from .const import ALLOWED_MODELS

# region: === OpenAI token estimation with functions (tools) ===


def _format_function_definitions(functions: List[Dict]) -> str:
    """
    Generates TypeScript function type definitions.

    Args:
        functions (List[Dict]): List of dictionaries representing function definitions.

    Returns:
        str: TypeScript function type definitions.
    """
    lines = ["namespace functions {"]

    for func in functions:
        if func.get("description"):
            lines.append(f"// {func['description']}")

        if func["parameters"].get("properties"):
            lines.append(f"type {func['name']} = (_: {{")
            lines.append(_format_object_properties(func["parameters"], 0))
            lines.append("}) => any;")
        else:
            lines.append(f"type {func['name']} = () => any;")

        lines.append("")

    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_properties(parameters: Dict, indent: int) -> str:
    """
    Formats object properties for TypeScript type definitions.

    Args:
        parameters (Dict): Dictionary representing object parameters.
        indent (int): Number of spaces for indentation.

    Returns:
        str: Formatted object properties.
    """
    lines = []
    for name, param in parameters["properties"].items():
        if param.get("description") and indent < 2:
            lines.append(f"// {param['description']}")

        is_required = parameters.get("required") and name in parameters["required"]
        lines.append(f"{name}{'?:' if not is_required else ':'} {_format_type(param, indent)},")

    return "\n".join([" " * indent + line for line in lines])


def _format_type(param: Dict, indent: int) -> str:
    """
    Formats a single property type for TypeScript type definitions.

    Args:
        param (Dict): Dictionary representing a parameter.
        indent (int): Number of spaces for indentation.

    Returns:
        str: Formatted type for the given parameter.
    """
    type_ = param["type"]
    if type_ == "string":
        return " | ".join([f'"{v}"' for v in param["enum"]]) if param.get("enum") else "string"
    elif type_ == "number":
        return " | ".join([str(v) for v in param["enum"]]) if param.get("enum") else "number"
    elif type_ == "integer":
        return " | ".join([str(v) for v in param["enum"]]) if param.get("enum") else "integer"
    elif type_ == "array":
        return f"{_format_type(param['items'], indent)}[]" if param.get("items") else "any[]"
    elif type_ == "boolean":
        return "boolean"
    elif type_ == "null":
        return "null"
    elif type_ == "object":
        return "{\n" + _format_object_properties(param, indent + 2) + "\n}"
    else:
        raise ValueError(f"Unsupported type: {type_}")


def _estimate_function_tokens(functions: List[Dict], encoding_name: str) -> int:
    """
    Estimates token count for a given list of functions.

    Args:
        functions (List[Dict]): List of dictionaries representing function definitions.

    Returns:
        int: Estimated token count.
    """
    prompt_definitions = _format_function_definitions(functions)
    tokens = _string_tokens(prompt_definitions, encoding_name)
    tokens += 9  # Add nine per completion
    return tokens


def _string_tokens(string: str, encoding_name: str) -> int:
    """
    Estimates token count for a given string using 'cl100k_base' encoding.

    Args:
        string (str): Input string.
        encoding_name (str): Encoding name, called as `tiktoken.get_encoding(encoding_name)`.

    Returns:
        int: Estimated token count.
    """
    return len(tiktoken.get_encoding(encoding_name).encode(string))


def _estimate_message_tokens(message: Dict, encoding_name: str) -> int:
    """
    Estimates token count for a given message.

    Args:
        message (Dict): Dictionary representing a message.
        encoding_name (str): Encoding name.

    Returns:
        int: Estimated token count.
    """
    components = [
        message.get("role"),
        message.get("content"),
        message.get("name"),
        message.get("function_call", {}).get("name"),
        message.get("function_call", {}).get("arguments"),
    ]
    components = [component for component in components if component]  # Filter out None values
    tokens = sum([_string_tokens(component, encoding_name) for component in components])

    tokens += 3  # Add three per message
    if message.get("name"):
        tokens += 1
    if message.get("role") == "function":
        tokens -= 2
    if message.get("function_call"):
        tokens += 3

    return tokens


def _estimate_tokens_with_functions(
    messages: List[Dict],
    encoding_name: str,
    tokens_per_message: int,
    tokens_per_name: int,
    functions: Optional[List[Dict]] = None,
    function_call: Union[str, Dict, None] = None,
) -> int:
    """
    Estimates token count for a given prompt with messages and functions.

    Notes:
    - This is adapted from:
        - https://github.com/rubenselander/openai-function-tokens
    - The whole discussion pertaining to this is here:
        - https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573
    - I do not fully understand the `function_call` argument, so it is ignored in our use case.

    Args:
        messages (list[Dict]): List of dictionaries representing messages.
        encoding_name (str): Encoding name.
        tokens_per_message (int): Number of tokens per message.
        tokens_per_name (int): Number of tokens per name.
        functions (list[Dict], optional): List of dictionaries representing function definitions. Default is None.
        function_call (str or dict, optional): Function call specification. Default is None.

    Returns:
        int: Estimated token count.
    """
    padded_system = False
    tokens = 0

    FUNCTION_CALL_NOT_NONE_NAME_PLUS_TOKENS = 4

    for msg in messages:
        if msg["role"] == "system" and functions and not padded_system:
            modified_message = {"role": msg["role"], "content": msg["content"] + "\n"}
            tokens += _estimate_message_tokens(modified_message, encoding_name)
            padded_system = True  # Mark system as padded
        else:
            tokens += _estimate_message_tokens(msg, encoding_name)

    tokens += tokens_per_message  # Each completion has a 3-token overhead
    if functions:
        tokens += _estimate_function_tokens(functions, encoding_name)

    if functions and any(m["role"] == "system" for m in messages):
        tokens -= 4  # Adjust for function definitions

    if function_call and function_call != "auto":
        tokens += (
            tokens_per_name
            if function_call == "none"
            else _string_tokens(function_call["name"]) + FUNCTION_CALL_NOT_NONE_NAME_PLUS_TOKENS  # type: ignore
        )

    return tokens


# endregion


def estimate_prompt_tokens_with_tools(
    messages_in: List[Dict[str, Any]],
    model: str,
    function_definitions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Return the number of tokens used by a list of messages and function (tool) definitions.

    **Important:** This is only an estimate and is not going to be exactly accurate! It appears to be especially
    inaccurate when there are more than on tool call requests in the message history. It seems reasonably accurate
    for normal messages, and the estimation of the tool definitions themselves are also pretty good.

    Notes:
    Input messages should be like:
    ```
    messages_in = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},

        # **Note:** If a tool call historic message, must be like this:
        {
            'content': '',
            'role': 'assistant',
            'tool_calls': [
                {
                    'function': {
                        'arguments': '<arguments_dict>',
                        'name': '<function_name>',
                    },
                    'id': '<SOME_ID>',
                    'type': 'function'
                },
                ...
            ]
        },
    ]
    ```

    Function (tool) definitions should be like:
    - **Note:** Each item is the stuff under "function" key in the full definition of a tool.
    ```
    function_definitions = [
    {
        {
            "name": "...",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "...",
                        "description": "...",
                    },
                    ...
                },
                "required": [...],
                "additionalProperties": False,
            },
        },
        ...
    ]
    ```
    """
    # Various best guesses for number of tokens used for various padding internally by OpenAI.
    FUNCTION_CALL_NOT_NONE_NAME_PLUS_TOKENS = 4
    FUNCTION_CALL_ADJUSTMENT_TOKENS = 7
    PRIMER_TOKENS = 3

    messages = messages_in
    add_function_call_request_extra_tokens = False

    # Validate model.
    if model in ALLOWED_MODELS:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")

    # Get encoding for model.
    encoding = tiktoken.encoding_for_model(model)
    # print(f"Using encoding {encoding.name} for model {model}.")

    if not function_definitions:
        # CASE: Normal token counting loop for messages, in case no functions are present.
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # The supported keys are below, else skip.
                if key not in ("name", "role", "content", "tool_calls", "tool"):
                    continue

                # Count tokens of tool_calls (tool call requests from the LLM).
                if key == "tool_calls":
                    add_function_call_request_extra_tokens = True
                    function_call_requests_message_like = []
                    for function_call_request in value:
                        function_call_requests_message_like.append(
                            {
                                "content": function_call_request["function"]["name"]
                                + (
                                    function_call_request["function"]["arguments"]
                                    if function_call_request["function"]["arguments"] is not None
                                    else ""
                                )
                            }
                        )
                    for function_call_request in function_call_requests_message_like:
                        # print(f"Function call request: {function_call_request}")
                        num_tokens += (
                            len(encoding.encode(function_call_request["content"]))
                            + FUNCTION_CALL_NOT_NONE_NAME_PLUS_TOKENS
                        )

                # The "name" case counter.
                if key == "name":
                    num_tokens += tokens_per_name

                # The "content" case counter.
                else:
                    # Discard any non-string values.
                    if not isinstance(value, str):
                        continue

                    num_tokens += len(encoding.encode(value))
                    # print(f"Token count for {key}: {len(encoding.encode(value))}")

        # Special case of primer tokens.
        num_tokens += PRIMER_TOKENS  # every reply is primed with <|start|>assistant<|message|>

    else:
        # CASE: Estimate tokens for messages with functions.
        # print("Functions found. Using best estimate.")
        num_tokens = _estimate_tokens_with_functions(
            messages,
            encoding.name,
            tokens_per_message,
            tokens_per_name,
            function_definitions,
            function_call=None,
        )

    # Adjust for function call requests.
    # This is based on my own testing, and probably not very accurate at all.
    if add_function_call_request_extra_tokens:
        num_tokens += FUNCTION_CALL_ADJUSTMENT_TOKENS

    return num_tokens
