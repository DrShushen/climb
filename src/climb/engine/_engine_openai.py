import copy
import functools
import json
from typing import Any, Callable, Dict, List, Literal, Optional

import rich.pretty
from openai import AzureOpenAI, OpenAI

from climb.common import (
    Agent,
    EngineParameter,
    EngineParameterValue,
    KeyGeneration,
    Message,
    Session,
    ToolCallRecord,
    ToolSpecs,
)
from climb.common.utils import engine_log, filter_out_lines
from climb.db import DB
from climb.tool import ToolBase, ToolOutput, ToolReturnIter, UserInputRequest, get_tool

from ._azure_config import (
    AZURE_OPENAI_CONFIG_PATH,
    AzureOpenAIConfig,
    load_azure_openai_config_item,
    load_azure_openai_configs,
)
from ._code_execution import CodeExecFinishedSentinel, CodeExecReturn, execute_code
from ._engine import (
    ChunkTracker,
    EngineBase,
    PrivacyModeParameter,
    tree_helpers,
)
from ._openai_token_estimation import estimate_prompt_tokens_with_tools
from .const import ALLOWED_MODELS, MODEL_MAX_MESSAGE_TOKENS

MAX_TOOL_CALL_LOGS_LENGTH = 300

DEBUG__PRINT_GET_MESSAGE_HISTORY = False
DEBUG__PRINT_APPEND_MESSAGE = False
DEBUG__PRINT_MESSAGES_TO_LLM = False
DEBUG__PRINT_TOOLS = False
DEBUG__PRINT_DELTA = False


class OpenAIEngineBase(EngineBase):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        # ---
        **kwargs: Any,
    ):
        super().__init__(
            db=db,
            session=session,
            conda_path=conda_path,
        )

        # Prepare the OpenAI client:
        self.api_key = api_key
        self.client = self.initialize_client(api_key=api_key)

        # Currently the `guardrail_with_approval` mode is not implemented.
        if session.engine_params["privacy_mode"] == "guardrail_with_approval":
            raise NotImplementedError("Privacy mode 'guardrail_with_approval' not yet implemented.")

    def _before_define_agents_hook(self) -> None:
        # Useful constants:
        self.max_tokens_per_message = MODEL_MAX_MESSAGE_TOKENS[self.engine_params["model_id"]]  # type: ignore

    @staticmethod
    def supports_streaming_token_count() -> bool:
        return True

    def get_token_counts(self) -> Dict[Agent, int]:
        # Define empty dictionary to store the token counts.
        tokens_dict = dict()
        for agent in self.agents.keys():
            tokens_dict[agent] = None

        # Get the message history in reverse order.
        message_history_rev = self.get_message_history()[::-1]

        # Traverse the message history in reverse order to get the token counts, until the first message with token
        # counts for each agent is found.
        # TODO: This doesn't handle no-agent case properly.
        for message in message_history_rev:
            if message.token_counts is not None:
                for agent, tc in message.token_counts.items():
                    if agent not in tokens_dict:
                        raise ValueError(f"Agent {agent} not found in the agents list.")
                    if tokens_dict[agent] is not None:
                        # Skip if the token count for the agent has already been set.
                        continue
                    if tc is not None:
                        tokens_dict[agent] = tc

        # Change None to zero (not found, so no tokens used).
        for agent in self.agents.keys():
            if tokens_dict[agent] is None:
                tokens_dict[agent] = 0

        return tokens_dict

    @staticmethod
    def get_engine_parameters() -> List[EngineParameter]:
        return [
            EngineParameter(
                name="model_id",
                description=(
                    "OpenAI model ID to use for the research session. "
                    "See [documentation](https://platform.openai.com/docs/models/overview)."
                ),
                kind="enum",
                enum_values=ALLOWED_MODELS,
                default="gpt-4-0125-preview",
            ),
            EngineParameter(
                name="temperature",
                description=(
                    "The temperature to use for the research session. "
                    "See [documentation](https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature)."
                ),
                kind="float",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
            ),
            PrivacyModeParameter,
        ]

    def initialize_client(self, api_key: str) -> Any:
        self.client = OpenAI(api_key=api_key)
        return self.client

    def initialize_completion(self) -> Callable:
        return functools.partial(
            self.client.chat.completions.create,
            model=self.engine_params["model_id"],
            temperature=self.engine_params["temperature"],
            stream_options={"include_usage": True},
        )

    # pylint: disable-next=unused-argument
    def stream_start_hook(self, chunk: Any) -> Literal["noop", "continue"]:
        return "noop"

    def get_message_history(self) -> List[Message]:
        message_list = tree_helpers.get_linear_message_history_to_terminal_child(self.session.messages)

        if DEBUG__PRINT_GET_MESSAGE_HISTORY:
            engine_log("--- GETTING MESSAGE HISTORY ---")
            rich.pretty.pprint(message_list)
            engine_log("--- GETTING MESSAGE HISTORY [DONE] ---")

        return message_list

    def _append_message(self, message: Message) -> None:
        if DEBUG__PRINT_APPEND_MESSAGE:
            engine_log("--- APPENDING MESSAGE ---")
            rich.pretty.pprint(message)
        tree_helpers.append_message_to_end_of_tree(self.session.messages, message)
        if message.visibility == "llm_only_ephemeral":
            self.ephemeral_messages_to_send.append(message.key)
        self.db.update_session(self.session)
        if DEBUG__PRINT_APPEND_MESSAGE:
            engine_log("--- APPENDING MESSAGE [DONE] ---")

    def ingest_user_input(self, user_input: str) -> None:
        if user_input != "":
            self._append_message(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="user",
                    text=user_input,
                    agent=self.session.engine_state.agent,
                    engine_state=self.session.engine_state,
                )
            )

    def _process_message_pre_api(self, message: Message) -> Message:
        # Any preprocessing of the message before its content is sent to the API.
        # Returns a copy to avoid modifying the original message in history.

        message = copy.deepcopy(message)

        # Currently, just used to shorten overly long tool call logs.
        if message.tool_call_logs is not None:
            # Cut out the *middle* of the logs and replace with "\n...\n".
            if len(message.tool_call_logs) > MAX_TOOL_CALL_LOGS_LENGTH:
                if message.outgoing_tool_call is None:
                    raise ValueError("Tool call message must have an outgoing_tool_call.")
                tool_name = message.outgoing_tool_call.name
                tool_logs_are_useful = get_tool(tool_name).logs_useful
                if not tool_logs_are_useful:
                    engine_log(f"Tool logs were > {MAX_TOOL_CALL_LOGS_LENGTH} characters. Shortening.")
                    # engine_log(f"LOGS ORIGINAL:\n{message.tool_call_logs}")
                    half_length = MAX_TOOL_CALL_LOGS_LENGTH // 2
                    message.tool_call_logs = (
                        message.tool_call_logs[:half_length] + "\n...\n" + message.tool_call_logs[-half_length:]
                    )
                    # engine_log(f"LOGS SHORTENED:\n{message.tool_call_logs}")
                else:
                    pass
                    # engine_log(
                    #     f"Tool logs were > {MAX_TOOL_CALL_LOGS_LENGTH} characters, but tool logs were "
                    #     "indicated useful. Not shortening."
                    # )
        return message

    def _handle_openai_message_format(self, message: Message) -> Dict[str, Any]:
        if message.visibility in ("ui_only", "system_only"):
            raise ValueError("UI-only/System-only messages are not to be sent to OpenAI's API.")

        # Process the message content.
        message = self._process_message_pre_api(message)

        if message.role == "code_execution":
            message_role = "assistant"
        else:
            # One of: system, user, assistant, tool
            message_role = message.role
        d = {"role": str(message_role)}

        # Process outgoing tool call message.
        if message.role == "tool":
            if message.outgoing_tool_call is None:
                raise ValueError("Tool call message must have an outgoing_tool_call.")
            d["name"] = message.outgoing_tool_call.name

            d["content"] = f"""Tool call produced logs:
            ---
            {message.tool_call_logs or ""}
            ---

            Tool call returned:
            ---
            {message.tool_call_return or ""}
            ---
            """

            # TODO: We probably want to TEXT content of the tool call report to also be sent to LLM...

            if message.outgoing_tool_call.engine_id is None:
                raise ValueError("Tool call result must have an engine_id.")
            d["tool_call_id"] = message.outgoing_tool_call.engine_id

        # Process incoming tool call message.
        elif message.incoming_tool_calls:
            d["content"] = ""
            d["tool_calls"] = [  # type: ignore
                {
                    "id": tc.engine_id,
                    "function": {"name": tc.name, "arguments": tc.arguments},
                    "type": "function",
                }
                for tc in message.incoming_tool_calls
            ]

        # Process generated code execution message.
        elif message.role == "code_execution":
            # Process code execution results for the message to be set to the LLM.
            code_exec_msg = (
                "Code execution completed successfully." if message.generated_code_success else "Code execution failed."
            )
            code_exec_msg += f"\nSTDOUT:\n\n```\n{message.generated_code_stdout}\n```"
            if message.generated_code_stderr:
                code_exec_msg += f"\nSTDERR:\n\n```\n{message.generated_code_stderr}\n```"
            d["content"] = code_exec_msg

        # Message text.
        else:
            if message.text is None:
                raise ValueError("Text message must have content.")
            d["content"] = message.text

        return d

    def _append_last_message_if_required(
        self,
        chunk_tracker: ChunkTracker,
        last_message_is_tool_call: bool,
        last_message_text_chunks: Optional[List[str]],
        last_message_tool_call_content: Optional[Dict[str, Any]],
        agent: Agent,
    ) -> None:
        if chunk_tracker.processing_required():
            if not last_message_is_tool_call:
                if last_message_text_chunks is None:
                    raise ValueError("`message_chunks` must not be None.")
                message = "".join(last_message_text_chunks)
                if message != "":
                    self._append_message(
                        Message(
                            key=KeyGeneration.generate_message_key(),
                            role="assistant",
                            text=message,
                            agent=agent,
                            engine_state=self.session.engine_state,
                        )
                    )
                    engine_log("[!] Message received from API call appended: TEXT.")
                else:
                    engine_log("[!] Message received from API call *not* appended: was blank.")
            else:
                if last_message_tool_call_content is None:
                    raise ValueError("`tool_call_content` must not be None.")

                # HACK - In future will properly handle 2x+ tool calls at once.
                # TODO - fix this HACK
                if len(last_message_tool_call_content["tool_calls"]) > 1:
                    last_message_tool_call_content["tool_calls"] = dict(
                        [list(last_message_tool_call_content["tool_calls"].items())[0]]
                    )
                    # ^ Cut off to just first k: v in dictionary.
                # -------------------------------------------------------------

                incoming_tool_calls = [
                    ToolCallRecord(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                        engine_id=tc["id"],
                    )
                    for tc in last_message_tool_call_content["tool_calls"].values()
                ]
                if len(incoming_tool_calls) > 1:
                    raise ValueError("More than one incoming tool call found.")
                self._append_message(
                    Message(
                        key=KeyGeneration.generate_message_key(),
                        role="assistant",
                        text=None,
                        visibility="llm_only",
                        incoming_tool_calls=incoming_tool_calls,
                        agent=agent,
                        engine_state=self.session.engine_state,
                    )
                )
                engine_log("[!] Message received from API call appended: TOOL_CALL.")
                if DEBUG__PRINT_TOOLS:
                    engine_log("--- --- ---")
                    engine_log("tool_call_content\n", last_message_tool_call_content)
                    engine_log("--- --- ---")

                self.session.engine_state.tool_request = incoming_tool_calls[0]

    def _count_tokens(
        self,
        # Provide when `supports_streaming_token_count()` is True.:
        usage: Optional[Any] = None,
        # Provide when `supports_streaming_token_count()` is False.:
        messages_in: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[ToolSpecs] = None,
    ) -> None:
        """Count the number or *prompt tokens* used in this LLM call step and record it on the Message."""

        # Validation.
        if self.supports_streaming_token_count():
            if usage is None:
                raise ValueError("`usage` must be provided when `supports_streaming_token_count()` is True.")
        else:
            if messages_in is None:
                raise ValueError("`messages_in` must be provided when `supports_streaming_token_count()` is False.")
            # NOTE: tools is optional here.

        # engine_log("usage:", usage)
        n_prompt_tokens = 0
        if self.supports_streaming_token_count():
            # Use official token count returned by the API.
            if usage is not None:
                engine_log("CASE: `usage` provided by API. Will use official token count.")
                n_prompt_tokens = usage.prompt_tokens
            else:
                raise ValueError(
                    "`usage` not provided by API even though engine.supports_streaming_token_count() is True. "
                    "This should not happen."
                )
        else:
            engine_log("CASE: `usage` not provided by API. Will use custom token count estimator.")
            n_prompt_tokens = estimate_prompt_tokens_with_tools(
                model=self.engine_params["model_id"],  # type: ignore
                messages_in=messages_in,  # type: ignore
                function_definitions=[t["function"] for t in tools] if tools else None,
            )

        # Update the record on the message.
        tokens_dict = dict()
        for agent in self.agents.keys():
            tokens_dict[agent] = None
        tokens_dict[self.session.engine_state.agent] = n_prompt_tokens
        self.get_last_message().token_counts = tokens_dict

    def _get_restartable_messages(self) -> List[Message]:
        messages = self.get_message_history()
        restartable_messages = [m for m in messages if m.role == "assistant" and m.engine_state is not None]
        return restartable_messages

    def discard_last(self) -> bool:
        self.stop_tool_execution()

        messages = self.get_message_history()
        restartable_messages = self._get_restartable_messages()
        if not restartable_messages:
            engine_log("discard_last fail: No restartable messages found.")
            return False
            # raise ValueError("No restartable messages found.")
        last_restartable_message = restartable_messages[-1]

        # If the last_restartable_message is also the last message, go back one more.
        if last_restartable_message.key == messages[-1].key:
            engine_log("last_restartable_message was the last message, going back one more.")
            if len(restartable_messages) < 2:
                raise ValueError("No previous restartable messages found.")
            last_restartable_message = restartable_messages[-2]

        engine_log("last_restartable_message", last_restartable_message)
        if last_restartable_message.engine_state is None:
            engine_log("discard_last fail: Engine state in last restartable message is None.")
            return False
            # raise ValueError("Engine state in last restartable message must not be None.")
        self.session.engine_state = last_restartable_message.engine_state

        # === Update message history - cut off all the messages after the last restartable message. ===
        last_restartable_message_node = self.session.messages.find_first(
            match=lambda m: m.data.key == last_restartable_message.key
        )
        if last_restartable_message_node is None:
            raise ValueError("Last restartable message not found in the message history.")
        last_restartable_message_node.remove_children()  # NOTE: This updates the underlying tree.
        # === Update message history - cut off all the messages after the last restartable message. [END] ===

        # Use a copy of the last_restartable_message's Engine state, to avoid accidental changes to the
        # historic snapshot object.
        # engine_log("last_restartable_message.engine_state\n", last_restartable_message.engine_state)
        self.session.engine_state = copy.deepcopy(last_restartable_message.engine_state)
        engine_log("self.session.engine_state\n", self.session.engine_state)

        # import rich.pretty
        # rich.pretty.pprint(self.get_message_history())

        return True

    # TODO: May want to improve/rethink.
    def restart_at_user_message(self, key: str) -> bool:
        self.stop_tool_execution()

        messages = self.get_message_history()

        # print("key", key)
        if key not in [m.key for m in messages]:
            engine_log("restart_at_user_message fail: Key not found in messages.")
            return False

        # Get the message one before the user message.
        user_input_message = None
        restart_message = None
        for m in messages:
            if m.key == key:
                user_input_message = m
                break
            restart_message = m
        if user_input_message is None:
            engine_log("restart_at_user_message fail: User input message not found.")
            return False
        if restart_message is None:
            engine_log("restart_at_user_message fail: Restart message not found.")
            return False
        if restart_message.engine_state is None:
            engine_log("restart_at_user_message fail: Engine state in restart message is None.")
            return False

        # === Update message history - cut off all the messages after the restart_message. ===
        restart_message_node = self.session.messages.find_first(match=lambda m: m.data.key == restart_message.key)
        print("restart_message_node", restart_message_node)
        if restart_message_node is None:
            raise ValueError("Restart message not found in the message history.")
        restart_message_node.remove_children()  # NOTE: This updates the underlying tree.
        # === Update message history - cut off all the messages after the last restartable message. [END] ===

        # Use a copy of the last_restartable_message's Engine state, to avoid accidental changes to the
        # historic snapshot object.
        self.session.engine_state = copy.deepcopy(restart_message.engine_state)
        self.session.engine_state.ui_controlled.input_placeholder = user_input_message.text
        engine_log("self.session.engine_state\n", self.session.engine_state)

        return True

    def execute_generated_code(self) -> CodeExecReturn:
        last_message = self.get_last_message()
        if last_message.generated_code is None:
            raise ValueError("No generated code to execute.")
        if last_message.generated_code_dependencies is None:
            last_message.generated_code_dependencies = []

        stdout_parts: List[str] = []
        error_msg = ""
        success = False
        for stream in execute_code(
            code_file_name=last_message.key,
            working_directory=self.working_directory_abs,
            generated_code=last_message.generated_code,
            dependencies=last_message.generated_code_dependencies,
            conda_path=self.conda_path,
        ):
            if isinstance(stream, CodeExecFinishedSentinel):
                if stream.status == "success":
                    success = True
                else:
                    success = False
                    error_msg = stream.error_message if stream.error_message else ""
            else:
                stdout_parts.append(stream)
            yield stream
        stdout = "".join(stdout_parts)

        self._append_message(
            Message(
                key=KeyGeneration.generate_message_key(),
                role="code_execution",
                text=None,
                visibility="all",
                # ---
                generated_code_success=success,
                generated_code_stderr=error_msg,
                generated_code_stdout=stdout,
                # ---
            )
        )

    def get_additional_tool_call_kwargs(
        self,
    ) -> Dict[str, Any]:
        return dict(
            api_key=self.api_key,
            engine_params=self.engine_params,
        )

    def execute_tool_call(
        self,
        tool_call: ToolCallRecord,
        user_input: UserInputRequest,
        # ui_message_processor: UiToolCallMessageProcessor,
    ) -> ToolReturnIter:
        tool = get_tool(tool_call.name)
        self.session.engine_state.executing_tool = tool_call.name
        self.executing_tool = tool

        tool.receive_working_directory(self.working_directory_abs)
        tool.receive_user_inputs_requested(user_input)

        try:
            kwargs: Dict[str, Any] = json.loads(tool_call.arguments)
            tool_specific_kwargs = copy.deepcopy(kwargs)
            kwargs["session"] = self.session
            kwargs["additional_kwargs_required"] = self.get_additional_tool_call_kwargs()
        except Exception as e:
            raise ValueError(f"Failed to parse tool call arguments returned from API:\n{e}") from e

        engine_log(f"Executing tool call: {tool_call.name}")
        engine_log(f"Tool call arguments:\n{tool_specific_kwargs}")

        # TODO: Handle tool execution fail case.
        logs_output = tool.execute(**kwargs)
        total_logs_output = ""
        return_holder = ToolOutput()
        for logs_o in logs_output:
            if not isinstance(logs_o, ToolOutput):
                total_logs_output += logs_o
                yield logs_o
            else:
                return_holder = logs_o

        total_logs_output = filter_out_lines(total_logs_output)

        self.session.engine_state.executing_tool = None
        self.executing_tool = None
        tool_execution_success = return_holder.success
        tool_return = return_holder.tool_return
        user_report_outputs = return_holder.user_report_outputs

        # Put *output for LLM* into a Message.
        self._append_message(
            Message(
                key=KeyGeneration.generate_message_key(),
                role="tool",
                text=None,
                outgoing_tool_call=tool_call,
                # ---
                tool_call_success=tool_execution_success,
                tool_call_logs=total_logs_output,
                tool_call_return=tool_return,
                tool_call_user_report=user_report_outputs,
                # ---
                files_in=return_holder.files_in,
                files_out=return_holder.files_out,
            )
        )

        yield return_holder

    def describe_tool_to_user(self, tool: ToolBase) -> str:
        tool_description_for_user = tool.description_for_user

        # Make sure tool_description_for_user starts with lower case letter and ends with "."
        if not tool_description_for_user.endswith("."):
            tool_description_for_user += "."
        tool_description_for_user = tool_description_for_user[0].lower() + tool_description_for_user[1:]

        tool_descr = (
            f"I will use the 🔧 tool `{tool.name}` to perform the task.\n\nThis tool {tool_description_for_user}"
        )
        self._append_message(
            Message(
                key=KeyGeneration.generate_message_key(),
                role="assistant",
                text=tool_descr,
                visibility="ui_only",  # NOTE.
            )
        )
        return tool_descr


class AzureOpenAIEngineMixin:
    # For typing only here:
    api_key: str
    azure_openai_config: AzureOpenAIConfig
    engine_params: Dict[str, EngineParameterValue]
    # --- --- --- --- ---

    def __init__(
        self,
        azure_openai_config: AzureOpenAIConfig,
    ):
        self.azure_openai_config = azure_openai_config

    @staticmethod
    def supports_streaming_token_count() -> bool:
        return False

    def initialize_client(self, api_key: str) -> Any:
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_openai_config.endpoint,
            api_version=self.azure_openai_config.api_version,
            api_key=api_key,
        )
        return self.client

    def initialize_completion(self) -> Callable:
        return functools.partial(  # pyright: ignore
            self.client.chat.completions.create,
            model=self.azure_openai_config.deployment_name,
            temperature=self.engine_params["temperature"],
            # NOTE: stream_options *not* supported by Azure OpenAI as of 2024-08.
        )

    def stream_start_hook(self, chunk: Any) -> Literal["noop", "continue"]:
        if not chunk.choices:
            return "continue"
        return "noop"

    @staticmethod
    def set_model_id(*, config_item_name: str) -> str:
        config = load_azure_openai_config_item(AZURE_OPENAI_CONFIG_PATH, config_item_name)
        return config.model

    @staticmethod
    def get_engine_parameters() -> List[EngineParameter]:
        configs = load_azure_openai_configs(AZURE_OPENAI_CONFIG_PATH)
        configs_found = bool(configs)
        return [
            EngineParameter(
                name="config_item_name",
                description=(
                    f"The `name` field from the Azure OpenAI config file (`{AZURE_OPENAI_CONFIG_PATH}`) "
                    "corresponding to the model you want to use."
                ),
                kind="enum",
                enum_values=[x.name for x in configs],
                default=[x.name for x in configs][0] if configs_found else "",
                disabled=not configs_found,
            ),
            EngineParameter(
                name="model_id",
                description=(
                    "OpenAI model ID to use for the research session. "
                    "See [documentation](https://platform.openai.com/docs/models/overview)."
                ),
                kind="enum",
                enum_values=ALLOWED_MODELS,
                default="gpt-4-0125-preview",
                set_by_static_method="set_model_id",
                disabled=True,
            ),
            EngineParameter(
                name="temperature",
                description=(
                    "The temperature to use for the research session. "
                    "See [documentation](https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature)."
                ),
                kind="float",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                disabled=not configs_found,
            ),
            PrivacyModeParameter,
        ]

    def get_additional_tool_call_kwargs(
        self,
    ) -> Dict[str, Any]:
        return dict(
            azure_endpoint=self.azure_openai_config.endpoint,
            api_version=self.azure_openai_config.api_version,
            api_key=self.api_key,
            engine_params=self.engine_params,
            azure_openai_config=self.azure_openai_config,
        )
