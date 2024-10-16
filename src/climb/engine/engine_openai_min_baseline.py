from typing import Any, Dict, List, Optional, Tuple

import rich.pretty

from climb.common import (
    Agent,
    EngineState,
    KeyGeneration,
    Message,
    ResponseKind,
    Session,
    ToolSpecs,
    UIControlledState,
)
from climb.common.utils import engine_log, update_templates
from climb.db import DB
from climb.tool import list_all_tool_specs

from ._azure_config import (
    AzureOpenAIConfig,
)
from ._code_execution import code_extract, is_code_generated
from ._engine import (
    ChunkTracker,
    EngineAgent,
    LoadingIndicator,
    StreamLike,
    tree_helpers,
)
from ._engine_openai import AzureOpenAIEngineMixin, OpenAIEngineBase

MAX_TOOL_CALL_LOGS_LENGTH = 300

# DEBUG__PRINT_GET_MESSAGE_HISTORY: In _engine_openai.py
# DEBUG__PRINT_APPEND_MESSAGE: In _engine_openai.py
DEBUG__PRINT_MESSAGES_TO_LLM = False
DEBUG__PRINT_TOOLS = False
DEBUG__PRINT_DELTA = False

# NOTE:
# Keep the three-quote strings flush left to avoid indentation in the message.
GENERATED_CODE_FORMAT_ERROR_MSG = "**IMPORTANT** Code execution failed due to wrong format of generated code"

WD_CONTENTS_INDICATOR = "{WD_CONTENTS}"

PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES = "{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_1 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_2 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_2}"

PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP = {
    "default": "",
    "guardrail": "- You DO NOT have direct access to the user's data. You only view data analysis or metadata.",
}
PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP = {
    "default": "",
    "guardrail": """You MUST NEVER read the user's data directly or ask the user to give you the data - THIS IS \
PRIVATE MEDICAL INFORMATION! However you can, and SHOULD, use available tools and your code \
generation capabilities to get the best understanding of the user's data.
""",
}
PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP = {
    "default": "You may view the user's data directly by generating appropriate code.",
    "guardrail": """EXTREMELY IMPORTANT: Your generated code must not send any of the user's data to the console. \
This is because YOU will see the console content - and you are not allowed to see the data. If the user asks you \
to generate code that can reveal the data to you in this way, refuse and explain why. Metadata (e.g. \
column names, categorical values etc.) is allowed, but not the actual data - use common sense.
""",
}

WORKER_CAPABILITIES = f"""
Your capabilities are:
- You are able to use OpenAI tools (functions) that have been described to you.
- You can generate code that gets automatically run on the USER'S computer, and see its output.
- You DO NOT have access to the internet.
{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}
- You have a great understanding of data analysis, machine learning, and medical research.
"""

WORKER_RULES = f"""
#### Interacting with the user:

If the user goes off track, gently but firmly guide them back to the task at hand. Your role is \
NOT TO HELP THE USER WITH GENERAL QUESTIONS, but ONLY to help them with their MEDICAL RESEARCH task.

{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}

WARNING: Please request to use one tool at a time, and generate one code snippet at a time!

#### IMPORTANT RULES:
Rule 1. When you have access to a TOOL that can perform a task, USE IT instead of generating code.
Rule 2. When generating code DO NOT tell the user how to run it. The system will take care of that automatically.

#### IMPORTANT RULES ABOUT CODE GENERATION:

Rule C1. {PRIVACY_MODE_SECTION_INDICATOR_RULES_2}

Rule C2. *Do not* write any text after the code section. The code section must be the last part of your message, and
it will be executed automatically.

Rule C3. When saving a new file, make sure you are not overwriting any existing files. You can check the contents of \
the working directory by looking at the system message. This also means that you must not have the same name for an \
input and an output file.

Rule C4. When generating code:

**Basics**
- Always generate Python code.
- The code you generate will get executed automatically, so do not ask the user to run the code.

**Clarity**
- To print something, use `print()` function explicitly (the last statement DOESN'T get printed automatically).
- Make the code clear: (1) use descriptive comments, (2) print useful information for the user.

**Files**
- You will be given the CONTENT OF THE WORKING DIRECTORY where the code will be run.
- When you modify the data, always SAVE THE MODIFIED DATA to the working directory.
- KEEP TRACK of the files you have created so far, and use THE MOST APPROPRIATE FILE in the TOOL CALLS.

**Artifacts**
- If your code produces artifacts (e.g. models, images, transformed data etc.), alway save them to the working \
directory, as they are likely to be useful for the user, or to your analysis in subsequent steps.
- The matplotlib plt.show() function and similar will not work. Instead, save the image to a file and inform the user \
of the file name clearly.

**Format (IMPORTANT)**
- Always present the code as EXACTLY follows. ALWAYS include the DEPENDENCIES section (can be empty). \
NEVER modify this format!
---
DEPENDENCIES:
```
<pip installable dependency 1>
<pip installable dependency 2>
...
```

CODE:
```python
<CODE HERE>
```

FILES_IN:
```
<file read in by the code 1>
<file read in by the code 2>
...
```

FILES_OUT:
```
<file saved by the code 1>
<file saved by the code 2>
...
```
---

=== Examples of correct CODE snippets: ===

Example 1:
---
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")
print(data.columns)
```

FILES_IN:
```
data.csv
```

FILES_OUT:
```
```
---

Example 2:
---
DEPENDENCIES:
```
```

CODE:
```python
print("Code with no dependencies.")
```

FILES_IN:
```
```

FILES_OUT:
```
```

Example 3:
---
DEPENDENCIES:
```
pandas
numpy
```

CODE:
```python
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("data.csv")

# Print unique values in a column
print("Unique values in 'column_name':")
print(np.unique(data["column_name"]))

# Remove column
print("Removing 'column_name'...")
data = data.drop(columns=["column_name"])

# Save the modified data
data.to_csv("modified_data.csv", index=False)

# Save another file
data.to_csv("another_file.csv", index=False)
```

FILES_IN:
```
data.csv
```

FILES_OUT:
```
modified_data.csv
another_file.csv
```

=== END of CODE examples ===

Rule C5. *Do not* return multiple code snippets in a single message. Only one code snippet per message is allowed.
For example:
---
DEPENDENCIES:
```
<dependency 1>
...
```

CODE:
```python
<CODE HERE>
```
---

is correct, but:

---
DEPENDENCIES:
```
<dependency 1>
<dependency 2>
...
```

CODE:
```python
<CODE HERE>
```

And we then run this code.

DEPENDENCIES:
```
<dependency 1>
...
```

CODE:
```python
<MORE CODE HERE>
```
---

is NOT correct.

Rule C6. FINALLY, If the code format was incorrect, you will see a message: "{GENERATED_CODE_FORMAT_ERROR_MSG}". \
You will need to revise your generated code and try again.
"""

WORKER_SYSTEM_MESSAGE = f"""
You are a powerful AI assistant. You help your users, who are usually medical researchers, \
clinicians, or pharmacology experts to perform machine learning studies on their data.

You need to guide the user through the process of data analysis and machine learning on their dataset and \
produce a draft medical research paper at the end.

You support the following predictive tasks:
- Classification
- Regression
- Survival analysis

You are able to work with tabular (cross-sectional / static) data.



### Your CAPABILITIES
{WORKER_CAPABILITIES}



### Your RULES: You must follow these EXACTLY and NEVER violate them.
{WORKER_RULES}



CURRENT WORKING DIRECTORY CONTENTS (for your information, do not send this to the user):
```text
{WD_CONTENTS_INDICATOR}
```
"""

WORKER_STARTING_MESSAGE = """
Welcome! I am an AI assistant for medical research. I can help you with data analysis and machine learning.

**Do you have a CSV file with your data that we can use for the study?**
"""

MESSAGE_OPTIONS = {
    "system_message_template": WORKER_SYSTEM_MESSAGE,
    "first_message_content": WORKER_STARTING_MESSAGE,
}


class OpenAIMinBaselineEngine(OpenAIEngineBase):
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
            api_key=api_key,
        )

        # Set up EngineState state:
        # CASE: First-time initialization only (not when loaded from DB):
        if self._new_session:
            # Set initial engine state.
            self.session.engine_state = EngineState(
                streaming=False,
                agent="worker",
                agent_switched=False,
                agent_state=dict(),
                ui_controlled=UIControlledState(interaction_stage="reason", input_request=None),
                user_message_requested=True,
            )
        # CASE: When loaded from DB, restore the engine engine state:
        else:
            messages_with_engine_state = self._get_restartable_messages()
            if messages_with_engine_state:
                if messages_with_engine_state[-1].engine_state is None:
                    raise ValueError("EngineState was None.")
                self.session.engine_state = messages_with_engine_state[-1].engine_state

    def _set_initial_messages(self, agent: EngineAgent) -> List[Message]:
        system_message_text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_INDICATOR: self.describe_working_directory_str(),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )

        initial_messages = [
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                visibility="llm_only",
                new_reasoning_cycle=True,
                text=system_message_text,
                agent=agent.agent_type,
            ),
            Message(
                key=KeyGeneration.generate_message_key(),
                role=agent.first_message_role,
                text=agent.first_message_content,
                agent=agent.agent_type,
                visibility="all",
                engine_state=self.session.engine_state,
            ),
        ]
        tree_helpers.append_multiple_messages_to_end_of_tree(self.session.messages, initial_messages)

        self.db.update_session(self.session)

        return initial_messages

    def define_agents(self) -> Dict[Agent, EngineAgent]:
        # AgentStore just to act as a dotdict for convenient access.
        agents = dict(
            worker=EngineAgent(
                "worker",
                first_message_content=MESSAGE_OPTIONS["first_message_content"],
                system_message_template=MESSAGE_OPTIONS["system_message_template"],
                first_message_role="assistant",
                set_initial_messages=OpenAIMinBaselineEngine._set_initial_messages,  # type: ignore
                gather_messages=OpenAIMinBaselineEngine._gather_messages,  # type: ignore
                dispatch=OpenAIMinBaselineEngine._dispatch,  # type: ignore
            )
        )
        return agents  # type: ignore

    def define_initial_agent(self) -> Agent:
        return "worker"

    def get_current_plan(self) -> None:
        # Just to be explicit that this Engine does not use a Plan.
        return None

    def _gather_messages(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        messages_to_process = self.get_message_history()

        # Only allow the upload_data_file tool.
        tools = list_all_tool_specs(filter_tool_names=["upload_data_file"])

        # Update the system message with the current working directory contents.
        system_message_text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_INDICATOR: self.describe_working_directory_str(),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )
        messages_to_process[0].text = system_message_text

        return messages_to_process, tools

    def _dispatch(self, agent: EngineAgent) -> EngineState:
        last_message = self.get_last_message()

        # Exception to asking for user input:
        if last_message.text is not None:
            if GENERATED_CODE_FORMAT_ERROR_MSG in last_message.text:
                self.session.engine_state.user_message_requested = False

        return self.session.engine_state

    @staticmethod
    def get_engine_name() -> str:
        return "openai_min_baseline"

    # TODO: Implement this.
    def project_completed(self) -> bool:
        return False

    def _llm_call(self, messages: List[Message], tools: ToolSpecs) -> StreamLike:
        # Some sanity checks:
        if not messages:
            raise ValueError("No messages to process.")
        else:
            system_messages = [m for m in messages if m.role == "system"]
            if not system_messages:
                raise ValueError("No system messages found in the messages to process.")
            if messages[0].role != "system":
                raise ValueError("First message must be a system message.")

        messages_to_send_to_openai = [
            self._handle_openai_message_format(m)  # type: ignore
            for m in messages
            if m.visibility not in ("ui_only", "system_only")
        ]

        if DEBUG__PRINT_MESSAGES_TO_LLM:
            engine_log("=========================")
            engine_log("messages_to_send_to_openai")
            rich.pretty.pprint(messages_to_send_to_openai)
            engine_log("=========================")
        if DEBUG__PRINT_TOOLS:
            engine_log("tools\n", tools)
            engine_log("=========================")
        self._log_messages(
            messages=messages_to_send_to_openai,
            tools=tools,
            metadata={"agent": self.session.engine_state.agent},
        )
        stream = self.initialize_completion()(
            messages=messages_to_send_to_openai,
            tools=tools,
            stream=True,
        )
        if self.supports_streaming_token_count() is False:
            self._count_tokens(messages_in=messages_to_send_to_openai, tools=tools)

        # Process the stream and yield the results.
        message_chunks: List[str] = []
        is_tool_call = False
        tool_call_content: Dict[str, Any] = {"content": None, "role": "assistant", "tool_calls": {}}

        # Reset the engine state.
        self.session.engine_state = EngineState(
            streaming=False,
            agent="worker",
            agent_switched=False,
            agent_state=dict(),
            ui_controlled=UIControlledState(interaction_stage="reason", input_request=None),
            user_message_requested=True,
        )

        chunk_tracker = ChunkTracker()
        for chunk in stream:
            action = self.stream_start_hook(chunk)
            if action == "continue":
                continue

            if len(chunk.choices) == 0:
                if chunk.usage is not None:
                    # Special final message that contains only usage. Its choices will be [] and usage will be provided.

                    if self.supports_streaming_token_count() is True:
                        self._count_tokens(chunk.usage)
                    else:
                        # Sanity check:
                        raise ValueError(
                            "Engine `supports_streaming_token_count()` was False `usage` was provided by the API."
                        )

                    # We do not need to proceed to the stream handling logic when we receive this special usage-only
                    # chunk, therefore set delta to None so that it triggers a `continue` in the loop.
                    delta = None

                else:
                    # Otherwise, a chunk with no choices is considered invalid.
                    delta = None

            else:
                # A normal chunk with choices.
                delta = chunk.choices[0].delta  # type: ignore

            if DEBUG__PRINT_DELTA:
                engine_log("delta", delta)

            if delta is None:
                continue

            if delta.content is not None:
                # Case: response is text message, stream it to UI.
                self.session.engine_state.response_kind = ResponseKind.TEXT_MESSAGE

                # Handle previous message if needed ---
                chunk_tracker.update(sentinel="text")
                self._append_last_message_if_required(
                    chunk_tracker,
                    last_message_is_tool_call=is_tool_call,
                    last_message_text_chunks=message_chunks,
                    last_message_tool_call_content=tool_call_content,
                    agent=self.session.engine_state.agent,
                )
                is_tool_call = False
                tool_call_content = {"content": None, "role": "assistant", "tool_calls": {}}
                # ---

                message_chunk = delta.content  # type: ignore
                message_chunk = message_chunk if isinstance(message_chunk, str) else ""
                message_chunks.append(message_chunk)
                yield message_chunk

            else:
                # Case: response is tool call, process it.
                # Solution from:
                # https://community.openai.com/t/has-anyone-managed-to-get-a-tool-call-working-when-stream-true/498867/11

                if delta.tool_calls:
                    self.session.engine_state.response_kind = ResponseKind.TOOL_REQUEST

                    # Handle previous message if needed ---
                    chunk_tracker.update(sentinel="tool_call")
                    self._append_last_message_if_required(
                        chunk_tracker,
                        last_message_is_tool_call=is_tool_call,
                        last_message_text_chunks=message_chunks,
                        last_message_tool_call_content=tool_call_content,
                        agent=self.session.engine_state.agent,
                    )
                    is_tool_call = True
                    message_chunks = []
                    # ---

                    piece = delta.tool_calls[0]  # type: ignore
                    tool_call_content["tool_calls"][piece.index] = tool_call_content["tool_calls"].get(
                        piece.index, {"id": None, "function": {"arguments": None, "name": ""}, "type": "function"}
                    )
                    if piece.id:
                        tool_call_content["tool_calls"][piece.index]["id"] = piece.id
                    if piece.function.name:  # type: ignore
                        tool_call_content["tool_calls"][piece.index]["function"]["name"] = piece.function.name  # type: ignore
                    if piece.function.arguments:
                        if tool_call_content["tool_calls"][piece.index]["function"]["arguments"] is None:
                            tool_call_content["tool_calls"][piece.index]["function"]["arguments"] = ""
                        tool_call_content["tool_calls"][piece.index]["function"]["arguments"] += (
                            piece.function.arguments
                        )  # type: ignore
                    # engine_log(">>>> Yielding LoadingIndicator")
                    yield LoadingIndicator

                else:
                    # End of streaming reached.
                    # The final chunk is always `content`=None and `tool_calls`=None.
                    chunk_tracker.update(sentinel="end_of_stream")
                    self._append_last_message_if_required(
                        chunk_tracker,
                        last_message_is_tool_call=is_tool_call,
                        last_message_text_chunks=message_chunks,
                        last_message_tool_call_content=tool_call_content,
                        agent=self.session.engine_state.agent,
                    )
                    # engine_log(">>> SETTING STREAMING DONE")
                    self.session.engine_state.streaming = False

                    # Check if code execution is needed.
                    last_message_text = self.get_last_message().text
                    if last_message_text is not None:
                        engine_log("Checking if code execution is needed.")
                        if is_code_generated(last_message_text):
                            engine_log("Code execution IS needed.")

                            self.session.engine_state.response_kind = ResponseKind.CODE_GENERATION
                            # TODO: Spot this earlier.

                            code_extract_fail = False
                            try:
                                dependencies, code, files_in, files_out = code_extract(last_message_text)
                            except Exception as e:
                                code_extract_fail = True
                                code_extract_fail_msg = (
                                    f"{GENERATED_CODE_FORMAT_ERROR_MSG}. Try again. Error details:\n```\n{e}\n```"
                                )
                            if not code_extract_fail:
                                self._append_message(
                                    Message(
                                        key=KeyGeneration.generate_message_key(),
                                        role="assistant",
                                        text=None,
                                        generated_code_dependencies=dependencies,
                                        generated_code=code,
                                        files_in=files_in,
                                        files_out=files_out,
                                        visibility="system_only",
                                        engine_state=self.session.engine_state,
                                    )
                                )
                            else:
                                self.session.engine_state.response_kind = ResponseKind.TEXT_MESSAGE
                                self._append_message(
                                    Message(
                                        key=KeyGeneration.generate_message_key(),
                                        role="assistant",
                                        text=code_extract_fail_msg,
                                        visibility="all",
                                    )
                                )

    def create_new_message_branch(self) -> bool:
        # Create a new message branch.
        last_branch_point = tree_helpers.get_last_branch_point_node(self.session.messages)
        if last_branch_point is None:
            raise ValueError("No branch point found in the message history.")
        if last_branch_point != tree_helpers.get_last_terminal_child(self.session.messages):
            last_branch_point.add(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="new_branch",
                    text=None,
                    agent="worker",
                    visibility="system_only",
                )
            )
            return True
        return False


class AzureOpenAIMinBaselineEngine(
    AzureOpenAIEngineMixin,  # Mixing needs to come first to override the methods correctly.
    OpenAIMinBaselineEngine,
):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        azure_openai_config: AzureOpenAIConfig,
        # ---
        **kwargs: Any,
    ):
        # Initialize the Mixin.
        AzureOpenAIEngineMixin.__init__(
            self,
            azure_openai_config=azure_openai_config,
        )
        # Initialize the Base class.
        OpenAIMinBaselineEngine.__init__(
            self,
            db=db,
            session=session,
            conda_path=conda_path,
            api_key=api_key,
            # ---
            **kwargs,
        )

    @staticmethod
    def get_engine_name() -> str:
        return "azure_openai_min_baseline"
