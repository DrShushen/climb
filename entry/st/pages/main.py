import os
import re
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, cast

import markdown
import matplotlib.figure
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac

from climb.common.disclaimer import DISCLAIMER_TEXT
from climb.common.exc import EXC_DOCS_REFS

try:
    from weasyprint import HTML

    WEASYPRINT_WARNING_FULL = ""
    WEASYPRINT_WORKING = True
except Exception as ex:
    WEASYPRINT_WARNING_FULL = (
        f"Error importing weasyprint:\n{ex}\n"
        "PDF report generation will not be supported but you can still use CliMB.\n"
        "For a possible resolution, see the troubleshooting documentation page:\n"
        f"{EXC_DOCS_REFS['troubleshooting_win_pangoft']}"
    )
    print(WEASYPRINT_WARNING_FULL)
    WEASYPRINT_WORKING = False

from climb.common import Message, create_new_session
from climb.common.data_structures import Agent, InteractionStage, ResponseKind, Session, UploadedFileAbstraction
from climb.common.utils import analyze_df_modifications, dedent, replace_str_from_dict, ui_log
from climb.db import DB
from climb.db.tinydb_db import TinyDB_DB
from climb.engine import (
    MODEL_CONTEXT_SIZE,
    CodeExecFinishedSentinel,
    CodeExecReturn,
    EngineBase,
    LoadingIndicator,
    StreamLike,
    create_engine,
)
from climb.tool import ToolOutput, ToolReturnIter, get_tool
from climb.ui import st_common

# region: === Constants and settings. ===

dotenv_config = st_common.get_dotenv_config()

# DEBUG controls. ---

DEBUG_PANEL = str(dotenv_config.get("DEBUG_PANEL", "False")).lower() == "true"

DEBUG__SHOW_ACTIVE_AGENT = False
DEBUG__SHOW_AGENT_IN_HISTORY = False

DEBUG__SHOW_SYSTEM_MESSAGES = False
DEBUG__SHOW_LLM_ONLY_MESSAGES = False
DEBUG__SHOW_LLM_ONLY_EPHEMERAL_MESSAGES = False

if DEBUG__SHOW_SYSTEM_MESSAGES:
    if "system" not in st_common.SHOW_ROLES:
        st_common.SHOW_ROLES.append("system")
if DEBUG__SHOW_LLM_ONLY_MESSAGES:
    if "llm_only" not in st_common.SHOW_VISIBILITIES:
        st_common.SHOW_VISIBILITIES.append("llm_only")
if DEBUG__SHOW_LLM_ONLY_EPHEMERAL_MESSAGES:
    if "llm_only_ephemeral" not in st_common.SHOW_VISIBILITIES:
        st_common.SHOW_VISIBILITIES.append("llm_only_ephemeral")

# DEBUG controls [END]. ---


TOOL_LOGS_PREFIX = "Tool logs:"
TOOL_RETURN_PREFIX = "Tool output:"
TOOL_USER_REPORT_PREFIX = "The tool produced the following report:"

CODE_ITSELF_PREFIX = "Generated code:"
CODE_EXECUTION_OUT_PREFIX = "Code execution output:"

CODE_GEN_DUMMY_MESSAGE = "📟 `MODEL IS GENERATING CODE...`"
COORDINATOR_PLAN_DUMMY_MESSAGE = "💡 `PLANNING WORK...`"

AVATAR_MAP = {
    "user": None,
    "assistant": None,
    "tool": "🔧",
    "code_execution": "📟",
    "system": "🖥️",
}

AGENT_TEXT_COLOR_MAP = {
    "coordinator": "green",
    "worker": "red",
    "supervisor": "blue",
}

# endregion


# region: === Database, streamlit state, session, engine. ===

# NOTE:
# Some streamlit state requires the database.
# Engine initialization requires the session, which requires the database.

# 1. Database.
db = TinyDB_DB("db.json")
user_settings = db.get_user_settings()

# 2. Streamlit state.

st_common.initialize_common_st_state(db)


# 3. Engine and session:


def initialize_session(db: DB) -> Session:
    if st.session_state.active_session_key is not None:
        session = db.get_session(st.session_state.active_session_key)
    else:
        session = create_new_session(
            session_name=st.session_state.new_session_settings["session_name"],
            engine_name=st.session_state.new_session_settings["engine_name"],
            engine_params=st.session_state.new_session_settings["engine_params"],
            db=db,
        )
    return session


if "engine" not in st.session_state:
    # Engine object saved on session_state to make sure it doesn't get re-instantiated on every rerun.
    session = initialize_session(db)

    engine_initialized = create_engine(db, session, dotenv_config)

    st.session_state.engine = engine_initialized
    st.session_state.active_session_key = st.session_state.engine.session.session_key

    st.rerun()


def engine() -> EngineBase:
    # A convenience function.
    return st.session_state.engine


if "active_dashboard_tab" not in st.session_state:
    st.session_state.active_tab = "tab_session"

# Get this value near the start of this streamlit script in order to avoid triggering unnecessary reruns later.
active_agent = engine().get_state().agent


# endregion


# region: === General helper functions. ===

LOADING_INDICATOR_EXC_MSG = "LoadingIndicator detected"


def show_in_history_if(message: Message) -> bool:
    show = message.visibility in st_common.SHOW_VISIBILITIES and message.role in st_common.SHOW_ROLES

    special = {
        # Show these visibilities (value) regardless of whether the role (key) is hidden.
        "system": ("llm_only_ephemeral",),
    }
    for role, visibilities in special.items():
        if (
            message.role == role
            and message.visibility in visibilities
            and message.visibility in st_common.SHOW_VISIBILITIES
        ):
            show = True

    return show


def tool_out_wrapper(
    iterable: ToolReturnIter,
    return_holder: ToolOutput,
    wrap_in: str = "```\n",
    wrap_out: str = "\n```\n",
    # hook_pre_fns: will be executed AFTER the streaming starts, only once.
    # Useful for any UI updates that require the tool running state.
    hook_pre_fns: List[Callable] = [],
) -> Iterable[str]:
    hook_pre_fns_executed = False
    yield wrap_in
    for x in iterable:
        if not isinstance(x, ToolOutput):
            if not hook_pre_fns_executed:
                for hook_fn in hook_pre_fns:
                    hook_fn()
                hook_pre_fns_executed = True
            yield x
        else:
            return_holder.tool_return = x.tool_return
            return_holder.user_report_outputs = x.user_report_outputs
            return_holder.success = x.success
    yield wrap_out


def code_out_wrapper(
    iterable: CodeExecReturn,
    finished_stl: CodeExecFinishedSentinel,
    wrap_in: str = "```\n",
    wrap_out: str = "\n```\n",
) -> Iterable[str]:
    yield wrap_in
    for x in iterable:
        if not isinstance(x, CodeExecFinishedSentinel):
            yield x
        else:
            finished_stl.status = x.status
            finished_stl.error_message = x.error_message
    yield wrap_out


def id_and_excise_code(
    text: str,  # pylint: disable=redefined-outer-name
) -> Tuple[str, Optional[str], Optional[str]]:
    # Helper functions remove_{end,front}_lines.
    # These are used to remove the "---" lines at the end and beginning of the code block, only for visual purposes.

    def remove_end_lines(t: str) -> str:
        if "---" in t[-5:]:
            return t[:-5] + t[-5:].replace("---", "")
        return t

    def remove_front_lines(t: str) -> str:
        if "---" in t[:5]:
            return t[:5].replace("---", "") + t[5:]
        return t

    pattern0 = r"(DEPENDENCIES:.*?```)(.*?)"
    match0 = re.search(pattern0, text, re.DOTALL)

    if match0:
        # CASE: Code generation at least started (DEPENDENCIES block found).

        # Regex pattern to extend through the end of the CODE block
        pattern1 = r"(DEPENDENCIES:.*?```.*?```.*?```)(.*?```)"
        match1 = re.search(pattern1, text, re.DOTALL)

        if match1:
            # CASE: Code generation finished (end of CODE block found).

            start1 = match1.start()
            end1 = match1.end()

            # Part 1: Before DEPENDENCIES.
            before_dependencies = text[:start1]
            # Part 2: DEPENDENCIES to the end of CODE.
            dependencies_to_code = text[start1:end1]
            # Part 3: After CODE.
            after_code = text[end1:]

            return remove_end_lines(before_dependencies), dependencies_to_code, remove_front_lines(after_code)

        else:
            # CASE: Code generation started but not finished (end of CODE block not reached).

            start0 = match0.start()

            # Part 1: Before DEPENDENCIES.
            before_dependencies = text[:start0]
            # Part 2: DEPENDENCIES to the end of string thus far (ongoing code generation).
            ongoing_code_gen = text[start0:]

            return remove_end_lines(before_dependencies), ongoing_code_gen, None

    else:
        # CASE: No code generation related blocks found.
        return text, None, None


def process_md_images(input_text: str, wd: str) -> str:
    if "<WD>/" not in input_text:
        return input_text

    # Get rid of ` if needed.
    pattern = r"`<WD>(.+?)(\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.tiff)`"
    replacement = r"<WD>\1\2"
    output_text = re.sub(pattern, replacement, input_text)

    # Now extract the path and apply the `img_to_html` function.
    pattern = r"<WD>(.+?)(\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.tiff)"
    matches = re.findall(pattern, output_text)
    for match in matches:
        img_path = wd + match[0] + match[1]
        img_html = st_common.img_to_html(img_path)
        output_text = output_text.replace(f"<WD>{match[0]}{match[1]}", img_html)

    return output_text


def update_debug_panel_ui_comm() -> None:
    if DEBUG_PANEL and "cont_uicomm" in globals():
        cont_uicomm.empty()
        cont_uicomm.code(engine().get_state())


# Tab warnings mechanism-related functions.

tab_warning_areas = dict()
tab_warning_tool_exec = (
    "🛠️ **Tool is currently executing. Interacting with the dashboard will restart the tool. Please wait.**"
)


def add_tab_warning_area(tab_name: str) -> Any:
    c, _ = st.columns([0.99, 1 - 0.99])
    with c:
        placeholder = st.empty()
        tab_warning_areas[tab_name] = placeholder
    return placeholder


def show_tab_warnings(warning: str = tab_warning_tool_exec) -> None:
    for tab_name in tab_warning_areas:
        if tab_name == st.session_state.active_tab:
            with tab_warning_areas[tab_name].container():
                st.markdown("")  # Spacer.
                st.warning(warning)


def show_tab_warnings_if_tool_running():
    if engine().get_state().executing_tool:
        show_tab_warnings()


# TODO: Needs to be tidied up.
def process_messages_for_report(messages: List[Message]) -> str:
    markdowns = []

    for message in messages:
        this_markdown = ""

        # 1. Show the message.text.
        message_text = message.text or ""
        this_markdown += f"**{message.role.capitalize()}:**\n\n"

        # 1-A: Handle code blocks (can result in multiple text sections)
        pre_code, code_block, post_code = id_and_excise_code(message_text)
        text_sections: List[Tuple[Literal["text", "code"], str]] = [("text", pre_code)]
        if code_block is not None:
            text_sections.append(("code", code_block))
            if post_code is not None:
                text_sections.append(("text", post_code))

        # 1-B: Any other post-processing, applied to each section individually.
        for text_or_code, content in text_sections:
            # Any text post-processing:
            if text_or_code == "text":
                # Handle markdown (embedded) images:
                # forces_unsafe = False

                # This section contains currently fairly hacky implementation of various message history
                # post-processings.
                # TODO: Make this systematic. Coordinate with in-stream filtering too.
                # - agent coordination special value replacements.
                if "TASK COMPLETED" in content:
                    content = content.replace("TASK COMPLETED", "✅🎉 Task completed!")
                if "= CONTINUE =" in content:
                    content = content.replace("= CONTINUE =", "▶️")
                if message.agent == "coordinator":
                    content = "💡 `Planning step`"
                    # content[: content.index("SYSTEM:")] + "💡 `Planning completed`"
                # - <WD>/ image replacement.
                if "<WD>/" in content:
                    content = process_md_images(content, wd=engine().working_directory)  # HACK # type: ignore
                    # forces_unsafe = True
                # --- --- ---

                this_markdown += content + "\n"
            # Any code post-processing:
            elif text_or_code == "code":
                # with st.expander(CODE_ITSELF_PREFIX, expanded=engine().session.session_settings.show_code):
                this_markdown += "\n" + f"**{CODE_ITSELF_PREFIX}**" + "\n\n" + content + "\n"
            else:
                raise ValueError(f"Unexpected `text_or_code` value: {text_or_code}")

        # 2. Show various tool call outputs.
        if message.role == "tool":
            # Validate.
            # TODO: Move this logic to `Message` eventually.
            if message.outgoing_tool_call is None:
                raise ValueError("`'tool'` message without `outgoing_tool_call`")
            if message.tool_call_success is None:
                raise ValueError("`'tool'` message without `tool_call_success`")
            if message.tool_call_return is None:
                raise ValueError("`'tool'` message without `tool_call_return`")
            # --- --- ---

            top_text = f"Tool `{message.outgoing_tool_call.name}`"
            if message.tool_call_success:
                top_text += " completed successfully ✅"
            else:
                top_text += " failed ❌"
            this_markdown += top_text + "\n"

            # Tool call logs.
            if message.tool_call_logs:
                # with st.expander(TOOL_LOGS_PREFIX, expanded=engine().session.session_settings.show_tool_call_logs):
                #    st.markdown(f"```\n{message.tool_call_logs}\n```")
                this_markdown += (
                    "\n" + f"**{TOOL_LOGS_PREFIX}**" + "\n\n" + f"```\n{message.tool_call_logs}\n```" + "\n"
                )

            # Tool call return.
            if message.tool_call_return:
                # with st.expander(
                #     TOOL_RETURN_PREFIX, expanded=engine().session.session_settings.show_tool_call_return
                # ):
                #     st.markdown(f"```\n{message.tool_call_return}\n```")
                this_markdown += (
                    "\n" + f"**{TOOL_RETURN_PREFIX}**" + "\n\n" + f"```\n{message.tool_call_return}\n```" + "\n"
                )

            # User-only report outputs.
            if message.tool_call_user_report:
                this_markdown += "\n" + f"**{TOOL_USER_REPORT_PREFIX}**" + "\n\n"
                # with st.expander(TOOL_USER_REPORT_PREFIX, expanded=True):
                for user_output in message.tool_call_user_report:  # pyright: ignore
                    if isinstance(user_output, str):
                        # st.markdown(user_output)
                        this_markdown += user_output + "\n"
                    elif isinstance(user_output, go.Figure):
                        # st.plotly_chart(user_output)
                        this_markdown += "\n**Plotly figures not yet supported in the report**\n"
                    elif isinstance(user_output, matplotlib.figure.Figure):
                        # st.pyplot(user_output)
                        this_markdown += "\n**Matplotlib figures not yet supported in the report**\n"
                    else:
                        raise ValueError(f"Unexpected user output type: {type(user_output)}")

        # 3. Show code execution outputs.
        if message.role == "code_execution":
            # Validate.
            if message.generated_code_success is None:
                raise ValueError("`'code_execution'` message without `generated_code_success`")
            # --- --- ---

            top_message = (
                "Code execution finished successfully ✅"
                if message.generated_code_success
                else "Code execution failed ❌"
            )
            # st.markdown(top_message)
            this_markdown += top_message + "\n"
            # with st.expander(CODE_EXECUTION_OUT_PREFIX, expanded=engine().session.session_settings.show_code_out):
            output_message = ""
            if message.generated_code_stdout:
                output_message += f"\n```\n{message.generated_code_stdout}\n```\n"
            # NOTE: Should we show stderr regardless of the show logs setting?
            if message.generated_code_stderr:
                output_message += f"\n**Error:**\n\n```\n{message.generated_code_stderr}\n```"
            # st.markdown(output_message)
            this_markdown += f"**{CODE_EXECUTION_OUT_PREFIX}**" + "\n\n" + output_message + "\n"

        markdowns.append(this_markdown)

    final = "\n\n---\n\n".join(markdowns)
    print(final)

    return final


# TODO: Needs to be tidied up.
def prepare_report(working_dir: str) -> str:
    messages = engine().get_message_history()
    messages = [m for m in messages if show_in_history_if(m)]
    report = process_messages_for_report(messages)

    def markdown_to_pdf(markdown_text: str, output_pdf: str):
        # Convert Markdown to HTML
        html_text = markdown.markdown(markdown_text, extensions=["fenced_code"])

        html_text = html_text.replace(os.path.abspath(working_dir), "./")
        html_text = html_text.replace(working_dir, "./")

        html_text = f"""
        <style>
        code {{
            font-size: 0.8em;
        }}
        @page {{
            size: A3;
        }}
        </style>

        <h2>Session report</h2>
        <p><b>Session name:</b> {engine().session.friendly_name}</p>
        <p><b>Working directory:</b></p>
        <p>You will find all the session files, including the transformed data and the created models here</p>
        <code><a href='{working_dir}'>{working_dir}</a></code>
        <br/>

        <h3>Full conversation history</h2>
        {html_text}
        """

        # print(html_text)
        # # Save the HTML content to a file
        # with open("output.html", "w") as f:
        #     f.write(html_text)

        # Generate a PDF from the HTML content
        HTML(string=html_text, base_url=os.path.abspath(working_dir)).write_pdf(output_pdf)

    output_pdf = os.path.join(working_dir, "report.pdf")
    markdown_to_pdf(report, output_pdf)

    return output_pdf


# Tab warnings mechanism-related functions [END].


def df_from_plan(plan_list_of_dicts: List[dict]) -> pd.DataFrame:
    COLS = ["Project Stage", "Task", "Subtask", "Status"]
    STATUS_MAP = {
        "not_started": "◼️ Not started",
        "in_progress": "⏳ In progress",
        "skipped": "⏩ Skipped",
        "completed": "✅ Completed",
        "needs_redoing": "❌ Needs redoing",
    }
    df = pd.DataFrame(columns=COLS)
    for task in plan_list_of_dicts:
        for subtask in task["subtasks"]:
            # Use pandas concat to add the subtask to the DataFrame
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                task["project_stage"],
                                task["task_name"],
                                subtask["subtask_name"],
                                STATUS_MAP[subtask["subtask_status"]],
                            ]
                        ],
                        columns=COLS,
                    ),
                ],
                ignore_index=True,
            )
    return df


def plot_proportion(token_count_per_agent: Dict[Agent, int], context_size: int) -> go.Figure:
    rough_bar_height = 20
    legend_gap = 20

    layout = go.Layout(
        height=rough_bar_height * len(token_count_per_agent) + legend_gap,
        margin=go.layout.Margin(l=10, r=10, b=0, t=(legend_gap + 10), pad=4),
    )

    fig = go.Figure(layout=layout)

    y = []
    x_used = []
    x_remaining = []
    x_custom_actual = []
    for agent, val in token_count_per_agent.items():
        y.append(agent)
        used = val / context_size
        remaining = 1 - used
        x_used.append(used)
        x_remaining.append(remaining)
        x_custom_actual.append([val, context_size])

    fig.add_trace(
        go.Bar(
            y=y,
            x=x_used,
            customdata=x_custom_actual,
            name="Used",
            orientation="h",
            marker=dict(color=px.colors.sequential.Blues[5]),
            hovertemplate="Used:<br>%{customdata[0]:,.0f} / %{customdata[1]:,.0f} tokens<br><b>%{x:0.0%}</b><br><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=y,
            x=x_remaining,
            customdata=x_custom_actual,
            name="Remaining",
            orientation="h",
            marker=dict(color=px.colors.sequential.Blues[2]),
            hovertemplate="Remaining:<br>%{customdata[0]:,.0f} / %{customdata[1]:,.0f} tokens<br><b>%{x:0.0%}</b><br><extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        showlegend=True,
        xaxis=dict(visible=False),
        hoverlabel=dict(font_size=13),
        legend=dict(
            orientation="h",
            traceorder="normal",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=-0.02,
        ),
    )

    return fig


# endregion


# region: === CSS Customization ===

# CSS Selectors (for CSS and JS hacks)
# NOTE: Be careful to use ' (NOT ") within the selectors if the selectors are being used in JS executions, to ensure
# string literals are properly handled.
SELECTOR_MAIN_SPACE = "section.stMain > div.block-container > div[data-testid='stVerticalBlockBorderWrapper'] > div > div[data-testid='stVerticalBlock'] > div:nth-child(4)"
SELECTOR_CHAT_HISTORY_CONTAINER = f"{SELECTOR_MAIN_SPACE} > div:nth-child(1) > div > div > div > div:nth-child(1)"

SELECTOR_TAB_MENU_CONTAINER = f"{SELECTOR_MAIN_SPACE} > div:nth-child(2) > div > div > div > div:nth-child(1)"
SELECTOR_TAB_MENU = f"{SELECTOR_TAB_MENU_CONTAINER} > iframe"

SELECTOR_TAB_CONTENT_AREA = f"{SELECTOR_MAIN_SPACE} > div:nth-child(2) > div > div > div > div:nth-child(2)"
SELECTOR_TAB_JS_CONTAINER = f"{SELECTOR_TAB_CONTENT_AREA} > div > div > div[data-testid='stVerticalBlockBorderWrapper']"

SELECTOR_TAB_DEBUG = f"{SELECTOR_TAB_CONTENT_AREA}.tab_debug"
SELECTOR_TAB_MESSAGE_TREE = f"{SELECTOR_TAB_CONTENT_AREA}.tab_message_tree"
SELECTOR_TAB_SESSION = f"{SELECTOR_TAB_CONTENT_AREA}.tab_session"
SELECTOR_TAB_WD = f"{SELECTOR_TAB_CONTENT_AREA}.tab_wd"
SELECTOR_TAB_PLAN = f"{SELECTOR_TAB_CONTENT_AREA}.tab_plan"
SELECTOR_TAB_TRANSFORMS = f"{SELECTOR_TAB_CONTENT_AREA}.tab_transforms"

SELECTOR_TAB_WD_FILTER_LABEL = f"{SELECTOR_TAB_WD} div.stMultiSelect:nth-child(1) > label"
SELECTOR_TAB_DEBUG_UICOMM = (
    f"{SELECTOR_TAB_DEBUG} div[data-testid='stExpander'] div[data-testid='stExpanderDetails'] code"
)
SELECTOR_TAB_DEBUG_MSG = (
    f"{SELECTOR_TAB_DEBUG} div[data-testid='stExpander'] div[data-testid='stExpanderDetails'] div.object-container"
)
SELECTOR_TAB_SESSION_TOKEN_NOTE = f"{SELECTOR_TAB_SESSION} > div > div[data-testid='stMarkdown']"

SELECTOR_SCROLL_JS_CONTAINER = f"{SELECTOR_MAIN_SPACE} > div:nth-child(1) > div > div > div > div:nth-child(3)"

# ---

CHAT_HISTORY_CONTAINER_OFFSET = 200
TAB_MENU_BOTTOM_ADJUSTMENT = 25
TAB_TOP_OFFSET = CHAT_HISTORY_CONTAINER_OFFSET - TAB_MENU_BOTTOM_ADJUSTMENT

# --- --- ---


def run_markdown_css_hack() -> None:
    # NOTE: Custom css hack.

    # This code has a workaround to completely "disappear" the stale messages from the chat.
    # It seems the "stale" message approach is used by streamlit to avoid excessive scroll jumping, so making them
    # invisible is problematic. However, to balance this with avoiding confusion of the user with duplicate messages,
    # we just fade them out to complete invisibility.

    # It also includes a hack to fix the chat box height to be a certain vertical screen height.

    REPLACE_IN_STYLE_STR = {
        "{SELECTOR_CHAT_HISTORY_CONTAINER}": SELECTOR_CHAT_HISTORY_CONTAINER,
        "{CHAT_HISTORY_CONTAINER_OFFSET}": CHAT_HISTORY_CONTAINER_OFFSET,
        "{SELECTOR_TAB_MENU_CONTAINER}": SELECTOR_TAB_MENU_CONTAINER,
        "{SELECTOR_TAB_MENU}": SELECTOR_TAB_MENU,
        "{SELECTOR_TAB_CONTENT_AREA}": SELECTOR_TAB_CONTENT_AREA,
        "{SELECTOR_TAB_JS_CONTAINER}": SELECTOR_TAB_JS_CONTAINER,
        "{SELECTOR_TAB_DEBUG}": SELECTOR_TAB_DEBUG,
        "{SELECTOR_TAB_SESSION}": SELECTOR_TAB_SESSION,
        "{SELECTOR_TAB_WD}": SELECTOR_TAB_WD,
        "{SELECTOR_TAB_PLAN}": SELECTOR_TAB_PLAN,
        "{SELECTOR_TAB_TRANSFORMS}": SELECTOR_TAB_TRANSFORMS,
        "{TAB_TOP_OFFSET}": TAB_TOP_OFFSET,
        "{SELECTOR_TAB_WD_FILTER_LABEL}": SELECTOR_TAB_WD_FILTER_LABEL,
        "{SELECTOR_TAB_DEBUG_UICOMM}": SELECTOR_TAB_DEBUG_UICOMM,
        "{SELECTOR_TAB_DEBUG_MSG}": SELECTOR_TAB_DEBUG_MSG,
        "{SELECTOR_TAB_SESSION_TOKEN_NOTE}": SELECTOR_TAB_SESSION_TOKEN_NOTE,
        "{SELECTOR_SCROLL_JS_CONTAINER}": SELECTOR_SCROLL_JS_CONTAINER,
        "{TAB_MENU_BOTTOM_ADJUSTMENT}": TAB_MENU_BOTTOM_ADJUSTMENT,
    }
    style_str = """
    <style>
    .block-container
    {
        padding-top: 1rem;
        padding-bottom: 0rem;
        margin-top: 1rem;
    }
    
    h3:nth-child(1) {
        line-height: 1.8;
    }

    /* A hack to dim out the stale messages _completely_. */
    div .stChatMessage [data-stale="true"] {
        opacity: 0.0;
    }

    /* Chat box, make it fixed height. */
    section.main > div.block-container {
        height: 100vh;
    }
    {SELECTOR_CHAT_HISTORY_CONTAINER} {
        /* DEBUG: Uncomment below and check the container is highlighted in red. */
        /* background-color: red; */

        height: calc(100vh - {CHAT_HISTORY_CONTAINER_OFFSET}px);
        scroll-behavior: auto !important;
    }

    {SELECTOR_TAB_MENU_CONTAINER} {
        /*background-color: purple;*/
        margin-bottom: -{TAB_MENU_BOTTOM_ADJUSTMENT}px;
    }
    {SELECTOR_TAB_MENU} {
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
    }

    /* Fix max height - RHS column */
    {SELECTOR_TAB_CONTENT_AREA} {
        /*background-color: green;*/
        max-height: calc(100vh - {TAB_TOP_OFFSET}px);
        overflow-y: scroll;
        overflow-x: hidden;
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
        margin-top: 0;
    }
    {SELECTOR_TAB_JS_CONTAINER} {
        display: none;
    }

    {SELECTOR_TAB_SESSION} {
        /*background-color: purple;*/
    }
    {SELECTOR_TAB_DEBUG} {
        /*background-color: blue;*/
    }
    {SELECTOR_TAB_WD} {
        /*background-color: yellow;*/
    }
    {SELECTOR_TAB_PLAN} {
        /*background-color: orange;*/
    }
    {SELECTOR_TAB_TRANSFORMS} {
        /*background-color: green;*/
    }

    /* Hide the label of WD filter. */
    {SELECTOR_TAB_WD_FILTER_LABEL} {
        /*background-color: red;*/
        display: none;
    }

    {SELECTOR_TAB_DEBUG_UICOMM} {
        /*background-color: pink;*/
        font-size: 9pt;
        line-height: 1.4;
        display: block;
    }
    {SELECTOR_TAB_DEBUG_MSG} {
        /*background-color: purple;*/
        font-size: 9pt;
        line-height: 1.3;
    }
    {SELECTOR_TAB_SESSION_TOKEN_NOTE} {
        /*background-color: red;*/
        line-height: 1;
    }
    {SELECTOR_TAB_SESSION_TOKEN_NOTE} p {
        margin-bottom: 0.75em;
    }

    /* JS container - hide */
    {SELECTOR_SCROLL_JS_CONTAINER} {
        /*background-color: red;*/
        display: none;
    }

    </style>
    """
    style_str = replace_str_from_dict(style_str, REPLACE_IN_STYLE_STR)
    st.markdown(style_str, unsafe_allow_html=True)


# endregion


# region: === Streamlit app layout and simple flow (including message history) ===

st.set_page_config(page_title=st_common.TITLE, layout="wide", initial_sidebar_state="expanded")

run_markdown_css_hack()

st_common.menu()

top_col_title, top_col_buttons = st.columns([3, 9])

with top_col_title:
    st.markdown(f"### {st_common.TITLE}: `{st_common.VERSION}`", unsafe_allow_html=True)

modal_disclaimer_title = "ℹ️ Disclaimer"
modal_settings_title = "⚙️ Settings"


def rerun_with_state(state: Optional[InteractionStage] = None) -> None:
    # Setting state argument to None will just rerun the UI in the present interaction stage.
    if state is None:
        state = engine().get_state().ui_controlled.interaction_stage
    engine().get_state().ui_controlled.interaction_stage = state
    engine().update_state()
    st.rerun()


@st.dialog(modal_disclaimer_title, width="large")
def modal_disclaimer():
    st.markdown(
        DISCLAIMER_TEXT.replace("# Disclaimer:\n", ""),  # Remove the title line, as it's already in the modal title.
    )
    user_settings.disclaimer_shown = True
    db.update_user_settings(user_settings)
    # Add an "Accept" button to close the modal.
    if st.button("Accept", type="primary"):
        st.rerun()


@st.dialog(modal_settings_title)
def modal_settings():
    show_tool_call_logs = st.checkbox(
        st_common.COMMON_SETTINGS["display"]["show_tool_call_logs"]["name"],
        value=engine().session.session_settings.show_tool_call_logs,
        help=st_common.COMMON_SETTINGS["display"]["show_tool_call_logs"]["help"],
    )
    show_tool_call_return = st.checkbox(
        st_common.COMMON_SETTINGS["display"]["show_tool_call_return"]["name"],
        value=engine().session.session_settings.show_tool_call_return,
        help=st_common.COMMON_SETTINGS["display"]["show_tool_call_return"]["help"],
    )
    show_code = st.checkbox(
        st_common.COMMON_SETTINGS["display"]["show_code"]["name"],
        value=engine().session.session_settings.show_code,
        disabled=False,
        help=st_common.COMMON_SETTINGS["display"]["show_code"]["help"],
    )
    show_code_out = st.checkbox(
        st_common.COMMON_SETTINGS["display"]["show_code_out"]["name"],
        value=engine().session.session_settings.show_code_out,
        help=st_common.COMMON_SETTINGS["display"]["show_code_out"]["help"],
    )
    show_full_message_history = st.checkbox(
        st_common.COMMON_SETTINGS["display"]["show_full_message_history"]["name"],
        value=engine().session.session_settings.show_full_message_history,
        help=st_common.COMMON_SETTINGS["display"]["show_full_message_history"]["help"],
    )
    show_message_history_length = int(
        st.number_input(  # type: ignore
            st_common.COMMON_SETTINGS["display"]["show_message_history_length"]["name"],
            value=engine().session.session_settings.show_message_history_length,
            help=st_common.COMMON_SETTINGS["display"]["show_message_history_length"]["help"],
            min_value=st_common.COMMON_SETTINGS["display"]["show_message_history_length"]["min_value"],
            max_value=st_common.COMMON_SETTINGS["display"]["show_message_history_length"]["max_value"],
            step=st_common.COMMON_SETTINGS["display"]["show_message_history_length"]["step"],
        )
    )
    with st.expander("🛑 Advanced settings", expanded=False):
        st.markdown("""
            **Only modify these settings if you know what you are doing.**
            """)
        show_planning_details = st.checkbox(
            st_common.COMMON_SETTINGS["display"]["show_planning_details"]["name"],
            value=engine().session.session_settings.show_planning_details,
            help=st_common.COMMON_SETTINGS["display"]["show_planning_details"]["help"],
        )
    # st.markdown("---")
    # ---
    st.markdown(
        """
        Please make sure to press `Save settings` for the changes to apply.
        > ⚠️ This will cancel any ongoing tools, code, or messages that haven't completed. 
        > The page will then be **reloaded**.
        """
    )
    save = st.button("Save settings", type="primary")
    if save:
        session_settings = engine().session.session_settings
        session_settings.show_code = show_code
        session_settings.show_tool_call_logs = show_tool_call_logs
        session_settings.show_tool_call_return = show_tool_call_return
        session_settings.show_code_out = show_code_out
        session_settings.show_planning_details = show_planning_details
        session_settings.show_full_message_history = show_full_message_history
        session_settings.show_message_history_length = show_message_history_length
        db.update_session(engine().session)
        rerun_with_state(state=None)


with top_col_buttons:
    st.markdown("")  # Dummy space.
    c1, c2, _ = st.columns([1.5, 1.5, 9])

    with c1:
        if user_settings.disclaimer_shown is False or st.button(modal_disclaimer_title, type="primary"):
            modal_disclaimer()
    with c2:
        if st.button(modal_settings_title):
            modal_settings()

main_col_1, main_col_2 = st.columns([6.5, 5.5])

with main_col_1:
    container_chat = st.container(height=500)  # NOTE: Height is a dummy value, it's set by CSS hack.
    container_msg_input = st.container()  # A container for the user chat input, to ensure it sticks to the bottom.
    container_scroll_js = st.container(height=0)

with main_col_2:
    tab_labels = []
    tab_names_map = {
        "tab_debug": "🐛 DEBUG",
        "tab_message_tree": "🌳 Message Tree",
        "tab_session": "📘 Session",
        "tab_wd": "📂 Working directory",
        "tab_plan": "📜 Plan",
        "tab_transforms": "🛢 Data transformations",
    }
    tab_names_map_rev = {v: k for k, v in tab_names_map.items()}
    if DEBUG_PANEL:
        tab_labels += [
            sac.ButtonsItem(label=tab_names_map["tab_debug"]),
            sac.ButtonsItem(label=tab_names_map["tab_message_tree"]),
        ]
    tab_labels += [
        sac.ButtonsItem(label=tab_names_map["tab_session"]),
        sac.ButtonsItem(label=tab_names_map["tab_wd"]),
        sac.ButtonsItem(label=tab_names_map["tab_plan"]),
        sac.ButtonsItem(label=tab_names_map["tab_transforms"]),
    ]
    active_tab_names = [tab_names_map_rev[x.label] for x in tab_labels]
    active_tab_buttons_out = sac.buttons(
        tab_labels,
        label="",
        align="start",
        size="xs",
        radius="xs",
        gap="xs",
        index=active_tab_names.index(st.session_state.active_tab),
    )
    st.session_state.active_tab = tab_names_map_rev[active_tab_buttons_out]  # type: ignore

    with st.container():
        container_tab_js = st.container(height=0)
        js_tabs = st_common.JSExecutor(
            js_code=dedent("""
                function addClassToElement(selector) {
                    try {
                        // Find the first element matching the provided CSS selector
                        const element = parent.document.querySelector(selector);
                        if (element) {
                            // Remove any existing classes that start with 'tab_'
                            element.classList.forEach(cls => {
                                if (cls.startsWith("tab_")) {
                                    element.classList.remove(cls);
                                }
                            });

                            // Add the class 'tab_foo' to the element
                            element.classList.add("<tab_name>");
                            return "success";
                        } else {
                            return "fail_no_element";
                        }
                    } catch (error) {
                        console.error('Error occurred:', error);
                        return "fail_error";
                    }
                }

                return addClassToElement("<selector>");
                """),
            container=container_tab_js,
            use_st_js=True,
            log_text="Active tab switched.",
        )
        js_tabs.execute_js(
            replacements={"<tab_name>": st.session_state.active_tab, "<selector>": SELECTOR_TAB_CONTENT_AREA}
        )

        if st.session_state.active_tab == "tab_debug":
            add_tab_warning_area("tab_debug")
            expander_uicomm = st.expander("`EngineState`", expanded=True)
            with expander_uicomm:
                cont_uicomm = st.empty()
            update_debug_panel_ui_comm()

            col1, col2 = st.columns([0.3, 0.7])
            cont_messages = st.expander("Messages", expanded=True)
            with col1:
                show_all_msg = st.checkbox("Show all messages", value=False, key="show_all_msg")
                reverse_order = st.checkbox("Reverse message order", value=True, key="reverse_order")
            with col2:
                shown_n_msg = st.number_input(
                    "Number of messages to show",
                    value=5,
                    min_value=1,
                    max_value=999,
                    step=1,
                    key="shown_n_msg",
                )
            with cont_messages:
                messages = engine().get_message_history()
                messages = [m.model_dump(by_alias=True) for m in messages]
                if reverse_order:
                    messages = messages[::-1]
                if not show_all_msg:
                    messages = messages[:shown_n_msg]
                st.write(messages)

        elif st.session_state.active_tab == "tab_message_tree":
            add_tab_warning_area("tab_message_tree")
            update_debug_panel_ui_comm()
            st.code(
                engine().session.messages.format(repr="{node.data.role}-{node.data.key}", style="round43c", join="\n")
            )

        elif st.session_state.active_tab == "tab_session":
            add_tab_warning_area("tab_session")

            with st.expander("**Session details**", icon="ℹ️", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    Session settings:
                    - Name: `{engine().session.friendly_name}`
                    - Started at: `{engine().session.started_at.strftime("%Y-%m-%d %H:%M:%S")}`
                    - Engine: `{engine().session.engine_name}`
                    """)
                with col2:
                    col2_txt = "Engine parameters:\n"
                    for param in engine().engine_params:
                        col2_txt += f"- `{param}`: `{engine().engine_params[param]}`\n"
                    st.markdown(col2_txt)

            context_size = MODEL_CONTEXT_SIZE[engine().engine_params["model_id"]]  # type: ignore
            token_counts = engine().get_token_counts()
            token_counts_show = dict()
            for key, value in token_counts.items():
                token_counts_show[key] = f"{value}/{context_size}"

            st.markdown("")  # Spacer.
            st.markdown("##### Token usage:")
            # st.write(token_counts_show)
            fig = plot_proportion(token_counts, context_size)
            st.plotly_chart(fig, config={"displayModeBar": False})
            # The config={"displayModeBar": False} above:
            # https://discuss.streamlit.io/t/removing-plotly-mode-bar-from-streamlit-plots/18114

            st.markdown(
                dedent("""
                **Note:**
                - The token counts may be estimated, so may not be exact.
                - The longer the conversation history gets, the more tokens are used.
                """)
            )

            # Add a streamlit button
            st.markdown("")  # Spacer.
            st.markdown("##### Report:")
            if st.button("🖨️ Generate session report", key="gen_report_btn_1"):
                if WEASYPRINT_WORKING:
                    with st.spinner("Generating report..."):
                        report_path = prepare_report(working_dir=engine().working_directory_abs)
                        with open(report_path, "rb") as file:
                            btn = st.download_button(
                                label="⬇️ Download report",
                                data=file,
                                file_name=os.path.basename(report_path),
                                mime="text/pdf",
                                type="primary",
                                key="download_report_btn_1",
                            )
                else:
                    st.warning(
                        "WeasyPrint import failed. Please check the troubleshooting section in the documentation.\n\n"
                        "---\n\n" + WEASYPRINT_WARNING_FULL.replace("\n", "\n\n")
                    )

            # Bottom spacer.
            for _ in range(2):
                st.markdown("")

        elif st.session_state.active_tab == "tab_wd":
            add_tab_warning_area("tab_wd")
            FILE_CATEGORY_EMOJI_MAP = {
                "data": "🛢",
                "image": "🖼️",
                "model": "🧠",
                "other": "📦",
            }
            FILE_CATEGORY_EMOJI_UNKNOWN = "❓"

            file_infos = engine().describe_working_directory_list()
            files_df = pd.DataFrame(
                [
                    {
                        "category_emoji": FILE_CATEGORY_EMOJI_MAP.get(file_info.category, FILE_CATEGORY_EMOJI_UNKNOWN),
                        "name": file_info.name,
                        "size": f"{file_info.size:3.1f} {file_info.size_units}",
                        "category": file_info.category,
                        "previewable": file_info.previewable,
                        "modified": file_info.modified.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for file_info in file_infos
                ]
            )
            files_in_wd = len(files_df) > 0

            if files_in_wd:
                with st.expander("Working directory contents", expanded=True):
                    col_wdc_1, col_wdc_2 = st.columns([5, 7])
                    with col_wdc_1:
                        st.markdown("⤵ Select a row to preview the file.")
                    with col_wdc_2:
                        cat_selected = st.multiselect(
                            label="Filter by category",
                            options=[
                                FILE_CATEGORY_EMOJI_MAP.get(x, FILE_CATEGORY_EMOJI_UNKNOWN) + f" {x}"
                                for x in FILE_CATEGORY_EMOJI_MAP.keys()
                            ],
                            placeholder="Select categories",
                        )
                    cat_selected = [x.split()[-1] for x in cat_selected]
                    # st.write(cat_selected)

                    # Pre-process the data itself:
                    files_df = files_df.sort_values(by="modified", ascending=False)
                    if cat_selected:
                        files_df = files_df[files_df["category"].isin(cat_selected)]
                    # print(files_df)

                    # Show the streamlit table:
                    files_table = st.dataframe(
                        files_df,
                        column_config={
                            "category_emoji": st.column_config.TextColumn(
                                label="ⓘ",
                                help=(
                                    "An icon representing the file category. "
                                    "🛢: data, 🖼️: image, 🧠: model, 📦: other."
                                ),
                                width=35,  # type: ignore
                            ),
                            "name": "Name",
                            "size": "Size",
                            "category": None,
                            "previewable": None,
                            "modified": st.column_config.DatetimeColumn(
                                "Last Modified",
                                format="D MMM, h:mm a",
                            ),
                        },
                        hide_index=True,
                        key="data",
                        on_select="rerun",
                        selection_mode="single-row",
                        height=300,
                    )

                # Select the file and show the preview if available.
                selected_row_idx = files_table.selection["rows"][0] if files_table.selection["rows"] else None
                if selected_row_idx is None:
                    st.markdown("No file selected.")
                    st.markdown("")
                else:
                    file = files_df.iloc[selected_row_idx, :]["name"]
                    file = [f for f in file_infos if f.name == file][0]
                    if file.previewable:
                        with st.expander(f"Preview of: `{file.name}`", expanded=True):
                            if file.category == "image":
                                col_temp, _ = st.columns([0.95, 0.05])
                                with col_temp:
                                    st.image(os.path.join(engine().working_directory_abs, file.name))
                            elif file.category == "data":
                                show_all = st.checkbox("Show all data (⚠️ can be slow)", value=False, key="show_preview")
                                try:
                                    df = pd.read_csv(os.path.join(engine().working_directory_abs, file.name))
                                    df_read_success = True
                                except Exception:
                                    df_read_success = False
                                if df_read_success:
                                    if not show_all:
                                        df = df.head()
                                    st.dataframe(df)
                                else:
                                    st.markdown("⛔ Failed to read this data file.")
                        st.markdown("")
                    else:
                        cat_ = (
                            FILE_CATEGORY_EMOJI_MAP.get(file.category, FILE_CATEGORY_EMOJI_UNKNOWN)
                            + " "
                            + file.category
                        )
                        st.markdown(f"Preview not available for file category: `{cat_}`")
                        st.markdown("")

            else:
                st.markdown("`No files in the working directory at the moment.`")

        elif st.session_state.active_tab == "tab_plan":
            add_tab_warning_area("tab_plan")
            # st.markdown("🚧 Work in progress 🚧")
            plan_list_of_dicts = engine().get_current_plan()
            if plan_list_of_dicts is None:
                st.markdown("*No plan found.*")
            else:
                df_plan = df_from_plan(plan_list_of_dicts)

                def style_by_status(s):
                    MAP = {
                        "not_started": ["background-color: rgb(0,0,0,0);"],  # Transparent.
                        "in_progress": ["background-color: DarkYellow;"],
                        "completed": ["background-color: DarkGreen;"],
                        "needs_redoing": ["background-color: DarkRed;"],
                    }
                    return MAP.get(s[0], [])

                df_styled = df_plan.style.apply(style_by_status, subset=["Status"], axis=1)

                # Show the plan in a table.
                st.dataframe(
                    df_styled,
                    hide_index=True,
                    height=790,
                    column_config={
                        "Project Stage": st.column_config.TextColumn(
                            width=135,  # type: ignore
                        ),
                        "Task": st.column_config.TextColumn(
                            width=195,  # type: ignore
                        ),
                        "Subtask": st.column_config.TextColumn(
                            width=240,  # type: ignore
                        ),
                        "Status": st.column_config.TextColumn(
                            width=104,  # type: ignore
                        ),
                    },
                )

        elif st.session_state.active_tab == "tab_transforms":
            add_tab_warning_area("tab_transforms")
            col_spacer, _ = st.columns([0.95, 0.05])
            with col_spacer:
                # st.markdown("🚧 Work in progress 🚧")

                msg_with_files = engine().get_message_history()
                msg_with_files = [m for m in msg_with_files if m.files_in is not None and m.files_out is not None]
                msg_with_files = [m for m in msg_with_files if len(m.files_in) == 1 and len(m.files_out) == 1]  # type: ignore
                msg_transforms = [m for m in msg_with_files if ".csv" in m.files_in[0] and ".csv" in m.files_out[0]]  # type: ignore
                wd = engine().working_directory_abs
                # Discard files if they don't exist.
                files = []
                for m in msg_transforms:
                    data = {"key": m.key}
                    data["in"] = [f for f in m.files_in if os.path.exists(os.path.join(wd, f))]  # type: ignore
                    data["out"] = [f for f in m.files_out if os.path.exists(os.path.join(wd, f))]  # type: ignore
                    if data["in"] and data["out"]:
                        files.append(data)
                df_transforms = pd.DataFrame({"Input file": [], " ": "→", "Output file": []})
                for f in files:
                    df_transforms = pd.concat(
                        [
                            df_transforms,
                            pd.DataFrame(
                                {
                                    "Input file": ", ".join(f["in"]),  # type: ignore
                                    " ": "→",
                                    "Output file": ", ".join(f["out"]),  # type: ignore
                                },
                                index=[0],
                            ),
                        ]
                    )
                    df_transforms = df_transforms[::-1]
                if len(df_transforms) > 0:
                    st.markdown(
                        dedent("""
                        ##### Data transformations found:
                        - Shown most recent first.
                        - Select a transformation to preview the changes.
                        """)
                    )
                    transforms_table = st.dataframe(
                        df_transforms,
                        hide_index=True,
                        # key="data",
                        on_select="rerun",
                        selection_mode="single-row",
                    )
                    # st.write(transforms_table)
                    if transforms_table.selection["rows"]:  # type: ignore
                        selected_row_idx = transforms_table.selection["rows"][0]  # type: ignore
                        file_in = df_transforms.iloc[selected_row_idx, :]["Input file"]
                        file_out = df_transforms.iloc[selected_row_idx, :]["Output file"]
                        # st.write(file_in, file_out)
                        loaded = False
                        try:
                            df_in = pd.read_csv(os.path.join(wd, file_in))
                            df_out = pd.read_csv(os.path.join(wd, file_out))
                            loaded = True
                        except Exception as e:
                            st.markdown("⛔ Unable to load the data files.")
                            ui_log("Failed to load data files for transformation preview:", e)

                        if loaded:
                            st.markdown(
                                dedent("""
                                ##### Data transformation preview:
                                - This is the *best guess* as to what was modified in the data transformations (there is always ambiguity in determining removed/added rows vs modified values).
                                - The comparison is made on a subset of rows only (set this below).
                                - Selecting more than a few 10s of rows is likely to be *very slow*.
                                """)
                            )

                            max_rows = max(len(df_in), len(df_out))
                            use_n_rows = st.number_input(
                                "Number of rows to compare:",
                                value=15,
                                min_value=1,
                                max_value=max_rows,
                                step=1,
                            )

                            df_in = df_in[:use_n_rows]
                            df_out = df_out[:use_n_rows]

                            with st.spinner("Comparing data..."):
                                changes_info = analyze_df_modifications(df_in, df_out, row_similarity_threshold=0.8)
                            # st.write(changes_info)

                            if not changes_info["success"]:
                                st.markdown("⛔ Unable to compare the data files.")
                            else:

                                def highlight_columns(s, cols):
                                    return ["background-color: DarkGreen" if s.name in cols else "" for _ in s]

                                # Function to highlight entire rows
                                def highlight_rows(s, rows):
                                    return ["background-color: DarkGreen" if s.name in rows else "" for _ in s.index]

                                # Function to highlight specific cells
                                def highlight_specific_cells(df, cells):
                                    df = df.copy()
                                    df.loc[:, :] = None
                                    for row, col, _ in cells:
                                        df.loc[row, col] = "background-color: DarkGreen"
                                    return df

                                # Replace the modified values with the "A -> B" change strings.
                                df_out = df_out.head(use_n_rows)  # type: ignore
                                for row, col, change in changes_info["modified_values"]:
                                    df_out.loc[row, col] = change

                                # Applying styles
                                styled_df = (
                                    df_out.style.apply(highlight_columns, cols=changes_info["columns_added"], axis=0)
                                    .apply(highlight_rows, rows=changes_info["rows_added"], axis=1)
                                    .apply(highlight_specific_cells, cells=changes_info["modified_values"], axis=None)
                                )

                                add_mod_true = bool(
                                    changes_info["columns_added"]
                                    or changes_info["rows_added"]
                                    or changes_info["modified_values"]
                                )
                                with st.expander(
                                    f"**Additions and modifications{' [None]' if not add_mod_true else ''}**",
                                    expanded=add_mod_true,
                                ):
                                    if not add_mod_true:
                                        st.markdown("ℹ️ No added columns, rows, or modified values found.")
                                    else:
                                        st.markdown(
                                            "ℹ️ Modified values or added rows/columns are :green-background[highlighted]."
                                        )
                                    st.dataframe(styled_df)

                                col_rm_true = bool(changes_info["columns_removed"])
                                with st.expander(
                                    f"**Removed columns{' [None]' if not col_rm_true else ''}**", expanded=col_rm_true
                                ):
                                    if col_rm_true:
                                        st.markdown(
                                            f"ℹ️ Number of columns removed: `{len(changes_info['columns_removed'])}`."
                                        )
                                        removed_columns_vis_mode = st.radio(
                                            "Show as:",
                                            ["List", "Dataframe"],
                                            horizontal=True,
                                        )
                                        if removed_columns_vis_mode == "Dataframe":
                                            df_cols_removed = df_in[changes_info["columns_removed"]].style.apply(
                                                lambda x: ["background-color: DarkRed"] * len(df_in), axis=0
                                            )
                                            # print(df_cols_removed)
                                            st.dataframe(df_cols_removed)
                                        else:
                                            removed_columns_bp = "\n" + "".join(
                                                [f"* `{c}`\n" for c in changes_info["columns_removed"]]
                                            )
                                            rm_cols_cont_ht = min(300, 30 + 30 * len(changes_info["columns_removed"]))
                                            with st.container(height=rm_cols_cont_ht):
                                                st.markdown(removed_columns_bp)
                                    else:
                                        st.markdown("ℹ️ No columns were removed.")

                                row_rm_true = bool(changes_info["rows_removed"])
                                with st.expander(
                                    f"**Removed rows{' [None]' if not row_rm_true else ''}**", expanded=row_rm_true
                                ):
                                    if row_rm_true:
                                        st.markdown(f"ℹ️ Number of rows removed: `{len(changes_info['rows_removed'])}`.")
                                        df_rows_removed = df_in.loc[changes_info["rows_removed"]].style.apply(
                                            lambda x: ["background-color: DarkRed"] * len(df_in.columns), axis=1
                                        )
                                        st.dataframe(df_rows_removed)
                                    else:
                                        st.markdown("ℹ️ No rows were removed.")
                    else:
                        st.markdown("`No data transformation selected.`")
                else:
                    st.markdown("`No data transformations found.`")

show_tab_warnings_if_tool_running()

# Print message history.
scroller_js = """
function scrollToBottom(selector) {
    // Find the element using the provided CSS selector
    const element = parent.document.querySelector(selector);

    if (element) {
        // Set the scrollTop property to the scrollHeight to scroll to the bottom
        element.scrollTop = element.scrollHeight;
        /*console.log('Scrolled to the bottom of the element with selector:', selector);*/
    } else {
        console.error('Element not found with selector:', selector);
    }
}

// Usage example: scroll to the bottom of the element with the class 'content'
scrollToBottom("<selector>");
"""
scroller = st_common.JSExecutor(
    js_code=scroller_js,
    container=container_scroll_js,
    log_text=None,  # "Scroll to bottom of messages container triggered.",
)

with container_chat:
    history = engine().get_message_history()
    history = [m for m in history if show_in_history_if(m)]
    # Do not show the messages that have incoming tool calls.
    # Show the messages that are allowed according to their role and visibility.

    show_message_history_length = engine().session.session_settings.show_message_history_length
    if engine().session.session_settings.show_full_message_history:
        show_message_history_length = len(history)

    if len(history) > show_message_history_length:
        history = history[-show_message_history_length:]
        ui_log("Trimming message history to show only the last", show_message_history_length, "messages.")
        st.markdown(
            "... **⚙️ Earlier messages were hidden for UI speed. "
            f"Showing only the last `{show_message_history_length}` messages.** ..."
        )
    # --- --- ---

    for message in history:
        message = cast(Message, message)

        with st.chat_message(name=message.role, avatar=AVATAR_MAP[message.role]):
            # 1. Show the message.text.
            message_text = message.text or ""

            message_agent = message.agent
            if DEBUG__SHOW_AGENT_IN_HISTORY and message.role != "user":
                agent_text_color = AGENT_TEXT_COLOR_MAP.get(message_agent, "black")
                agent_indicator = f"**Agent: :{agent_text_color}-background[{message_agent}]**\n\n"
            else:
                agent_indicator = ""
            message_text = agent_indicator + message_text

            # 1-A: Handle code blocks (can result in multiple text sections)
            pre_code, code_block, post_code = id_and_excise_code(message_text)
            text_sections: List[Tuple[Literal["text", "code"], str]] = [("text", pre_code)]
            if code_block is not None:
                text_sections.append(("code", code_block))
                if post_code is not None:
                    text_sections.append(("text", post_code))

            # 1-B: Any other post-processing, applied to each section individually.
            for text_or_code, content in text_sections:
                # Any text post-processing:
                if text_or_code == "text":
                    # Handle markdown (embedded) images:
                    forces_unsafe = False

                    # This section contains currently fairly hacky implementation of various message history
                    # post-processings.
                    # TODO: Make this systematic. Coordinate with in-stream filtering too.
                    # - agent coordination special value replacements.
                    if "TASK COMPLETED" in content:
                        content = content.replace("TASK COMPLETED", "✅🎉 Task completed!")
                    if "= CONTINUE =" in content:
                        content = content.replace("= CONTINUE =", "▶️")
                    if (
                        engine().session.session_settings.show_planning_details is False
                        and message.agent == "coordinator"
                    ):
                        ui_log(f"PLANNING:\n{content}")
                        content = "💡 `Planning step`"
                        # content[: content.index("SYSTEM:")] + "💡 `Planning completed`"
                    # - <WD>/ image replacement.
                    if "<WD>/" in content:
                        content = process_md_images(content, wd=engine().working_directory)  # HACK # type: ignore
                        forces_unsafe = True
                    # --- --- ---

                    st.markdown(content, unsafe_allow_html=forces_unsafe)
                # Any code post-processing:
                elif text_or_code == "code":
                    with st.expander(CODE_ITSELF_PREFIX, expanded=engine().session.session_settings.show_code):
                        st.markdown(content)
                else:
                    raise ValueError(f"Unexpected `text_or_code` value: {text_or_code}")

            # 2. Show various tool call outputs.
            if message.role == "tool":
                # Validate.
                # TODO: Move this logic to `Message` eventually.
                if message.outgoing_tool_call is None:
                    raise ValueError("`'tool'` message without `outgoing_tool_call`")
                if message.tool_call_success is None:
                    raise ValueError("`'tool'` message without `tool_call_success`")
                if message.tool_call_return is None:
                    raise ValueError("`'tool'` message without `tool_call_return`")
                # --- --- ---

                top_text = f"Tool `{message.outgoing_tool_call.name}`"
                if message.tool_call_success:
                    top_text += " completed successfully ✅"
                else:
                    top_text += " failed ❌"
                st.markdown(top_text)

                # Tool call logs.
                if message.tool_call_logs:
                    with st.expander(TOOL_LOGS_PREFIX, expanded=engine().session.session_settings.show_tool_call_logs):
                        st.markdown(f"```\n{message.tool_call_logs}\n```")

                # Tool call return.
                if message.tool_call_return:
                    with st.expander(
                        TOOL_RETURN_PREFIX, expanded=engine().session.session_settings.show_tool_call_return
                    ):
                        st.markdown(f"```\n{message.tool_call_return}\n```")

                # User-only report outputs.
                if message.tool_call_user_report:
                    with st.expander(TOOL_USER_REPORT_PREFIX, expanded=True):
                        for user_output in message.tool_call_user_report:  # pyright: ignore
                            if isinstance(user_output, str):
                                st.markdown(user_output)
                            elif isinstance(user_output, go.Figure):
                                st.plotly_chart(user_output)
                            elif isinstance(user_output, matplotlib.figure.Figure):
                                st.pyplot(user_output)
                            else:
                                raise ValueError(f"Unexpected user output type: {type(user_output)}")

            # 3. Show code execution outputs.
            if message.role == "code_execution":
                # Validate.
                if message.generated_code_success is None:
                    raise ValueError("`'code_execution'` message without `generated_code_success`")
                # --- --- ---

                top_message = (
                    "Code execution finished successfully ✅"
                    if message.generated_code_success
                    else "Code execution failed ❌"
                )
                st.markdown(top_message)
                with st.expander(CODE_EXECUTION_OUT_PREFIX, expanded=engine().session.session_settings.show_code_out):
                    output_message = ""
                    if message.generated_code_stdout:
                        output_message += f"```\n{message.generated_code_stdout}\n```\n"
                    # NOTE: Should we show stderr regardless of the show logs setting?
                    if message.generated_code_stderr:
                        output_message += f"\n**Error:**\n\n```\n{message.generated_code_stderr}\n```"
                    st.markdown(output_message)

            if message.role == "user":
                if st.button("🖉", key=f"btn_{message.key}", help="Restart from here and edit"):
                    success = engine().restart_at_user_message(message.key)
                    if success:
                        rerun_with_state(state=None)
                    else:
                        ui_log("Restart at user message failed.")

            scroller.execute_js(replacements={"<selector>": SELECTOR_CHAT_HISTORY_CONTAINER})


if DEBUG__SHOW_ACTIVE_AGENT:
    # Show currently active agent.
    with container_chat:
        agent_text_color = AGENT_TEXT_COLOR_MAP.get(message_agent, "white")
        st.markdown(f"**ℹ️ Active agent: :{agent_text_color}-background[{active_agent}]**")

# endregion


# region: === Main message handling flow. ===


# TODO: Clean this up.
def handle_stream(stream: StreamLike) -> Iterable[str]:
    # Any dynamic processing of the stream goes here.
    #
    # This currently contains:
    #     - the code block detection and hiding logic (if code hidden option is set).
    active_agent = engine().get_state().agent

    so_far = ""  # Accumulator for the stream.

    dummy_issued_code_gen = False  # Flag to issue a dummy message to indicate code generation.
    post_code_continue = False  # Flag to check for the first item in the post-code section.

    dummy_issued_coordinator_plan = False  # Flag to issue a dummy message to indicate coordinator planning.

    for item in stream:
        if item == LoadingIndicator:
            raise ValueError(LOADING_INDICATOR_EXC_MSG)

        so_far += item  # Accumulate the stream.

        to_yield = item

        if engine().session.session_settings.show_code is False:
            # ^ Only need to bother with code block detection if the code hidden option is enabled.

            # Use the helper function to detect and excise code blocks:
            _, code_block, post_code = id_and_excise_code(so_far)  # pylint: disable=redefined-outer-name

            if code_block is not None:
                # ^ Code block was detected, proceed to hiding it.

                if dummy_issued_code_gen is False:
                    # If we haven't issued the dummy message yet, do so now.
                    to_yield = f" ...\n\n {CODE_GEN_DUMMY_MESSAGE} \n\n"
                    dummy_issued_code_gen = True

                else:
                    # If dummy message has already been issued...
                    if post_code is not None:
                        # ... unless we have gone past the end of the code block...
                        # (In which case, restart yielding the stream as normal.)
                        if not post_code_continue:
                            # ^ If we are at the very first post-code item (stream chunk):
                            # Tidy this first item as it can be a bit messy.
                            to_yield = item.replace("`", "")
                        else:
                            # Yield the rest of the post-code section as normal.
                            post_code_continue = True
                            # ^ Indicate that we've already dealt with the first post-code item.
                            to_yield = item
                    else:
                        # ...hide the code block (keep yielding ""s).
                        to_yield = ""

            else:
                # No code block detected, pass the stream on as normal.
                to_yield = item

        if engine().session.session_settings.show_planning_details is False and active_agent == "coordinator":
            if dummy_issued_coordinator_plan is False:
                # If we haven't issued the dummy message yet, do so now.
                to_yield = f" ...\n\n {COORDINATOR_PLAN_DUMMY_MESSAGE} \n\n"
                dummy_issued_coordinator_plan = True
            else:
                to_yield = ""

        # Pass the stream on as normal.
        yield to_yield


def user_input_box(disabled: bool = False, **kwargs: Any) -> Optional[str]:
    with container_msg_input:
        c_msg_btn1, c_msg_btn2, c_msg_btn3, c_msg_input = st.columns([0.6, 0.6, 0.6, 8.2])
        with c_msg_btn1:
            if st.button("✅", disabled=disabled, help="A shortcut for proceeding to next step."):
                return "All correct. Proceed."
        with c_msg_btn2:
            if st.button("↻", disabled=False, help="Restart from the last reasoning step."):
                ui_log("Restart button pressed.")
                # CASE: Streaming, restart means to just restart from the last streaming message. Just restart the
                # UI in present interaction stage.
                if engine().get_state().streaming is True:
                    ui_log("Restart CASE: Streaming. Restarting from the last streaming message.")
                    rerun_with_state(state=None)
                # CASE: Not streaming, restart means to discard the last message and restart from the last
                # reasoning step.
                else:
                    ui_log(
                        "Restart CASE: Not streaming. Discarding last message and "
                        "restarting from the last reasoning step."
                    )
                    # TODO: Make robust - currently if discard_last fails, no fallback.
                    success = engine().discard_last()
                    if success:
                        rerun_with_state(state=None)
                    else:
                        ui_log("Discard last failed.")
        with c_msg_btn3:
            if st.button(
                "🔃",
                disabled=False,
                help="Go back to just after the last user message and create new branch in message tree.",
            ):
                ui_log("Go back and branch button pressed.")

                # TODO: Understand this better in the context of the paper method
                # CASE: Streaming, restart means to just restart from the last streaming message. Just restart the
                # UI in present interaction stage.
                if engine().get_state().streaming is True:
                    ui_log("Restart CASE: Streaming. Branching from last user message.")
                    rerun_with_state(state=engine().get_state().ui_controlled.interaction_stage)
                # CASE: Not streaming, restart means to discard the last message and restart from the last
                # reasoning step.
                else:
                    ui_log("Restart CASE: Not streaming. Branching from last user message.")
                    success = engine().create_new_message_branch()
                    if success:
                        rerun_with_state(state="reason")
                    else:
                        ui_log("create new branch failed.")
                        rerun_with_state(state="await_user_input")
        with c_msg_input:
            return st.chat_input("Type your message here...", disabled=disabled, **kwargs)


def execute_tool(tool_request) -> None:
    ui_log("In execute_tool")
    with st.chat_message("assistant"):
        with st.spinner("Executing tool..."):
            # Live stream the tool output.
            return_ = ToolOutput()
            # print(engine().get_state().ui_controlled.input_request)
            tool_out = engine().execute_tool_call(
                tool_request,
                user_input=engine().get_state().ui_controlled.input_request,  # type: ignore
            )
            # print("tool_out", tool_out)

            tool_out_stream = tool_out_wrapper(
                tool_out,
                return_,
                hook_pre_fns=[
                    update_debug_panel_ui_comm,
                    show_tab_warnings_if_tool_running,
                ],
            )
            # ^ Keep only strings in stream, exclude the final Return object (put value into `return_` instead).

            with st.expander(TOOL_LOGS_PREFIX, expanded=engine().session.session_settings.show_tool_call_logs):
                st.write_stream(tool_out_stream)

            list(tool_out_stream)  # Just in case.
            update_debug_panel_ui_comm()


def handle_user_input() -> None:
    ui_log("ui_state == 'await_user_input'")

    if engine().get_state().ui_controlled.input_request is None:
        ui_log("In main_flow > ui_state == 'await_user_input' > input_request is None [= Chat INPUT]")
        # CASE: Simple chat message input.
        user_input = user_input_box(disabled=False)

        # HACK --- --- ---
        js_input_placeholder = st_common.JSExecutor(
            js_code=dedent("""
                console.log("Executing js_input_placeholder...");

                // Find the textarea element with the specific aria-label attribute
                let textarea = parent.document.querySelector('textarea[aria-label="Type your message here..."]');

                // Check if the textarea element exists, and then insert the input placeholder into it
                if (textarea) {
                    // Focus on the textarea element
                    textarea.focus();
                    
                    // A timeout needed for the update to actually happen.
                    setTimeout(() => {
                        // Insert the text into the textarea
                        textarea.value = "<input_placeholder>";
                        textarea.focus();
                    }, 625);
                           
                    console.log("Inserted input placeholder value into the chat input box.");
                } else {
                    console.log("Could not find chat input box.");
                }
                """),
            container=container_tab_js,
            use_st_js=True,
            log_text="js_input_placeholder",
        )
        input_placeholder = engine().get_state().ui_controlled.input_placeholder
        if input_placeholder is not None:
            input_placeholder = input_placeholder.replace("\n", " ").replace("\r", " ")
            try:
                js_input_placeholder.execute_js(replacements={"<input_placeholder>": input_placeholder})
            except Exception as e:
                ui_log("Failed to insert input placeholder into the chat input box:", e)
        # HACK (end) --- --- ---

        print("user_input", user_input)
        if user_input:
            engine().ingest_user_input(user_input)
        else:
            # NOTE: Why st.stop()? It seems otherwise streamlit tends to take the input as None
            # and continue running, breaking the flow.
            st.stop()

        rerun_with_state(state="reason")

    else:
        print("[UI]>>> In main_flow > ui_state == 'await_user_input' > input_request is not None [= Tool call INPUT]")
        # CASE: Tool input request.
        input_request = engine().get_state().ui_controlled.input_request
        if input_request is None:
            raise ValueError("Input request was None")
        user_input_box(disabled=True)

        if input_request.kind == "file":
            file_types = input_request.extra["file_types"]
            with st.chat_message("assistant"):
                st_uploaded_file = st.file_uploader(
                    input_request.description or "",
                    accept_multiple_files=False,
                    type=file_types,
                )

            print("st_uploaded_file:", st_uploaded_file)
            if st_uploaded_file is not None:
                input_request.received_input = UploadedFileAbstraction(
                    name=st_uploaded_file.name,  # type: ignore
                    content=st_uploaded_file.getvalue(),  # type: ignore
                )
                engine().get_state().ui_controlled.input_request = input_request
                engine().update_state()
            else:
                # NOTE: Why st.stop()? It seems otherwise streamlit tends to take the input as None
                # and continue running, breaking the flow.
                st.stop()
        else:
            raise NotImplementedError("Only file uploads are supported at the moment")

        rerun_with_state(state="output")  # --> to tool call step (again).


def handle_reason_stream() -> None:
    # Send the conversation history etc. to LLM and get the response stream.
    ui_log("ui_state == 'reason'")
    user_input_box(disabled=True)

    ui_log("engine().reason() called")
    stream = engine().reason()

    # Handle the incoming stream.
    if stream is not None:
        # ui_log("In main_flow > stream is not None")
        with st.chat_message("assistant"):
            spinner_message = "Responding..."
            if engine().get_state().agent == "coordinator":
                spinner_message = "Planning..."
            elif engine().get_state().agent == "supervisor":
                spinner_message = "Checking work..."
            try:
                # Text stream, stream it as normal.
                with st.spinner(spinner_message):
                    st.write_stream(handle_stream(stream))
            except ValueError as e:
                if str(e) == LOADING_INDICATOR_EXC_MSG:
                    # If we happen to detect the LoadingIndicator during handling the stream, show the spinner
                    # with the appropriate message (it is going to be a tool request case) and list the stream,
                    # that is, process it under the hood without streaming anything.
                    with st.spinner("Choosing tools..."):
                        list(stream)
                else:
                    # If it's any other exception, raise it as normal.
                    raise e

    else:
        ui_log("In main_flow > stream is None")
        if engine().get_state().response_kind == ResponseKind.NOT_SET:
            raise ValueError("Engine response kind was not set after reasoning step.")

    rerun_with_state(state="output")


def handle_output_text_message() -> None:
    ui_log("In main_flow > ui_state == 'output' > In ResponseKind.TEXT_MESSAGE")

    if engine().get_state().user_message_requested:
        ui_log("In main_flow > In ResponseKind.TEXT_MESSAGE > user_message_requested")

        rerun_with_state(state="await_user_input")

    else:
        print("[UI]>>> In main_flow > ui_state == 'output' > In ResponseKind.TEXT_MESSAGE > No user_message_requested")

        rerun_with_state(state="reason")


def handle_output_code_generation() -> None:
    ui_log("In main_flow > ui_state == 'output' > In ResponseKind.CODE_GENERATION")

    finished_sentinel = CodeExecFinishedSentinel(status="success")
    with st.chat_message("code_execution", avatar=AVATAR_MAP["code_execution"]):
        with st.spinner("Executing code..."):
            output = engine().execute_generated_code()
            with st.expander(CODE_EXECUTION_OUT_PREFIX, expanded=engine().session.session_settings.show_code_out):
                st.write_stream(code_out_wrapper(output, finished_stl=finished_sentinel))
            list(output)  # Just in case.

    rerun_with_state(state="reason")


def handle_output_tool_request() -> None:
    ui_log("In main_flow > ui_state == 'output' > In ResponseKind.TOOL_REQUEST")

    tool_request = engine().get_state().tool_request
    if tool_request is None:
        raise ValueError("Tool request was None")
    tool = get_tool(tool_request.name)

    # CASE [TOOL_REQUEST > INPUT]: Handle user input requests if any.
    if (tool.user_input_requested) and (  # Tool has requested user input and...
        engine().get_state().ui_controlled.input_request is None  # ... No input request has been set, or...
        or (
            # ... An input request has been set, but no input has been received yet.
            (engine().get_state().ui_controlled.input_request)
            and (engine().get_state().ui_controlled.input_request.received_input is None)  # type: ignore
        )
    ):
        ui_log("In main_flow > ui_state == 'output' > In ResponseKind.TOOL_REQUEST > user_input_requested")

        if len(tool.user_input_requested) > 1:
            raise NotImplementedError("Multiple user inputs are not supported at the moment")
        engine().get_state().ui_controlled.input_request = tool.user_input_requested[0]
        engine().update_state()
        print(
            "engine().get_state().ui_controlled.input_request:\n",
            engine().get_state().ui_controlled.input_request,
        )

        rerun_with_state(state="await_user_input")

    # CASE [TOOL_REQUEST > EXECUTION]: No user input requests, proceed to tool execution.
    else:
        ui_log("In main_flow > ui_state == 'output' > In ResponseKind.TOOL_REQUEST > No user_input_requested")

        # Handle tool execution.
        execute_tool(tool_request)

        # Reset UI State tool input tracking.
        engine().get_state().ui_controlled.input_request = None
        engine().update_state()

        rerun_with_state(state="reason")


def handle_output() -> None:
    ui_log("ui_state == 'output'")
    user_input_box(disabled=True)
    print("engine().get_state().response_kind:", engine().get_state().response_kind)

    # CASE: The incoming stream is a TEXT message from Engine:
    if engine().get_state().response_kind == ResponseKind.TEXT_MESSAGE:
        handle_output_text_message()

    # CASE [CODE_GENERATION]: The incoming stream is a code generation request from Engine:
    elif engine().get_state().response_kind == ResponseKind.CODE_GENERATION:
        handle_output_code_generation()

    # CASE [TOOL_REQUEST]: The incoming stream is a list of tool calls from Engine:
    elif engine().get_state().response_kind == ResponseKind.TOOL_REQUEST:
        handle_output_tool_request()

    else:
        raise ValueError(f"Unexpected engine response kind: {engine().get_state().response_kind}")


def main_flow() -> None:
    ui_log("In main_flow")

    scroller.execute_js(replacements={"<selector>": SELECTOR_CHAT_HISTORY_CONTAINER})

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # "Preview only" mode
    # user_input_box(disabled=True)
    # st.stop()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO: If there is an active tool on reload - kill it!

    with container_chat:
        # 0. Project end:
        if engine().project_completed():
            ui_log("Project end detected.")
            with st.chat_message("assistant"):
                st.markdown("🏁 **Project finished.** 🏁")
                if st.button("🖨️ Generate session report"):
                    if WEASYPRINT_WORKING:
                        with st.spinner("Generating report..."):
                            report_path = prepare_report(working_dir=engine().working_directory_abs)
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download report",
                                    data=file,
                                    file_name=os.path.basename(report_path),
                                    mime="text/pdf",
                                    type="primary",
                                )
                    else:
                        st.warning(
                            "WeasyPrint import failed. Please check the troubleshooting section in the documentation."
                        )
            user_input_box(disabled=True)

        # I. Handle user input.
        elif engine().get_state().ui_controlled.interaction_stage == "await_user_input":
            handle_user_input()

        # II. Handle the reason & stream flow.
        elif engine().get_state().ui_controlled.interaction_stage == "reason":
            handle_reason_stream()

        # III. Handle the different final responses.
        elif engine().get_state().ui_controlled.interaction_stage == "output":
            handle_output()


main_flow()

# endregion
