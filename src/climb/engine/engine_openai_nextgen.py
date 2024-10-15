import ast
import copy
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import pydantic
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
from climb.common.utils import d2m, engine_log, m2d, update_templates
from climb.db import DB
from climb.tool import list_all_tool_names, list_all_tool_specs

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

# DEBUG__PRINT_GET_MESSAGE_HISTORY: In _engine_openai.py
# DEBUG__PRINT_APPEND_MESSAGE: In _engine_openai.py
DEBUG__PRINT_MESSAGES_PER_AGENT = False
# ---
DEBUG__PRINT_MESSAGES_TO_LLM = False
DEBUG__PRINT_TOOLS = False
# ---
DEBUG__PRINT_DELTA = False
# ---
DEBUG__USE_FILTER_TOOLS = True  # If False, the worker will always be given all tools.

# region: === Prompt templates ===

LOG_PREAMBLE = "HERE IS THE LOG OF ALL THE AGENT-USER CONVERSATIONS SO FAR"
LAST_WORKER_LOG_PREAMBLE = "BELOW IS THE LOG OF THE WORKER AGENT AND USER CONVERSATION FOR THE *LAST TASK*"
# ---
CONTINUE_INDICATOR = "YOUR WORK FROM HERE"
TASK_COMPLETED_INDICATOR = "TASK COMPLETED"
LONG_MESSAGE_SPLIT_INDICATOR = "= CONTINUE ="
# ---
WORKER_ACTUAL_TASK_REPLACE_MARKER = "{WORKER_ACTUAL_TASK}"
WD_CONTENTS_REPLACE_MARKER = "{WD_CONTENTS}"
STRUCTURED_PLAN_REPLACE_MARKER = "{STRUCTURED_PLAN}"
MAX_CHARS_REPLACE_MARKER = "{MAX_CHARS}"

PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES = "{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_1 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_2 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_2}"

GENERATED_CODE_FORMAT_ERROR_MSG = "**IMPORTANT** Code execution failed due to wrong format of generated code"
PROBLEM_WITH_SUBTASK_STATUS_UPDATES_MARKER = "PROBLEM WITH SUBTASK STATUS UPDATES"
PROBLEM_WITH_SUBTASK_SELECTION_MARKER = "PROBLEM WITH SUBTASK SELECTION"
PROJECT_END_MARKER = "PROJECT END"

TASKS_PLANNED_AFTER = "Tasks planned after this"

# TODO:
# Consider:
# - Limiting subtask tools.
# - Specifying subtask dependencies.

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
You are a powerful AI assistant. You help your users, who are usually medical researchers, \
clinicians, or pharmacology experts to perform machine learning studies on their data.

Your capabilities:
- You are able to use OpenAI tools (functions) that have been described to you.
- You can generate code that gets automatically run on the USER'S computer, and see its output.
- You DO NOT have access to the internet.
{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}
- You have a great understanding of data analysis, machine learning, and medical research.
- You are able to show images from the working directory to the user (see Rules).
- You are able to return long messages to the user using the special indicator text (see Rules).
"""

# Extract everything from "Your capabilities:" to the end of the string (inclusive):
WORKER_CAPABILITIES_FOR_COORDINATOR = (
    WORKER_CAPABILITIES[WORKER_CAPABILITIES.index("Your capabilities:") :]
    .replace("Your capabilities:", "Worker agents' capabilities:")
    .replace("Your", "Their")
    .replace("You", "They")
)

WORKER_RULES_LONG_MESSAGES = f"""\
#### LONG MESSAGES:
- Your design limits you to {MAX_CHARS_REPLACE_MARKER} characters per message. 
- In some cases, you may need to generate a longer message than this.
- We will work around this like so:
    1. You will generate the long message in chunks that are shorter than {MAX_CHARS_REPLACE_MARKER} characters.
    2. You will add "{LONG_MESSAGE_SPLIT_INDICATOR}" at the end of each chunk.
    3. The chunk generated so far will be added to the message history, and you will be able to continue from where you \
left off.
    4. To finish, simply finish the message as normal.
- Use this capability sparingly, only when really necessary, or when the plan explicitly suggests you to do so.
"""

# NOTE: Use for testing:
# STRUCTURED_PLAN = [
#     {
#         "task_id": "TEST_TASK",
#         "selection": "mandatory",
#         "selection_condition": None,
#         "task_name": "Test task",
#         "project_stage": "Alignment check",
#         "coordinator_guidance": None,
#         "worker_guidance": None,
#         "task_status": "not_started",
#         "subtasks": [
#             {
#                 "subtask_id": "TEST_TASK_1",
#                 "selection": "mandatory",
#                 "selection_condition": None,
#                 "subtask_status": "not_started",
#                 "status_reason": None,
#                 "subtask_name": "Generate a long story",
#                 "subtask_details": """
# This is a test task not related to medical machine learning.

# 1. Use the tool for uploading a data file to get the data file from user.
# 2. Use the hyperimpute tool to impute the data.
# """,
#                 "coordinator_guidance": None,
#                 "worker_guidance": None,
#                 "tools": None,
#             },
#         ],
#     },
# ]

STRUCTURED_PLAN = [
    {
        "task_id": "ENV",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Getting the environment ready",
        "project_stage": "Alignment check",
        "coordinator_guidance": """
- Issue all subtasks in this task together to the worker.
""",
        "worker_guidance": """
- User interaction **IMPORTANT!**: this should be minimal in this section, proceed with the steps as instructed quickly.
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "ENV_1",
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_status": "not_started",
                "status_reason": None,
                "subtask_name": "Upload data file",
                "subtask_details": """
- Introduce yourself as an AI assistant that will help the user with their clinical machine learning study.
- Ask the user if they have their data file ready as a CSV file.
- Then summon the `upload_data_file` tool so that the user can upload their data file.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": ["upload_data_file"],
            },
            {
                "subtask_id": "ENV_2",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Check hardware",
                "subtask_details": """
Use the `hardware_info` tool to get information about the user's hardware. Using the report, determine whether the \
user's hardware is suitable for the task. As a rough guide, we want a machine with a CPU with at least 4 cores, 16GB \
of RAM, and a GPU with at least 4GB of memory. If the user's hardware is not suitable, suggest they find a machine \
that meets these requirements or use a cloud service, but allow the option to proceed anyway.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": ["hardware_info"],
            },
            {
                "subtask_id": "ENV_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Check data file can be loaded",
                "subtask_details": """
Generate code to check whether the data file can be loaded with `pd.read_csv(<file_path>)`, as that is how the tools \
expect it. CHECK that the loaded dataframe has MORE THAN ONE column and more than one row - otherwise it usually means \
the separator or delimiter is wrong. Try to find a way to load the file (e.g. try different delimiters), and then save \
the modified file in way that can be loaded with `pd.read_csv(<file_path>)`. If not possible, suggest to the user that \
they fix the data and upload it again.""",
                "coordinator_guidance": None,
                "worker_guidance": """
- You MUST NOT use any tool here. DO NOT SUMMON ANY TOOLS.
- You MUST generate code in this step!
- So, your response MUST have:
DEPENDENCIES:
```
pandas
```
CODE:
```
... your code to complete the subtask ...
```
""",
                "tools": [],
            },
        ],
    },
    {
        "task_id": "INFO",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Getting information from the user",
        "project_stage": "Alignment check",
        "coordinator_guidance": """
- Issue all subtasks in this task together to the worker.
""",
        "worker_guidance": """
- Do not ask the user about the specific details of their data, as the data will be investigated in detail later. \
Focus on the high-level information here.
- DO NOT PROCEED to the EDA, this will be done later.
- DO NOT PROCEED to the ML study yet, this will be done later.
- DO NOT suggest any specific preprocessing steps to take, that will be done later.
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "INFO_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "High-level information from the user",
                "subtask_details": """
Ask the user whether they would like to provide high-level information about the dataset, especially:
    - How is it structured (what does a row or a column represent)?
    - Any background information about the data.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": [],
            },
            {
                "subtask_id": "INFO_2",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Experiment setup and research question from the user",
                "subtask_details": """
1. Ask the user to describe the experiment setup and the research question they wish to investigate.
2. Confirm with the user what the name of the target column in their dataset is.
3. (Conditional step) Only IF you suspect that the user wants to do SURVIVAL ANALYSIS, confirm with the user that (A) \
the target column represents the event (usually 1 = event, and 0 = censoring) and (B) that they have a column that \
represents the time to the event. Make a mental note of this. If the user has columns that can be transformed into \
these, that is also not a problem, and another agent will handle this later.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": [],
            },
            {
                "subtask_id": "INFO_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Assess data suitability and tool support",
                "subtask_details": """
Given what the user has told you, ask yourself two things:
    Q1: Is the data suitable for the task?
        > Example problem: the data is in a format that is not supported by the tools (e.g. time series data).
        > Example problem: more than one row per patient, but the tools expect one row per patient.
        > Think of any such problems...
    Q2: Does the AutoPrognosis set of tools that you have access to support the task?
- If the answer to Q1 is NO, think whether the data can be somehow transformed to fit the task. If you think this \
is possible, suggest this to the user. If not, suggest how the user can get the right data.
- If the answer to Q2 is NO, apologize to the user, mention that your capabilities are still being enhanced, but \
for now this task cannot be performed.
- If on the basis of the above you think the task CAN be performed, proceed to the next step. Otherwise, ask the \
user if you can help them with anything else.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
DO NOT actually execute the AutoPrognosis tools here! Use their specifications for your information, but DO NOT \
invoke them!
""",
                "tools": [
                    "autoprognosis_classification",
                    "autoprognosis_regression",
                    "autoprognosis_survival",
                ],
            },
        ],
    },
    {
        "task_id": "EDA",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Exploratory Data Analysis",
        "project_stage": "Data exploration",
        "coordinator_guidance": """
- Issue subtasks in this task together to the worker, apart from EDA_4, which is conditional.
""",
        "worker_guidance": """
- User interaction: avoid asking user for confirmation too much, only ask important questions.
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "EDA_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Exclude/keep columns",
                "subtask_details": """
Generate code to list the names of all the columns in the dataset and print this clearly to the user. Ask the user \
if they would like to exclude certain columns from the analysis, or conversely, only keep certain columns. If so, find \
out which columns these are. Then generate code to drop the columns that the user wants to exclude or to only keep the \
columns that the user wants to keep. Save this modified dataset with a suffix `_user_cols` in the filename.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
When generating this code, print the columns line by line (not as one list) so that the user can easily see them.
""",
                "tools": [],
            },
            {
                "subtask_id": "EDA_2",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Perform EDA",
                "subtask_details": """
Perform exploratory data analysis on the data using the `EDA` tool.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
**IMPORTANT**: This step needs executing a TOOL called `EDA`. **DO NOT** write your own code for this step!

- The `target` (name of the target column) argument for the EDA tool should be clear from previous steps, in most cases. \
PROVIDE it to the tool unless definitely not possible.

- After executing the tool, provide the user with a summary of what you see in the EDA. Use your best understanding of \
data science and machine learning. **DO NOT** make suggestions of what needs to be done next! That will be handled \
later in the process. **Just summarize your learnings.**
""",
                "tools": ["EDA"],
            },
            {
                "subtask_id": "EDA_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Generate descriptive statistics",
                "subtask_details": """
- Ask the user if they would like to generate descriptive statistics. If yes:
- Generate descriptive statistics using the `descriptive_statistics` tool.
- Suggest that the user reviews the EDA and descriptive statistics at this stage. Ask them if they have reviewed \
these and if they have any questions (answer as needed). Only then proceed to the next step.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
(1) After executing the descriptive statistics tool, provide the user with a summary of what you found out. Use your best \
understanding of medical research and data science.

(2) Check the tool logs for the names of the figures generated by the tool. Think about which ones are most important \
(let's say five most important ones). Then use your rules for showing images to the user to show these images for \
them to review.

**IMPORTANT**: to show an image simply include `<WD>/image_name.extension` in your message. Always use this EXACT \
format when showing an image!

**DO NOT** make suggestions of what needs to be done next! That will be handled later in the process.
**Just summarize your learnings here.**
""",
                "tools": ["descriptive_statistics"],
            },
            {
                "subtask_id": "EDA_4",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if data analysis reveals fewer than 50 samples",
                "subtask_name": "Warn about small sample size if necessary",
                "subtask_details": """
ONLY IF there are fewer than about 50 samples, warn the user that the results may not be reliable as there is not \
enough data. Allow to continue if the user is happy with that. Skip this step completely if there are more than 50 \
samples.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": [],
            },
            {
                "subtask_id": "EDA_5",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "If the user's problem is SURVIVAL ANALYSIS",
                "subtask_name": "Show Kaplan-Meier plot",
                "subtask_details": """
Reminder: this task is only relevant if the user's problem is SURVIVAL ANALYSIS.

- Ask the user if they would like to see a Kaplan-Meier plot for the survival analysis. If yes:
- Make sure that you know exactly which column represents the time to the event and which column represents the event.
- Generate code to show the Kaplan-Meier plot. Hint: use the `lifelines` library.
- Show the plot to the user.
""",
                "coordinator_guidance": """
Review the conversation history to see if the user's task is likely to be survival analysis. If it seems like it is, \
you MUST issue this subtask, do NOT skip it in that case!
""",
                "worker_guidance": """
Important:
* To show the plot you must save it as an image file in the working directory.
* After the code has been run, and if the plot was successfully saved (check CURRENT WORKING DIRECTORY contents \
to confirm this), show it to the user using the `<WD>/image_name.extension` format in your next message.
""",
                "tools": [],
            },
        ],
    },
    {
        "task_id": "DP-BM",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Data processing - before missing data handling",
        "project_stage": "Data engineering",
        "coordinator_guidance": None,
        "worker_guidance": """
- User interaction: this section requires especially heavy back-and-forth with the user. Let the user guide you.
- DO NOT PROCEED to the ML study yet, this will be done later.
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "DP-BM_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Column background information",
                "subtask_details": """
Go through EACH columns with the and gather background information.
    - IF you have some idea what the column represents, provide the user with a short summary of this. Ask the user if \
this is correct, and if not, ask them to provide the correct information.
    - IF you are not sure what the column represents, ask the user to provide this information about the column \
straight away.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
- Do *NOT* ask about one column at a time, but rather go through several columns at once, so that the process is \
more efficient.
- If there is a reasonable number of columns, roughly < 30, ensure that you have gone through all of them.
- If there are many columns, focus on the most important ones, or the ones that are most likely to be relevant to the \
task. Once you have done this, ask the user if they would like to continue with the remaining columns.
- *ALWAYS* go over the columns whose meaning is not clear to you - and ask the user for clarification.
""",
                "tools": [],
            },
            # NOTE: This step doesn't work very well in the current form.
            #             {
            #                 "subtask_id": "DP-BM_2",
            #                 "subtask_status": "not_started",
            #                 "status_reason": None,
            #                 "selection": "mandatory",
            #                 "selection_condition": None,
            #                 "subtask_name": "Ensure sensible column data types",
            #                 "subtask_details": """
            # Are any columns encoded incorrectly? Consider the data types: numeric, categorical, date-time-like. Remember that \
            # `object` in pandas often means string. Convert the obviously incorrect columns by generating code. \
            # If it's date-time-like, be especially careful about the format. Use all data available to you to infer the format.
            # """,
            #                 "coordinator_guidance": """
            # - If this step was unsuccessful due to missing values, you should reissue this step AFTER the missing data handling \
            # steps have been completed.
            # """,
            #                 "worker_guidance": """
            # - ***IMPORTANT**: Refer to the output of the EDA step from the conversation history to help you with this task.
            # - If there are missing values, you may experience some issues here. If so, skip this step. The missing values will be \
            # imputed later, and the data types will be corrected then.
            # - NEVER proceed to the MISSING VALUE HANDLING here. Remember, this is planned later, and will be handled by a separate \
            # agent who has the expertise to do this.
            # """,
            #                 "tools": [],
            #             },
        ],
    },
    {
        "task_id": "DP-M",
        "selection": "conditional",
        "selection_condition": "Only if missing data is present",
        "task_name": "Data processing - missing data handling",
        "project_stage": "Data engineering",
        "coordinator_guidance": """
- Use the output of the EDA step in message history to decide if the dataset needs missing data handling.
- If there is no missing data, skip this task.
- **IMPORTANT**: Remember, some missing values may be encoded as non-standard NaNs. If you suspect this may be the case, \
this task MUST still be carried out!
- **IMPORTANT** Issue the steps BEFORE HyperImpute step separately first.
- **IMPORTANT** Then issue the HyperImpute step ON ITS OWN.
""",
        "worker_guidance": None,
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "DP-M_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Represent missing data as NaN",
                "subtask_details": """
Any missing data must be first represented as `numpy.nan` values.

1. Check the results from the EDA step to figure out if there are non-standard NaNs. If you have reason to suspect \
this is the case, list for the user, what specific values you suspect are used to represent missing data. Ask the user \
to confirm if these are indeed used to represent missing data.
**Note:** if you suspect there are no non-standard NaNs, tell this to the user and confirm. If the user confirms, you can \
move on with the plan.

2. Generate code to replace the non-standard NaNs with `numpy.nan` values. Save this modified dataset with the suffix \
`_nan` in the filename.
""",
                "coordinator_guidance": "It is recommended to issue this task together with DP-M_2 and DP-M_3.",
                "worker_guidance": """
**IMPORTANT**
- Do NOT assume some "typical" non-standard NaN placeholders.
- Always check based on the data analysis results, and CONFIRM wit the user what the placeholders are.
Otherwise you could end up replacing values that are not actually missing data!
""",
                "tools": [],
            },
            {
                "subtask_id": "DP-M_2",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if missing data is present",
                "subtask_name": "Consider dropping columns with high missing values",
                "subtask_details": """
If there is any missing data in the dataset:
    - Generate code to show per-column % missing values. Show these in descending order of % missing values.
    - Suggest that as a rule of thumb any columns with 80%+ missing values should be removed. Ask the user what their \
acceptable threshold is for including a column in the analysis.
    - Ask the user if they are happy to drop the columns that exceed this threshold? Do they want to keep some columns \
that exceed this threshold? If so, which ones?
    - Given the user responses, generate code to drop the appropriate columns.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
**IMPORTANT**: When generating code to show per-column % missing values:
Remember that pandas shortens the printed output of the dataframe.
Use the statements:
pd.options.display.max_rows = None
pd.options.display.max_columns = None
in your code, to ensure that all the results are shown!
""",
                "tools": [],
            },
            {
                "subtask_id": "DP-M_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if missing data is present",
                "subtask_name": "Consider dropping rows with missing values",
                "subtask_details": """
Generate code to show:
    - per-column % missing values,
    - % of total rows that have missing values.
If there are still missing values in the dataset:
    - If the user's intended TARGET VARIABLE has missing values, we must make sure to handle this. Ask them if they are \
happy to drop rows with missing values in the target variable. Point out to the user that imputing the target variable \
is not recommended.
    - Ask the user if they are happy to drop rows with missing data (in any column), or if they are happy to use an \
imputation tool (HyperImpute). Suggest that imputation is usually the better option, especially if most rows have \
missing values. If the percentage of rows with missing values is high, strongly suggest using imputation rather than \
dropping rows, due to the risk of losing valuable data.
    - If they want to drop the rows, generate code to do so.
    - If they want to use HyperImpute, complete your task, the next agent will handle this.
""",
                "coordinator_guidance": None,
                "worker_guidance": "You MUST NOT use HyperImpute tool! The next agent will handle this.",
                "tools": [],
            },
            {
                "subtask_id": "DP-M_4",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if missing data is present",
                "subtask_name": "Impute missing values",
                "subtask_details": """
Step 1. Generate code to show:
    - per-column % missing values,
    - % of total rows that have missing values.
Step 2. Explain to the user what the hyperimpute tool will do and check they are happy to use it.
Step 3. If there are still missing values in the dataset:
    - Use the `hyperimpute_imputation` tool (do not write own code for imputation).
After imputation, generate code to show the per-column % missing values again, and confirm with the user that there are \
no more missing values.
""",
                "coordinator_guidance": """
**IMPORTANT**:
- This subtask MUST be issued on its own AFTER the other subtasks in this task!
- If the dataset has ANY missing values at all, we MUST IMPUTE THEM before we can proceed to the predictive \
modelling stage. Hence, you MUST issue this subtask IF there are missing values.

The ONLY situation where you may skip this subtask is when it is EXPLICITLY CLEAR from the conversation history \
that there are NONE - that is ZERO - missing values in the dataset. Carefully review the per column % missing values \
to confirm this. If UNSURE, issue this subtask JUST IN CASE.
""",
                "worker_guidance": """
* Use the LATEST version of the dataset, after all the previous steps have been completed. Check the conversation \
history and the modification date-time of the files in the working directory to ensure this.
* Save the imputed dataset with a suffix `_hyperimputed` in the filename.
""",
                "tools": ["hyperimpute_imputation"],
            },
        ],
    },
    {
        "task_id": "DP-AM",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Data processing - after missing data handling",
        "project_stage": "Data engineering",
        "coordinator_guidance": """
- Proceed to this step only after the missing data handling step has been completed, OR if there was no missing data.
""",
        "worker_guidance": None,
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "DP-AM_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Discuss data preprocessing with the user",
                "subtask_details": """
Discuss with the user whether they have any particular data preprocessing steps in mind. work with them to \
generate code to perform these steps. **IMPORTANT** Since your tools down the line support standard preprocessing \
like normalization, feature selection, etc., DO NOT suggest these here (unless the user explicitly asks to do these). \
This step is about whether the user has any specific data transformations in mind, that have medical meaning.
""",
                "coordinator_guidance": """
This is a LONG subtask and it involves a lot of back-and-forth with the user. Issue this subtask ON ITS OWN.
The worker may have many things to do here, to satisfy the user needs.
""",
                "worker_guidance": """
- **IMPORTANT:** You MUST keep asking the user if they have any MORE preprocessing steps in mind. The user may want to \
do multiple things here! You must keep asking until the user says they are finished and do not have any more \
preprocessing steps in mind. Only then can you proceed to the next subtask (if any).
- **IMPORTANT:** You must work from the latest version of the dataset, after missing data handling!
""",
                "tools": ["EDA", "descriptive_statistics"],
            },
            {
                "subtask_id": "DP-AM_2",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Feature selection",
                "subtask_details": """
- Describe to the user what the `feature_selection` tool does and what it's useful for.
- Ask the user if they want to use this tool to select the most important features in the dataset.
- Only if the user says yes, summon the `feature_selection` tool, otherwise skip this step.
- Use the `feature_selection` tool to find the most important features in the dataset.
- Suggest to the user that they may want to further drop some or all features that are not in the list of important \
features,as it can help simplify the task and improve the performance of machine learning models.
- List the features that are selected as important and the features that are not important. Ask the user if they \
would like to drop any of the unimportant features.
- If so, generate code to do this.
- Save this modified dataset with a suffix `_selected_features` in the filename.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
**IMPORTANT**
If the task is survival analysis, the you must do the following BEFORE running the feature selection tool:
1. Confirm what is the TIME variable (the variable that has the time index representing event time)
2. Confirm what is the EVENT variable (the variable that represents the event itself, usually binary)
""",
                "tools": ["feature_selection"],
            },
        ],
    },
    {
        "task_id": "MLC",
        "task_name": "Machine learning study pre-checks",
        "selection": "mandatory",
        "selection_condition": None,
        "project_stage": "Model building",
        "coordinator_guidance": """
This task is critical for catching potential problems with the setup before we proceed to the actual machine learning.

Task MLC_1 is mandatory and must be completed first - it confirms which problem type the user has. Issue this task \
ON ITS OWN first.
""",
        "worker_guidance": """
**IMPORTANT**
Do NOT discuss the details of the ML study itself in this step. A subsequent agent will handle the ML study using \
the appropriate AutoPrognosis tool. Your job here is to do important checks only!
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "MLC_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Confirm ML problem type",
                "subtask_details": """
Confirm with the user to what the target variable is and whether it is a classification, regression, or survival \
analysis task - provide your best suggestion based on the message history and check with the user if this is correct. \
**IMPORTANT**: You must **explicitly** ask the user for confirmation on this, as this is a critical decision.
""",
                "coordinator_guidance": "Issue this task on its own first",
                "worker_guidance": """
Once you have the information, mark this task as completed! Do not proceed to running the study, this will be done \
later.
""",
                "tools": [],
            },
            {
                "subtask_id": "MLC_2-SURVIVAL",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if the ML problem is SURVIVAL ANALYSIS",
                "subtask_name": "Check time and event columns",
                "subtask_details": """
ONLY IF the task is SURVIVAL ANALYSIS, you need to make sure both the *time* and *event* (target) columns are present \
in the data. Check what format the time is in. The tool expects the time to be a number (integer or float) representing a duration \
until the event. If the time is in a date-like format, you need to convert it to a number representing the duration. \
Generate code to do this.
""",
                "coordinator_guidance": """
ONLY issue this task if the user has confirmed the problem setting is SURVIVAL ANALYSIS.
IF the problem is CLASSIFICATION or REGRESSION, you MUST SKIP this task. Do NOT issue it to the worker agent in that case.
""",
                "worker_guidance": """
Use your best understanding of the data from EDA to determine what should be considered the starting point for the \
time and what units to use. This is a tricky step, think carefully and step by step. \
The user will appreciate your help here!
""",
                "tools": [],
            },
            {
                "subtask_id": "MLC_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Check for data leakage",
                "subtask_details": """
We need to check for data leakage at this point. This is a critical step to ensure the integrity of the machine learning \
study. Data leakage is when information that would not be available at the time of prediction is used in the model. This \
can lead to overly optimistic results and a model that cannot be used in practice.

Follow the below substeps EXACTLY.
- (1) Explicitly list ALL THE COLUMNS (the current columns, after all the preprocessing steps we have done) except the \
target (and also except the event in case of SURVIVAL ANALYSIS). Do this by generating code. REMEMBER to use the \
LATEST version of the dataset!
- (2) Consult the message history, to check the meaning and details of each of these columns. Write a bullet point list \
of columns that you suspect are likely to represent a data leak, with a reason for each. Be careful with columns that \
represent information from different time points, especially in the survival analysis setting, as they may reveal the \
target. Remember, information not available at the time of prediction is likely data leakage.
Explain to the user why data leakage is a problem and suggest removing them. Example:
    - `cause_of_death`: This column is likely to reveal the target variable `death`.
    - `column_name`: Reason for suspecting a data leak.
    ...
NOTE: If there are no such columns, explicitly state that you do not suspect any data leakage, but ask the user to \
double check this.
NOTE: Since you might not have picked up all the data leakage columns, always ask the user to double-check whether \
they think there are any others.
- (3) If the user wants to exclude any of the columns discussed, GENERATE THE CODE to do so STRAIGHT AWAY. This cannot \
be done later!
""",
                "coordinator_guidance": None,
                "worker_guidance": """
**IMPORTANT**
- This is an important step that prevents data leakage. If not performed correctly, the user's ML study could be meaningless!
- Perform all substeps STEP BY STEP - DO NOT DEVIATE FROM THESE STEPS!
- Pause and discuss with the user at each step, don't do it all together - that's confusing. DO NOT RUSH!
- **IMPORTANT** If writing code to drop any columns, make sure to start from the LATEST version of the dataset \
(after missing data handling and any other preprocessing steps).

DO NOT do this subtask together with the "Check for irrelevant columns" task. Complete this subtask first! It will be \
very confusing for the user otherwise.
""",
                "tools": [],
            },
            {
                "subtask_id": "MLC_4",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Check for irrelevant columns",
                "subtask_details": """
We need to check for irrelevant columns at this point. Inclusion of irrelevant columns can lead to overfitting and \
misleading feature importance. Hence it is important to remove them before continuing with the machine learning study.

Follow the below substeps EXACTLY.
- (1) Now you need to check if there are any meaningless/irrelevant columns still left in the dataset. This usually \
means various ID columns or similar. Consult the message history to check the meaning and details of each of the \
columns to help you. Write a bullet point list of columns that you suspect are irrelevant, with a reason for each. \
Explain to the user why irrelevant columns are a problem and suggest removing them. Example:
    - `patient_id`: This column is an identifier and does not contain any useful information for the analysis.
    - `column_name`: Reason for suspecting an irrelevant column.
    ...
NOTE: If there are no such columns, explicitly state that you do not suspect any irrelevant columns, but ask the user \
to double check this.
NOTE: Since you might not have picked up all the irrelevant columns, always ask the user to double-check whether they \
think there are any others.
- (2) If the user wants to exclude any of the columns discussed, GENERATE THE CODE to do so STRAIGHT AWAY. This \
cannot be done later!

(3) Finally, list the feature columns that are left, and check that the user is happy to use all of these features in \
the machine learning study. This is to make it clear to the user what features we are going to use and serves as a \
final check.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
**IMPORTANT**
- This is an important step that prevents irrelevant columns from being used in the ML study.
- Perform all substeps STEP BY STEP - DO NOT DEVIATE FROM THESE STEPS!
- Pause and discuss with the user at each step, don't do it all together - that's confusing. DO NOT RUSH!
- **IMPORTANT** If writing code to drop any columns, make sure to start from the LATEST version of the dataset \
(after missing data handling and any other preprocessing steps, and after the data leakage check column removals).
""",
                "tools": [],
            },
        ],
    },
    {
        "task_id": "ML",
        "task_name": "Machine learning study",
        "selection": "mandatory",
        "selection_condition": None,
        "project_stage": "Model building",
        "coordinator_guidance": """
**VERY IMPORTANT**
This task has subtasks that are conditional on the user's problem setup.
Pay attention to the "selection" and "selection_condition"!
In MLC_1 task, the user would have confirmed the problem type: classification, regression, or survival analysis.
Only issue the subtask from this task that corresponds to the user's problem type.
""",
        "worker_guidance": """
If any of the tools you are executing throw errors, attempt to fix the issue - it is often a problem with the data, \
you are able to generate code to modify the data appropriately. Explain to the user what you are doing, if so
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "ML_1-CLASSIFICATION",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if the ML problem is CLASSIFICATION",
                "subtask_name": "Machine learning study - classification",
                "subtask_details": """
Perform a machine learning study using the `autoprognosis_classification` tool. Set the `mode` parameter to "all".
After the study is done, ask the user if they want to also try using linear models only. If so, then set the `mode` \
parameter to "linear" and run the tool again.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
NOTE: You must invoke the tool rather than writing your own code for this step.

If you receive an error that a minimum performance threshold was not met, suggest to the user the reasons as to \
why this may be the case, and what needs to be done to improve model performance. Provide advice SPECIFIC to the \
user's case.
""",
                "tools": ["autoprognosis_classification"],
            },
            {
                "subtask_id": "ML_1-REGRESSION",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if the ML problem is REGRESSION",
                "subtask_name": "Machine learning study - regression",
                "subtask_details": """
Perform a machine learning study using the `autoprognosis_regression` tool. Set the `mode` parameter to "all".
After the study is done, ask the user if they want to also try using linear models only. If so, then set the `mode` \
parameter to "linear" and run the tool again.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
NOTE: You must invoke the tool rather than writing your own code for this step.

If you receive an error that a minimum performance threshold was not met, suggest to the user the reasons as to \
why this may be the case, and what needs to be done to improve model performance. Provide advice SPECIFIC to the \
user's case.
""",
                "tools": ["autoprognosis_regression"],
            },
            {
                "subtask_id": "ML_1-SURVIVAL",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "Only if the ML problem is SURVIVAL ANALYSIS",
                "subtask_name": "Machine learning study - survival analysis",
                "subtask_details": """
NOTE: You must invoke the tool rather than writing your own code for this step.

Perform a machine learning study using the `autoprognosis_survival` tool. Set the `mode` parameter to "all".
After the study is done, ask the user if they want to also try using linear models only. If so, then set the `mode` \
parameter to "linear" and run the tool again.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
If you receive an error that a minimum performance threshold was not met, suggest to the user the reasons as to \
why this may be the case, and what needs to be done to improve model performance. Provide advice SPECIFIC to the \
user's case.
""",
                "tools": ["autoprognosis_survival"],
            },
        ],
    },
    {
        "task_id": "MLE",
        "task_name": "Machine learning study exploitation",
        "selection": "mandatory",
        "selection_condition": None,
        "project_stage": "Model exploitation",
        "coordinator_guidance": """
**VERY IMPORTANT**
This task has certain subtasks that are conditional on the user's problem setup.
Pay attention to the "selection" and "selection_condition"!
From conversation history, check what the user's problem type is: classification, regression, or survival analysis.
Do not issue a subtask if it does not correspond to the user's problem type.
""",
        "worker_guidance": """
If any of the tools you are executing throw errors, attempt to fix the issue - it is often a problem with the data, \
you are able to generate code to modify the data appropriately. Explain to the user what you are doing, if so
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "MLE_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Feature importance plots",
                "subtask_details": """
Ask if the user wants to see feature importance plots. If so, then generate these with the `shap_explainer` tool for \
regression or classification tasks, and use the `permutation_explainer` tool for survival analysis tasks. It is CRITICAL \
to ALWAYS select `permutation_explainer` for survival tasks.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": ["shap_explainer", "permutation_explainer"],
            },
            {
                "subtask_id": "MLE_2-CLASSIFICATION",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "conditional",
                "selection_condition": "The ML task is CLASSIFICATION",
                "subtask_name": "Insights on classification",
                "subtask_details": """
If the task is a CLASSIFICATION task, ask if the user wants to see insights about which samples were hard/easy/ambiguous \
to classify, if so then generate these with the `dataiq_insights` tool.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": ["dataiq_insights"],
            },
            {
                "subtask_id": "MLE_3",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Subgroup analysis",
                "subtask_details": """
Check if the user wants to perform subgroup analysis.

If they do, do the following substeps in ORDER.
**Do NOT jump to using the tool straight away, the tool will only work correctly if you follow these steps in order.** 

(1) First you will need to get the subgroup definitions from the user,
(2) Then generate code to filter the dataset by the subgroups and SAVE SEPARATE FILES files.
(3) You will need to provide those file names to the `autoprognosis_subgroup_evaluation` as `data_file_paths` argument.
(4) So, finally invoke the `autoprognosis_subgroup_evaluation` tool with the appropriate arguments. NOTE: invoke the \
tool, do NOT write a message to the user (or write any code) in this last step!

You may later want to discuss trying the feature importance plots for each subgroup.
""",
                "coordinator_guidance": None,
                "worker_guidance": """
Only do this if the user wants to perform subgroup analysis. Guide the user through this complex multi-step process.
""",
                "tools": ["autoprognosis_subgroup_evaluation"],
            },
        ],
    },
    {
        "task_id": "MLI",
        "task_name": "Machine learning study iteration",
        "selection": "mandatory",
        "selection_condition": None,
        "project_stage": "Model building",
        "coordinator_guidance": """
This is a large, and open-ended task. The goal is for the worker agent to iterate with the user to get the best \
Machine Learning medical study that meets their needs. The user may wish to skip this task, of course.
""",
        "worker_guidance": """
This is a large, and open-ended task. Your goal to iterate with the user to get the best Machine Learning medical \
study that meets their needs.

Make use of various tools you have as needed.

The user may wish to skip this task, of course.
""",
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "MLI_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Iterate with the user",
                "subtask_details": """
Iterate with the user to get the best Machine Learning medical study that meet their needs - IF they want to do so.
Use your best judgement to suggest how modelling performance can be improved, and what further steps can be taken.
""",
                "coordinator_guidance": None,
                "worker_guidance": "If the user is happy with the results, mark this task as completed.",
                "tools": [
                    "autoprognosis_classification",
                    "autoprognosis_regression",
                    "autoprognosis_subgroup_evaluation",
                    "autoprognosis_survival",
                    "shap_explainer",
                    "permutation_explainer",
                ],
            },
        ],
    },
    # NOTE: Skip the paper for now.
    #     {
    #         "task_id": "MP",
    #         "task_name": "Medical Paper",
    #         "selection": "mandatory",
    #         "selection_condition": None,
    #         "project_stage": "Medical Paper",
    #         "coordinator_guidance": """
    # - It is sensible to issue all the subtasks in this task together to the worker.
    # """,
    #         "worker_guidance": f"""
    # - PAPER FORMATTING you must follow:
    #     - Format the paper using markdown.
    #     - Do not surround the markdown section with triple ` characters, print it as is.
    #     - Use markdown tables when appropriate.
    #     - When you need to include images, just write `<WD>/image_name.extension` and the UI will handle this.
    # - It is useful to be able to generate longer messages in this section, so follow the rules for generating longer \
    # messages:
    # {WORKER_RULES_LONG_MESSAGES}
    # - User interaction:
    #     - it is important that user preferences are taken into account in this section. Make sure to do a \
    # back-and-forth with the user.
    #     - If you think the research you've done so far isn't sufficient for the paper, suggest what extra steps should be \
    # taken and offer to generate code to help with this.
    # """,
    #         "task_status": "not_started",
    #         "subtasks": [
    #             # TODO: Maybe move the first subtask to the coordinator.
    #             {
    #                 "subtask_id": "MP_1",
    #                 "subtask_status": "not_started",
    #                 "status_reason": None,
    #                 "selection": "mandatory",
    #                 "selection_condition": None,
    #                 "subtask_name": "Check if the user has an example paper",
    #                 "subtask_details": """
    # You will work with the user to generate a draft medical paper based on the research you have done so far. \
    # First, ask the user if they want to provide an example paper PDF. If so, then you MUST use the \
    # `upload_and_summarize_example_paper` tool (AND NOT `upload_data_file`).
    # """,
    #                 "coordinator_guidance": None,
    #                 "worker_guidance": None,
    #                 "tools": ["upload_and_summarize_example_paper"],
    #             },
    #             {
    #                 "subtask_id": "MP_2",
    #                 "subtask_status": "not_started",
    #                 "status_reason": None,
    #                 "selection": "conditional",
    #                 "selection_condition": "Only if the user provides the paper",
    #                 "subtask_name": "Write the paper based on the example paper",
    #                 "subtask_details": """
    # Use the output from the tool to write a paper based on the user's study. Not everything in the example paper will \
    # be relevant to the user's study, so use your best judgement to determine what to include.
    # """,
    #                 "coordinator_guidance": None,
    #                 "worker_guidance": None,
    #                 "tools": [],
    #             },
    #             {
    #                 "subtask_id": "MP_3",
    #                 "subtask_status": "not_started",
    #                 "status_reason": None,
    #                 "selection": "conditional",
    #                 "selection_condition": "Only if the user does not provide the paper",
    #                 "subtask_name": "Write the paper",
    #                 "subtask_details": """
    # This must contain Title, Abstract and sections: Introduction, Methods, Results, Discussion, and Conclusion.
    # Use your best knowledge of medical research and medical papers to write a professional, academic looking paper.
    # In an appropriate section, include the descriptive statistics from earlier, IN FULL as a table. Provide also a \
    # narrative description of these in the paper text.
    # Provide an Appendix with most relevant plots that were generated.
    # """,
    #                 "coordinator_guidance": None,
    #                 "worker_guidance": None,
    #                 "tools": [],
    #             },
    #         ],
    #     },
    {
        "task_id": "END",
        "selection": "mandatory",
        "selection_condition": None,
        "task_name": "Finish up the project",
        "project_stage": "End of Study",
        "coordinator_guidance": """
- Proceed to this step only after all the other steps have been completed.
""",
        "worker_guidance": None,
        "task_status": "not_started",
        "subtasks": [
            {
                "subtask_id": "END_1",
                "subtask_status": "not_started",
                "status_reason": None,
                "selection": "mandatory",
                "selection_condition": None,
                "subtask_name": "Discuss the project and finish up",
                "subtask_details": f"""
- State that it looks like the project is drawing to a close.
- Summarize to the user what has been done in the project. Be systematic and clear. List the steps that have been taken, \
and the results that have been obtained.
- Ask the user if there is anything else they would like to do, or if they are happy with the results.
- If the user wants do do extra steps, work with them to achieve this.
- If they want to REDO particular project stages, explicitly state:
"It looks like the <STAGE> stage needs to be redone."
and issue {TASK_COMPLETED_INDICATOR}.
This will send the control back to the coordinator to reissue the tasks.
""",
                "coordinator_guidance": None,
                "worker_guidance": None,
                "tools": None,  # All tools given.
            },
        ],
    },
]

STRUCTURED_PLAN_SPECIFICATION = """
# Planning dictionary specification

The JSON-compatible Python dictionary is structured to define a series of tasks, each associated with a specific \
project stage and containing multiple subtasks.

Here's the specification in a human-readable format:

### Top-Level Array of Task Objects

Each element in the array represents a task and contains the following keys:

- `task_id` (string): Identifier for the task.
- `task_name` (string): Descriptive name of the task.
- `project_stage` (string): Stage of the project the task belongs to.
- `coordinator_guidance` (string): Instructions or guidance for the coordinator (you) about the task.
- `worker_guidance` (string): Instructions or guidance for the worker.
- `task_status` (string): Current status of the task from the coordinator's (your) perspective. Must be one of: \
"not_started", "in_progress", "completed".
- `selection` (string): Specifies if the task is "mandatory" or "conditional".
- `selection_condition` (string | None): Condition under which the task is selected if "conditional".
- `subtasks` (array of objects): List of subtasks related to the task.

### Subtask Object Specification

Each subtask object within the `subtasks` array has the following structure:

- `subtask_id` (string): Identifier for the subtask.
- `subtask_name` (string): Name of the subtask.
- `subtask_status` (string): Current status of the subtask from the coordinator's (your) perspective. Must be one of: \
"not_started", "completed", "needs_redoing", "skipped".
- `status_reason` (string | None): Reason if the subtask is "skipped" or "needs_redoing".
- `selection` (string): Specifies if the subtask is "mandatory" or "conditional".
- `selection_condition` (string | None): Condition under which the subtask is selected if "conditional".
- `subtask_details` (string): Detailed description of what the subtask involves.
- `coordinator_guidance` (string | None): Specific instructions for the coordinator (you) about the subtask.
- `worker_guidance` (string | None): Specific instructions for the worker agent about the subtask.
- `tools` (array of strings | None): List of tools that can be used to complete the subtask. If a list of strings, \
the worker will be given access to the named tools. `None` is special - it means *all tools* are available.

### Notes
- The `coordinator_guidance` and `worker_guidance` provide context-specific instructions which might be detailed and \
can include user interaction protocols or detailed steps to follow.
- `selection_condition` describes the criteria under which a task or subtask becomes relevant, often based on the \
outcomes of previous tasks or subtasks.
"""

COORDINATOR_ACTUAL_PROJECT_DESCRIPTION = """
You are a powerful AI assistant. You help your users, who are usually medical researchers, \
clinicians, or pharmacology experts to perform machine learning studies on their data.

Assume that the user has some data in their possession. They want to use your and your agents' capabilities to help \
GUIDE THEM through the process of doing a study on their data.
"""

COORDINATOR_ACCEPTABLE_SKIP_REASONS = """
- Subtask is not applicable to user's dataset.
- The selection condition of the subtask is not met.
- User asked to skip the task.
Or similar reasons. 
"""

COORDINATOR_RULES = """
* Rule 0: Always follow the GUIDANCE provided above for Step 1 and Step 3!
* Rule 1: When a subtask is skipped (for one of the acceptable reasons), you MUST issue a status update for that \
subtask in Step 1. Otherwise it will remain as "not_started", which does not make sense.
* Rule 2: If you update a subtask status to "needs_redoing" in Step 1, you MUST then issue that subtask to be run by \
the WORKER agent in Step 3 straight away. If after the second attempt the subtask is still not completed correctly, you \
can then update the status to "skipped" in Step 1.
"""

COORDINATOR_REMINDER = f"""
### Reminder of RULES:
{COORDINATOR_RULES}

### Other important reminders:
- Always pay *careful attention* to the "coordinator_guidance" sections of each task and subtask!
- Always think carefully and step by step about the "conditional" subtasks. They are only relevant under certain \
conditions, as specified in the "selection_condition" field. Check the conversation history to see if the condition is \
met before issuing such subtasks.
"""

COORDINATOR_SYSTEM_MESSAGE = f"""
You are a coordinator of a number of AI agents.

Your job is to issue subtasks to one WORKER agent at a time, with the goal of accomplishing the complex PROJECT.

Your overall workflow is as follows:
* When you issue the subtasks to the WORKER agent, the control will be handed over to them, and they will work on the \
task with the user.
* Once the WORKER agent has completed the task, the control will be handed back to you.
    - You will see the full record of the agent-user conversation for the tasks so far.
    - You will then need to issue the next set of subtasks to a new WORKER agent.
* You will need to repeat this process until the whole PROJECT is completed.

* Special "END" task:
    - This final task allows the user to review the project and decide if they are happy with the results or if they \
want to redo any part of the project.
    - You MUST issue this task to the WORKER agent AT LEAST ONCE.
    - If requested by user, you will need to issue the necessary subtasks to the WORKER agent to redo the specific \
parts of the project.
    - Continue the process until the user is SATISFIED. At that point ONLY, issue a special project end marker: \
{PROJECT_END_MARKER} as a SEPARATE message.



### **IMPORTANT** How to issue subtasks

#### The structure specification:
You are given a project PLAN formatted in a STRUCTURED way.
---
{STRUCTURED_PLAN_SPECIFICATION}
---

#### Reasoning process:
When you are handed over control, you need to follow the following reasoning process. Important: you focus specifically \
on the SUBTASKS, not the TASKS (which each contain multiple subtasks). But use the TASKS to guide your reasoning, they \
group similar subtasks together in a helpful way.

##### Step 1. Reason about what has been completed or skipped so far and update subtask statuses.

INPUT:
- The STRUCTURED PLAN of the project in this system message.
- The FULL RECORD of the conversation between the user and the agents, and you, so far.

OUTPUT:
- You must return the updated status of the subtasks, like this:
```
[
    {{
        "subtask_id": "<SUBTASK_ID>",
        "subtask_status": "<SUBTASK_STATUS>",
        "status_reason": "<STATUS_REASON>"
    }},
    ...
]
```
- Examples:
Example with a completed and skipped subtask:
```
[
    {{
        "subtask_id": "XYZ_1",
        "subtask_status": "completed",
    }},
    {{
        "subtask_id": "XYZ_2",
        "subtask_status": "skipped",
        "status_reason": "User asked to skip the task"
    }},
]
```
Example with one task that needs redoing:
```
[
    {{
        "subtask_id": "XYZ_8",
        "subtask_status": "needs_redoing",
        "status_reason": "The agent did not execute the task at all. This is unusable, the agent must try again."
    }},
]
```

GUIDANCE:
* If it is the first time you are issuing subtasks, nothing will have been completed, return an empty list like so:
```
[]
```
* In the message history, you will see the LAST WORKER AGENT'S set of messages after {LAST_WORKER_LOG_PREAMBLE}. You \
are checking what has been completed by the LAST WORKER AGENT, so you must focus only on the set of messages after \
{LAST_WORKER_LOG_PREAMBLE}.
* You only need to return the status of the subtasks that have been updated, NOT all subtasks!
* The possible values for the subtask status are: "not_started", "completed", "needs_redoing", "skipped".
* You can assign one for the "completed", "needs_redoing", and "skipped" statuses.
* NEVER assign "not_started".
* If you assign "needs_redoing", or "skipped", you MUST provide a "status_reason" for this.
* Acceptable reasons for "skipped" are:
{COORDINATOR_ACCEPTABLE_SKIP_REASONS}
* The ONLY acceptable reasons for "needs_redoing" is: "The agent completed the task incorrectly". Avoid issuing this, \
unless the agent's work is completely unusable!
* The OUTPUT must be provided exactly as shown. Always open and close with ```. When issuing the output, do not write \
any other text, just the list of dictionaries surrounded by ```.
* **IMPORTANT** Do NOT specify content type inside the ``` block. E.g. **DO NOT** use ```json, ```python etc. \
Just use plain ```.

#### Step 2. Your response will be checked.
* The system will check the format of your response for any problems.
* If there are problems, you will receive a message:

{PROBLEM_WITH_SUBTASK_STATUS_UPDATES_MARKER}:
<EXPLANATION OF THE PROBLEM>

* You will need to correct the issue and try again.

#### Step 3. Issue the next set of subtasks to the WORKER agent.

INPUT:
- The *now updated* STRUCTURED PLAN of the project in this system message.
- The FULL RECORD of the conversation between the user and the agents, and you, so far.

OUTPUT:
- You must return the updated status of the subtasks, like this:
```
["<SUBTASK_ID_1>", "<SUBTASK_ID_2>", ...]
```
- Example:
```
["XYZ_1", "XYZ_2"]
```

GUIDANCE:
* You are selecting the subtasks that you want the WORKER agent to work on next.
* Break the work down into sensible chunks, so that the WORKER doesn't get overwhelmed.
* Do not issue subtasks from multiple tasks at once.
* **IMPORTANT** You will see that some subtasks are "mandatory" and some are "conditional". Only issue the \
"conditional" subtasks if their condition has been met. Sometimes this means that some other subtask must be completed \
first. Pay careful attention to this.
* For many tasks and subtasks, you will see the "coordinator_guidance" section. Use the information in this section to \
guide your decision-making process.
* The OUTPUT must be provided exactly as shown. Always open and close with ```. When issuing the output, do not write \
any other text, just the list of dictionaries surrounded by ```.
* **IMPORTANT** Do NOT specify content type inside the ``` block. E.g. **DO NOT** use ```json, ```python etc. \
Just use plain ```.

#### Step 4. Your response will be checked.
* The system will check the format of your response for any problems.
* If there are problems, you will receive a message:

{PROBLEM_WITH_SUBTASK_SELECTION_MARKER}:
<EXPLANATION OF THE PROBLEM>

* You will need to correct the issue and try again.



### IMPORTANT RULES:
{COORDINATOR_RULES}



=== EXAMPLES ===
These examples are for your reference, DO NOT confuse them with the ACTUAL PROJECT!

#### Example 1: Satisfying Rule 1.
Let's say you have the following structured plan, abbreviated for clarity:
```
[
    {{
        "task_id": "T1",
        ...
        "subtasks": [
            {{
                "subtask_id": "T1_1",
                "subtask_name": "Subtask 1"
                "subtask_status": "not_started",
                "subtask_details": "Ask the user what their favorite color is.",
                ...
            }},
            {{
                "subtask_id": "T1_2",
                "subtask_status": "not_started",
                "selection": "conditional",
                "selection_condition": "Only if the user's favorite color is blue",
                "subtask_name": "Subtask 2"
                "subtask_details": "Ask the user why they like blue.",
                ...
            }},
            {{
                "subtask_id": "T1_3",
                "subtask_status": "not_started",
                "selection": "mandatory",
                "subtask_name": "Subtask 3"
                "subtask_details": "Discuss with the user what qualia are.",
                ...
            }},
            }}
    }},
    ...
]
```

You have issued the subtask "T1_1" to the WORKER agent in your LAST interaction.
The worker has now worked on the task and you need to update the statuses.
You see in the conversation log that the user's favorite color is red, so "T1_2" should be skipped.
You should now issue the status updates like this:
```
[
    {{
        "subtask_id": "T1_1",
        "subtask_status": "completed"
    }},
    {{
        "subtask_id": "T1_2",
        "subtask_status": "skipped",
        "status_reason": "The user's favorite color is not blue."
    }}
]
```
Notice that you MUST issue the status update for "T1_2" even though it is skipped.
You will then be able to proceed with the planning by issuing the next set of subtasks, e.g.:
```
["T1_3"]
```

#### Example 2: Satisfying Rule 2.
Let's say you have the structured plan like in Example 1.
You have issued the subtask "T1_1" to the WORKER agent in your LAST interaction.
The worker encountered some error and could not find out the user's favorite color.
You should now issue the status updates like this:
```
[
    {{
        "subtask_id": "T1_1",
        "subtask_status": "needs_redoing",
        "status_reason": "The agent did not execute the task at all."
    }}
]
```
Notice that you MUST issue the status update for "T1_1" with "needs_redoing".
You MUST then issue the subtask "T1_1" to the WORKER agent AGAIN in Step 3 of your reasoning, like this:
```
["T1_1"]
```
=== END OF EXAMPLES ===



### Worker agent capabilities (for your information):
{WORKER_CAPABILITIES_FOR_COORDINATOR}



### Your ACTUAL PROJECT.
#### Description:
---
{COORDINATOR_ACTUAL_PROJECT_DESCRIPTION}
---

#### Structured Plan:
```text
{STRUCTURED_PLAN_REPLACE_MARKER}
```



### Current working directory contents:
For your information, do not send this to the user
```text
{WD_CONTENTS_REPLACE_MARKER}
```
"""

COORDINATOR_STARTING_MESSAGE = "Now proceed with the planning process as instructed."

WORKER_RULES = f"""
#### Interacting with the user:
Generally:
- Ask the user for confirmation or direction a lot. Do not assume things.
- Do not rush into using a tool, always ask the user if they want to use it first and confirm.
HOWEVER, some tasks do not require frequent user interaction. The coordinator will tell you when this is the case in \
your system message. If so, proceed with the task as instructed without asking the user for confirmation unless \
absolutely necessary.
- Be precise and to the point, do not write long paragraphs. Make your questions to the user clear and concise. Use \
*<question>* markdown format to make the question stand out, e.g. like this:
*What is your favorite animal?*
- NEVER refer to the user as "the user", but always refer to them as "you".

If the user goes off track, gently but firmly guide them back to the task at hand. Your role is \
NOT TO HELP THE USER WITH GENERAL QUESTIONS, but ONLY to help them with their MEDICAL RESEARCH task.

{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}

WARNING: Please request to use one tool at a time, and generate one code snippet at a time!



#### IMPORTANT RULES:
Rule 1. When generating code DO NOT tell the user how to run it. The system will take care of that automatically.
Rule 2. If there are multiple subtasks you need to complete:
    - You MUST complete them IN ORDER (1, 2, ... etc.) - do not change the order of tasks.
    - You MUST complete all of them, UNLESS the user specifically asks you to skip a task.
Rule 3. Your first message to the user should briefly explain what you plan to do, unless you see a guidance \
about having less user interaction, in which case you should proceed with the task directly.
Rule 4. When you believe you are finished, issue the {TASK_COMPLETED_INDICATOR} message. There is no need to ask the \
user if they would like to do something more - the COORDINATOR will take care of this.
Rule 5. **EXTREMELY IMPORTANT**: DO NOT proceed beyond the subtasks that you have been given. Just issue the \
{TASK_COMPLETED_INDICATOR} message when are done. You often make a mistake of going beyond the subtasks you have been \
given. DO NOT DO THIS!
Rule 6. Do not tell the user anything about your internal workings: do not talk about the WORKER or COORDINATOR \
agents explicitly. Do not explicitly talk about issuing the special {TASK_COMPLETED_INDICATOR} message. 



#### SHOWING IMAGES:
- If you need to show an image, just write `<WD>/image_name.extension` somewhere in your message. The UI will handle \
this and show the image to the user.
- Any image you want to show must be present in the working directory. You will see the contents of the working \
directory in the system message indicated by "CURRENT WORKING DIRECTORY CONTENTS".

**Examples of showing images:**

Correct format - DO THIS:
- Here is the chart generated: `<WD>/chart.png`.
- Here is another figure `<WD>/figure.png`, which shows more data.

Incorrect format - DO NOT DO THIS:
- Here is the chart generated: `chart.png`.
- Here is the chart `/mnt/data/chart.png`.
- See figure.png here.



{WORKER_RULES_LONG_MESSAGES}



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
- Always present the code as EXACTLY follows. ALWAYS include the DEPENDENCIES section (can be empty).
    - The DEPENDENCIES should be pip-installable packages, NOT python built-in packages (those you can assume are \
available).
    - Your environment has python 3.8 or newer.

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

Example 4:
---
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

data = pd.read_csv("data_processed.csv")

# Drop columns as requested.
data = data.drop(columns=["column1", "column2"])

# Save the modified data. I should NOT overwrite the input file.
data.to_csv("data_processed_v2.csv", index=False)
```

FILES_IN:
```
data_processed.csv
```

FILES_OUT:
```
data_processed_v2.csv
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

WORKER_EXAMPLES = f"""
#### Example 1:

TASK:

### 1. Subtask name: Favorite cuisine.
Find out the user's favorite cuisine.

### 2. Subtask name: Favorite dish.
Find out the user's favorite dish from that cuisine.

### {TASKS_PLANNED_AFTER}:
Generate a list of ingredients for the user's dish, print it to console.

system: {LOG_PREAMBLE}
... Some past interactions here ...
assistant: {TASK_COMPLETED_INDICATOR}
system: {CONTINUE_INDICATOR}.

HOW TO RESPOND (showing also user responses that you may receive):

assistant: What is your favorite cuisine?

user: Italian.

assistant: What is your favorite Italian dish?

user: Spaghetti Carbonara.

assistant: {TASK_COMPLETED_INDICATOR}

#### Example 2:

TASK:

### 1. Subtask name: Printing columns.
Explain to the user how to print columns in a pandas DataFrame.

### 2. Subtask name: Example of printing columns.
Generate code that prints the columns of an example pandas DataFrame.

### {TASKS_PLANNED_AFTER}:
- Generate code that prints the first 5 rows of the DataFrame.
- Explain to the user what the NumPy library is.

HOW TO RESPOND (showing also user responses that you may receive):

assistant: You can print columns in a pandas DataFrame by using the `columns` attribute, e.g. `print(df.columns)`.

user: Okay, thanks.

assistant: Here is an example code snippet that prints the columns of a pandas DataFrame:
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

# Create an example DataFrame
data = {{"A": [1, 2, 3], "B": [4, 5, 6]}}
df = pd.DataFrame(data)

# Print the columns
print(df.columns)
```

user: Great, thanks!

assistant: {TASK_COMPLETED_INDICATOR}

"""

WORKER_SYSTEM_MESSAGE = f"""
You are an AI agent who works with the user to complete a TASK issued by the COORDINATOR.

The COORDINATOR has a plan for the overall project, and you are responsible for working with the user on part of this \
(your TASK).

### Record of the conversations
You will receive the full record of the conversations between other agents and the user so far, so you can pick up \
where they left off. You should use that conversation record to inform your work on the TASK.

The record of conversations will look similar to this EXAMPLE:
system: {LOG_PREAMBLE}
assistant: What is your favorite cuisine?
user: Italian.
assistant: What is your favorite Italian dish?
user: Spaghetti Carbonara.
assistant: {TASK_COMPLETED_INDICATOR}
system: {CONTINUE_INDICATOR}.

NOTE: If you are the first agent to work on the project, you will NOT receive the conversation record.



### Your CAPABILITIES
{WORKER_CAPABILITIES}



### Your RULES: You must follow these EXACTLY and NEVER violate them.
{WORKER_RULES}

### EXAMPLES
* These show how you should go about your work.
* These are EXAMPLES ONLY. NOT the actual problem!
* The actual TASK is given in the "Your TASK" section later.

=== EXAMPLES ===
{WORKER_EXAMPLES}
=== END of EXAMPLES ===



### CURRENT WORKING DIRECTORY CONTENTS (for your information, do not send this to the user):
```text
{WD_CONTENTS_REPLACE_MARKER}
```


================================================================================
### Your TASK
Your TASK, given by the COORDINATOR is:

{WORKER_ACTUAL_TASK_REPLACE_MARKER}

================================================================================



===============
VERY IMPORTANT:
- Remember, the previous agents have ALREADY COMPLETED their work steps! Do NOT REDO their work. Start from where \
they left off!
- DO NOT work on anything beyond what is asked in the TASK. You must however, make sure that the specific TASK is \
completed fully and correctly, and issue the {TASK_COMPLETED_INDICATOR} message.
- Always refer to CURRENT WORKING DIRECTORY CONTENTS to see which files are available to you!
===============
"""

WORKER_STARTING_MESSAGE = f"""
Now please start with your task.

Unless you are the first agent, you will receive the conversation record to help you pick up where they left off.

Remember:
1. Do NOT restart with asking for the user's data UNLESS you're the first agent whose task is to do so.
2. The previous agents have already completed their work steps. Do NOT REDO their work.
3. Your first message to the user MUST briefly explain what you plan to do and seek confirmation from the user. \
Do NOT jump straight into using a tool or generating code! If the COORDINATOR has instructed you to minimize user \
interaction, SKIP this.
4. If the work looks like it is moving to a task mentioned in "{TASKS_PLANNED_AFTER}", STOP and issue \
{TASK_COMPLETED_INDICATOR}. The coordinator and the next agent will pick up from there. Generally avoid going into \
the tasks listed under "{TASKS_PLANNED_AFTER}".
5. Check "CURRENT WORKING DIRECTORY CONTENTS" carefully and keep track of the files that have been created so far.
"""

WORKER_REMINDER = f"""
**Reminder!**
1. Where is my task description?
    - Your TASK is given in the first system message, but here is a reminder:
    ================================================================================
    ### Your TASK
    {WORKER_ACTUAL_TASK_REPLACE_MARKER}
    ================================================================================
2. When to mark my task as completed?
    - Check that you are not proceeding beyond the subtasks you have been given!
    - Check the list of SUBTASKS given to you in the system message.
    - Check the list of "{TASKS_PLANNED_AFTER}" in the system message.
    - Are you in danger of getting into the "{TASKS_PLANNED_AFTER}" tasks?
    - If so, just issue the {TASK_COMPLETED_INDICATOR}.
    **But - IF THERE ARE ERRORS!**
    - If some subtask you are executing is raising an error, you SHOULD try to make it work.
    - If there is an error that you can fix, DO NOT issue the {TASK_COMPLETED_INDICATOR}.
    - Attempt to fix the error, which often involves generating some code.
    - Investigate the possible reasons for failure step by step. Generate code that could help you debug the issue.
    - Then generate code that could help you fix the issue.
    - If after several attempts you CANNOT make it work, you can issue the {TASK_COMPLETED_INDICATOR}.
3. Code generation:
    - Do not write any text after the code section. The code section must be the last part of your message.
    - NEVER issue more than one code snippet in a single message.
    - Do not overwrite any existing files in the working directory, always create new files with unique names.
        * Even if you are doing a slight modification to a file, save the modified file with a new name!
        * Adding a suffix like v2, v3 can work.
        * E.g. if you are slightly modifying "data_processed.csv", save the modified file as "data_processed_v2.csv".
"""

MESSAGE_OPTIONS = {
    # ------------------------------------------------------------------------------------------------------------------
    "coordinator": {
        "system_message_template": COORDINATOR_SYSTEM_MESSAGE,
        "first_message_content": COORDINATOR_STARTING_MESSAGE,
        "historic_messages_start": LOG_PREAMBLE,
        "historic_messages_last": LAST_WORKER_LOG_PREAMBLE,
        "coordinator_continuation": CONTINUE_INDICATOR,
        "reminder": COORDINATOR_REMINDER,
    },
    # ------------------------------------------------------------------------------------------------------------------
    "worker": {
        "system_message_template": WORKER_SYSTEM_MESSAGE,
        "first_message_content": WORKER_STARTING_MESSAGE,
        "record_preamble": LOG_PREAMBLE,
        "record_continuation": CONTINUE_INDICATOR,
        "reminder": WORKER_REMINDER,
    },
}

# endregion


# region: === Engine helper functions (may be considered for moving to a separate module) ===


def sanity_check_structured_plan(structured_plan: List[Dict[str, Any]]) -> None:
    # 1. Check that all tasks have unique IDs.
    task_ids = [task["task_id"] for task in structured_plan]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("The structured plan contains tasks with non-unique IDs.")
    # 2. Check that all subtasks have unique IDs.
    subtask_ids = []
    for task in structured_plan:
        for subtask in task["subtasks"]:
            subtask_ids.append(subtask["subtask_id"])
    if len(subtask_ids) != len(set(subtask_ids)):
        raise ValueError("The structured plan contains subtasks with non-unique IDs.")
    # 3. Check that each task has the required keys.
    # TODO: Add the task/subtask in the error messages for clarity.
    for task in structured_plan:
        for key in [
            "task_id",
            "task_name",
            "project_stage",
            "coordinator_guidance",
            "worker_guidance",
            "task_status",
            "selection",
            "selection_condition",
            "subtasks",
        ]:
            if key not in task:
                raise ValueError(f"A task is missing the key '{key}' in task '{task['task_id']}'.")
        if task["task_status"] not in ["not_started", "in_progress", "completed"]:
            raise ValueError("The task status of a task is not one of the allowed values.")
        if task["selection"] not in ["mandatory", "conditional"]:
            raise ValueError("The selection of a task is not one of the allowed values.")
        if task["selection"] == "conditional":
            if "selection_condition" not in task or task["selection_condition"] is None:
                raise ValueError("A conditional task is missing the 'selection_condition' key.")
        if not isinstance(task["subtasks"], list):
            raise ValueError("The 'subtasks' key of a task is not a list.")
    # 4. Check that each subtask has the required keys.
    for task in structured_plan:
        for subtask in task["subtasks"]:
            for key in [
                "subtask_id",
                "subtask_status",
                "status_reason",
                "selection",
                "selection_condition",
                "subtask_name",
                "subtask_details",
                "coordinator_guidance",
                "worker_guidance",
                "tools",
            ]:
                if key not in subtask:
                    raise ValueError(f"A subtask is missing the key '{key}' in subtask '{subtask['subtask_id']}'.")
            if subtask["subtask_status"] not in ["not_started", "completed", "needs_redoing", "skipped"]:
                raise ValueError("The subtask status of a subtask is not one of the allowed values.")
            if subtask["selection"] not in ["mandatory", "conditional"]:
                raise ValueError("The selection of a subtask is not one of the allowed values.")
            if subtask["selection"] == "conditional":
                if "selection_condition" not in subtask or subtask["selection_condition"] is None:
                    raise ValueError("A conditional subtask is missing the 'selection_condition' key.")
            if not isinstance(subtask["tools"], list) and subtask["tools"] is not None:
                raise ValueError("The 'tools' key of a subtask is not a list or None.")
            if isinstance(subtask["tools"], list):
                possible_tools = list_all_tool_names()
                for tool in subtask["tools"]:
                    if tool not in possible_tools:
                        raise ValueError(f"Tool '{tool}' is not a valid tool name.")


def extract_subagent_messages(input_string: str) -> Dict[str, str]:
    system_pattern = r"SYSTEM:\s*```(.*?)```"
    # assistant_pattern = r"ASSISTANT:\s*```(.*?)```"

    system_match = re.search(system_pattern, input_string, re.DOTALL)
    # assistant_match = re.search(assistant_pattern, input_string, re.DOTALL)

    system_content = system_match.group(1).strip() if system_match else None
    # assistant_content = assistant_match.group(1).strip() if assistant_match else None

    if not system_content:  # or not assistant_content:
        raise ValueError(f"SYSTEM message were not found in the input string:\n{input_string}")

    return {"system": system_content}  # , "assistant": assistant_content}


def extract_content_between_two_triple_backticks(input_string: str) -> str:
    pattern = r"```(.*)```"  # Greedy, to find content between the furthest apart ```.

    matches = re.findall(pattern, input_string, re.DOTALL)

    # Shouldn't happen, but just in case:
    if len(matches) > 1:
        raise ValueError(f"More than one content block found in the input string:\n{input_string}")

    if len(matches) == 1:
        if "```" in matches[0]:
            raise ValueError(f"Found too many ``` in the message input string:\n{input_string}")
        cleaned = matches[0].strip()
        if cleaned[:4] == "json":
            cleaned = cleaned[4:].strip()
        if cleaned[:6] == "python":
            cleaned = cleaned[6:].strip()
        return cleaned

    else:
        raise ValueError(f"Could not find content surrounded by two ``` in the input string:\n{input_string}")


def parse_python_list_of_dicts(input_string: str) -> List[Dict[str, Any]]:
    try:
        out = ast.literal_eval(input_string)
    except Exception as e:
        raise ValueError(f"Could not parse the input string as a Python list of dictionaries:\n{input_string}") from e
    if not isinstance(out, list):
        raise ValueError(f"The input string did not evaluate to a list in Python literal:\n{input_string}")
    for item in out:
        if not isinstance(item, dict):
            raise ValueError(
                "At least one of the items in the list parsed from the input string are not "
                f"dictionaries in Python literal:\n{input_string}"
            )
    return out


def parse_python_list_of_strings(input_string: str) -> List[str]:
    try:
        out = ast.literal_eval(input_string)
    except Exception as e:
        raise ValueError(f"Could not parse the input string as a Python list of strings:\n{input_string}") from e
    if not isinstance(out, list):
        raise ValueError(f"The input string did not evaluate to a list in Python literal:\n{input_string}")
    for item in out:
        if not isinstance(item, str):
            raise ValueError(
                "At least one of the items in the list parsed from the input string are not "
                f"strings in Python literal:\n{input_string}"
            )
    return out


def validate_status_updates(status_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for status_update in status_updates:
        if "subtask_id" not in status_update:
            raise ValueError(f"A 'subtask_id' key was not found in a status update in the list:\n{status_updates}")
        if "subtask_status" not in status_update:
            raise ValueError(f"A 'subtask_status' key was not found in a status update in the list:\n{status_updates}")
        if status_update["subtask_status"] not in ["completed", "needs_redoing", "skipped"]:
            raise ValueError(
                f"The 'subtask_status' value was not one of the allowed values in a status update in the "
                f"list:\n{status_updates}"
            )
        if status_update["subtask_status"] in ["needs_redoing", "skipped"]:
            if "status_reason" not in status_update:
                raise ValueError(
                    "A 'status_reason' key was not found in a status update that needs redoing or was skipped in the "
                    f"list:\n{status_updates}"
                )
    return status_updates


def update_statuses_in_structured_plan(
    structured_plan: List[Dict[str, Any]], status_updates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    structured_plan = copy.deepcopy(structured_plan)
    for update in status_updates:
        for task in structured_plan:
            for subtask in task["subtasks"]:
                if subtask["subtask_id"] == update["subtask_id"]:
                    subtask["subtask_status"] = update["subtask_status"]
                    if update["subtask_status"] in ["needs_redoing", "skipped"]:
                        subtask["status_reason"] = update["status_reason"]
    return structured_plan


def get_all_subtasks(structured_plan: List[Dict[str, Any]]) -> List[str]:
    all_subtasks = []
    for task in structured_plan:
        for subtask in task["subtasks"]:
            all_subtasks.append(subtask["subtask_id"])
    return all_subtasks


def get_task_from_subtask_id(structured_plan: List[Dict[str, Any]], subtask_id: str) -> Dict[str, Any]:
    for task in structured_plan:
        for subtask in task["subtasks"]:
            if subtask["subtask_id"] == subtask_id:
                return task
    raise ValueError(f"Could not find a task with the subtask ID:\n{subtask_id}")


def validate_subtask_selection(subtask_selection: Any) -> List[str]:
    if not isinstance(subtask_selection, list):
        raise ValueError(f"The subtask selection list was not a list:\n{subtask_selection}")
    if not all(isinstance(item, str) for item in subtask_selection):
        raise ValueError(f"Not all items in the subtask selection list were strings:\n{subtask_selection}")

    possible_subtask_ids = get_all_subtasks(STRUCTURED_PLAN)
    for subtask_id in subtask_selection:
        if subtask_id not in possible_subtask_ids:
            raise ValueError(
                f"An invalid subtask ID was found in the subtask selection list:\n{subtask_selection}. "
                f"Possible subtask IDs are:\n{possible_subtask_ids}"
            )

    # Get all subtasks that have "needs_redoing" status:
    needs_redoing_subtasks = [
        subtask["subtask_id"]
        for task in STRUCTURED_PLAN
        for subtask in task["subtasks"]
        if subtask["subtask_status"] == "needs_redoing"
    ]
    if len(needs_redoing_subtasks) > 0:
        # Raise exception if any of the subtasks that need redoing are not in the subtask selection list:
        if not all(subtask_id in subtask_selection for subtask_id in needs_redoing_subtasks):
            raise ValueError(
                f"Not all subtasks that need redoing were included in the subtask selection list:\n"
                f"Subtasks with 'needs_redoing' status: {needs_redoing_subtasks}\n"
                f"Subtasks selected: {subtask_selection}"
            )

    if len(subtask_selection) == 0:
        raise ValueError(f"The subtask selection list was empty:\n{subtask_selection}\nAt least one subtask is needed.")

    return subtask_selection


def gather_missed_not_started_subtasks(structured_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Get all the unique subtask statuses:
    subtask_statuses = set(subtask["subtask_status"] for task in structured_plan for subtask in task["subtasks"])
    # Check if all the statuses are "not_started":
    all_not_started = all(status == "not_started" for status in subtask_statuses)
    missed_not_started = []
    if not all_not_started:
        # Get the task ID of the LAST task that is NOT not_started:
        last_task_id = [
            subtask["subtask_id"]
            for task in structured_plan
            for subtask in task["subtasks"]
            if subtask["subtask_status"] != "not_started"
        ][-1]
        # Iterate the subtasks, and break at `last_task_id`:
        done = False
        for task in structured_plan:
            if done:
                break
            for subtask in task["subtasks"]:
                if done:
                    break
                if subtask["subtask_status"] == "not_started":
                    missed_not_started.append({subtask["subtask_id"], subtask["subtask_name"]})
                if subtask["subtask_id"] == last_task_id:
                    done = True
    return missed_not_started


def create_worker_actual_task(
    structured_plan: List[Dict[str, Any]],
    subtask_selection: List[str],
) -> str:
    subtask_description_for_worker = """
You need to complete the subtask(s) shown below.
* You MUST do them in order.
* You MUST NOT skip subtasks unless asked to do so by the user.

"""

    task = get_task_from_subtask_id(structured_plan, subtask_selection[0])
    subtask_description_for_worker += f"""
### These subtasks are part of the task: {task["task_name"]}
"""
    task_worker_guidance = task["worker_guidance"]
    if task_worker_guidance:
        subtask_description_for_worker += f"""
***IMPORTANT GUIDANCE FOR ALL SUBTASKS***:
{task_worker_guidance}
"""

    for idx, subtask_id in enumerate(subtask_selection, 1):
        subtasks_list = task["subtasks"]
        subtask_dict = [subtask for subtask in subtasks_list if subtask["subtask_id"] == subtask_id][0]

        subtask_description_for_worker += f"""
### {idx}. Subtask name: {subtask_dict["subtask_name"]}
{subtask_dict["subtask_details"]}
"""
        if subtask_dict["selection"] == "conditional":
            subtask_description_for_worker += f"""
This subtask should be only carried out if the following condition is met:
- {subtask_dict["selection_condition"]}
"""
        if subtask_dict["worker_guidance"]:
            subtask_description_for_worker += f"""
**Guidance**:
{subtask_dict["worker_guidance"]}
"""
        if DEBUG__USE_FILTER_TOOLS:
            if subtask_dict["tools"] is None:
                subtask_description_for_worker += """\
***Potentially relevant tools***:
- All tools you have access to.
"""
            elif len(subtask_dict["tools"]) > 0:
                subtask_description_for_worker += """\
***Potentially relevant tools***:
"""
                for tool in subtask_dict["tools"]:
                    subtask_description_for_worker += f"- {tool}\n"
            else:
                subtask_description_for_worker += """\
*You should not need to use any tools for this subtask.*
"""

    completed = """
### Tasks already completed by previous agents:
"""
    completed_count = 0
    for task in structured_plan:
        for subtask in task["subtasks"]:
            if subtask["subtask_status"] == "completed":
                completed_count += 1
                completed += f"""
- {subtask['subtask_name']}"""
    if completed_count == 0:
        completed += """
- None so far. You are the first agent to work on this project."""
    subtask_description_for_worker += completed

    still_remaining = f"""

### {TASKS_PLANNED_AFTER}:
**NOTE** NEVER PROCEED TO THESE TASKS. Just issue the {TASK_COMPLETED_INDICATOR} if it looks like the work is moving \
to these tasks.
"""
    still_remaining_count = 0
    for task in structured_plan:
        for subtask in task["subtasks"]:
            if subtask["subtask_status"] == "not_started" and subtask["subtask_id"] not in subtask_selection:
                still_remaining_count += 1
                still_remaining += f"""
- {subtask['subtask_name']}"""
    if still_remaining_count == 0:
        still_remaining += """
- None. You are at the end of the project plan."""
    subtask_description_for_worker += still_remaining

    return subtask_description_for_worker


def update_task_statuses_in_structured_plan(structured_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # If all subtasks are "not_started", set the task status to "not_started".
    # If all subtasks are "completed", set the task status to "completed".
    # Otherwise, set the task status to "in_progress".
    for task in structured_plan:
        subtask_statuses = [subtask["subtask_status"] for subtask in task["subtasks"]]
        if all(status == "not_started" for status in subtask_statuses):
            task["task_status"] = "not_started"
        elif all(status == "completed" for status in subtask_statuses):
            task["task_status"] = "completed"
        else:
            task["task_status"] = "in_progress"
    return structured_plan


def filter_messages_by_agent(
    messages: List[Message],
    agent_or_tuple: Union[Agent, Union[Tuple[Agent], List[Agent]]],
) -> List[Message]:
    if not isinstance(agent_or_tuple, (tuple, list)):
        agent_or_tuple = [agent_or_tuple]
    else:
        agent_or_tuple = list(agent_or_tuple)
    return [m for m in messages if m.agent in agent_or_tuple]


def split_message_list_by_last_new_reasoning_cycle_marker(
    messages: List[Message],
) -> Tuple[List[Message], List[Message]]:
    # all_messages = self.get_message_history()
    new_cycle_messages = [m for m in messages if m.new_reasoning_cycle is True]
    if not new_cycle_messages:
        return messages, []
    separating_message = new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
    separating_message_idx = next(i for i, m in enumerate(messages) if m.key == separating_message.key)
    return (
        messages[:separating_message_idx],  # Historic messages up to last worker new cycle system message.
        messages[separating_message_idx:],  # All messages from last worker new cycle system message.
    )


def get_last_message_like(messages: List[Message], like: Callable[[Message], bool]) -> Optional[Message]:
    for message in reversed(messages):
        if like(message):
            return message
    return None


# Get all messages that have .current_structured_plan:
def _has_structured_plan(m: Message) -> bool:
    return (
        m.engine_state is not None
        and m.engine_state.agent_state is not None
        and "coordinator" in m.engine_state.agent_state
        and d2m(m.engine_state.agent_state["coordinator"], CoordinatorState).current_structured_plan is not None
    )


# Get last message that have worker delegated_content.
def _has_delegated_content(m: Message) -> bool:
    return (
        m.engine_state is not None
        and m.engine_state.agent_state is not None
        and "worker" in m.engine_state.agent_state
        and d2m(m.engine_state.agent_state["worker"], WorkerState).delegated_content is not None
    )


# endregion


class AgentStore(pydantic.BaseModel):
    coordinator: EngineAgent
    worker: EngineAgent

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


CoordinatorReasoningStage = Literal["update_subtask_status", "dispatch_next_subtasks", "done"]
StructuredPlan = List[Dict[str, Any]]


# This class is only for convenience of typing.
# We transform it back and forth to a dictionary when storing it in the EngineState.
class CoordinatorState(pydantic.BaseModel):
    coordinator_reasoning_stage: CoordinatorReasoningStage
    current_structured_plan: StructuredPlan
    whole_project_completed: bool = False


# This class is only for convenience of typing.
# We transform it back and forth to a dictionary when storing it in the EngineState.
class WorkerState(pydantic.BaseModel):
    delegated_content: Optional[str] = None

    # NOTE: None = All tools!
    # If DEBUG__USE_FILTER_TOOLS is True, default to an empty list (no tools)
    # If DEBUG__USE_FILTER_TOOLS is False, default to None (all tools)
    delegated_tools: Optional[List[str]] = [] if DEBUG__USE_FILTER_TOOLS else None


class OpenAINextGenEngine(OpenAIEngineBase):
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
            # Will throw errors if the structured plan as defined is invalid:
            sanity_check_structured_plan(STRUCTURED_PLAN)

            # Set initial engine state.
            self.session.engine_state = EngineState(
                streaming=False,
                agent="coordinator",
                agent_switched=False,
                agent_state={
                    "coordinator": m2d(
                        CoordinatorState(
                            coordinator_reasoning_stage="update_subtask_status",
                            current_structured_plan=copy.deepcopy(STRUCTURED_PLAN),
                        )
                    ),
                    "worker": m2d(
                        WorkerState(
                            delegated_content=None,
                        )
                    ),
                },
                ui_controlled=UIControlledState(interaction_stage="reason", input_request=None),
            )
        # CASE: When loaded from DB, restore the engine engine state:
        else:
            messages_with_engine_state = self._get_restartable_messages()
            if messages_with_engine_state:
                if messages_with_engine_state[-1].engine_state is None:
                    raise ValueError("EngineState was None.")
                self.session.engine_state = messages_with_engine_state[-1].engine_state

    def _set_initial_messages(self, agent: EngineAgent) -> List[Message]:
        if agent.agent_type == "worker":
            msg_w_dc = get_last_message_like(self.get_message_history(), _has_delegated_content)
            delegated_content = d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerState).delegated_content  # type: ignore
        else:
            delegated_content = None

        system_message_text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(delegated_content),  # Only applicable for worker.
                STRUCTURED_PLAN_REPLACE_MARKER: rich.pretty.pretty_repr(self.get_current_plan()),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
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
                visibility="llm_only_ephemeral",
                engine_state=self.session.engine_state,
            ),
        ]
        tree_helpers.append_multiple_messages_to_end_of_tree(self.session.messages, initial_messages)

        self.db.update_session(self.session)

        return initial_messages

    def _gather_messages_coordinator(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        engine_log("_gather_messages_coordinator")

        coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorState)
        messages_to_process = []
        tools = None

        historic_messages, last_coordinator_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            self.get_message_history()
        )
        if not last_coordinator_messages:
            raise ValueError("No coordinator messages found for coordinator.")
        if not last_coordinator_messages[0].agent == "coordinator" and last_coordinator_messages[0].role == "system":
            raise ValueError("The last message set did not begin with a coordinator system message.")

        historic_messages, last_worker_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            historic_messages
        )
        if last_worker_messages:
            if not last_worker_messages[0].agent == "worker" and last_worker_messages[0].role == "system":
                raise ValueError(
                    "The last message set in the historic messages component did not begin with "
                    "a worker system message."
                )

        historic_messages = self.exclude_system_messages(historic_messages)
        # ^ This will also remove the old system message.
        last_worker_messages = self.exclude_system_messages(last_worker_messages)

        coordinator_system_message = last_coordinator_messages[0]
        coordinator_system_message.text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                STRUCTURED_PLAN_REPLACE_MARKER: rich.pretty.pretty_repr(self.get_current_plan()),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
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
        last_coordinator_messages = last_coordinator_messages[1:]

        # [system]
        messages_to_process.append(coordinator_system_message)

        if historic_messages:
            # <separator>
            messages_to_process.append(
                # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["coordinator"]["historic_messages_start"],
                    agent="coordinator",
                )
            )
            # [historic record] - w/o system messages.
            messages_to_process.extend(historic_messages)

        if last_worker_messages:
            # <separator>
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            messages_to_process.append(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["coordinator"]["historic_messages_last"],
                    agent="coordinator",
                )
            )
            messages_to_process.extend(last_worker_messages)

        # [current conversation]
        # <separator>
        # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
        messages_to_process.append(
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["coordinator"]["coordinator_continuation"],
                agent="coordinator",
            )
        )
        messages_to_process.extend(last_coordinator_messages)

        # [reminder]
        # TODO: Move this to dispatch and reorganize.
        # Check for missed "not_started" subtasks.
        missed_not_started = gather_missed_not_started_subtasks(self.get_current_plan())
        if missed_not_started:
            REMINDER_EXTRA = f"""
**Reminder!**
The following subtasks are marked as 'not_started', even though the tasks after them have been worked on.
Have you forgotten to mark these as "skipped"?
But ALSO remember, you must update all the "skipped" AND "completed" subtasks, in ONE MESSAGE!
Double-check and do not miss out any "skipped" OR "completed" tasks.
{missed_not_started}
"""
            # engine_log("missed_not_started")
            # rich.pretty.pprint(missed_not_started)
            # engine_log("REMINDER_EXTRA")
            # engine_log(REMINDER_EXTRA)
        else:
            REMINDER_EXTRA = ""

        if coordinator_state.coordinator_reasoning_stage == "update_subtask_status":
            REMINDER_EXTRA += """
### Current reasoning stage:
Step 1. Reason about what has been completed so far and update subtask statuses.
"""
        elif coordinator_state.coordinator_reasoning_stage == "dispatch_next_subtasks":
            REMINDER_EXTRA += """
### Current reasoning stage:
Step 3. Issue the next set of subtasks to the WORKER agent.
"""

        # Reminder message - to avoid going beyond the subtasks.
        reminder_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            text=MESSAGE_OPTIONS["coordinator"]["reminder"] + REMINDER_EXTRA,
            agent="coordinator",
            visibility="llm_only_ephemeral",
            engine_state=self.session.engine_state,
        )
        self._append_message(reminder_message)  # Add to history.
        messages_to_process.append(reminder_message)  # And add to list of messages to send to LLM straight away too.
        # Reminder message [END].

        if DEBUG__PRINT_MESSAGES_PER_AGENT:
            engine_log("--- COORDINATOR messages_to_process ")
            rich.pretty.pprint(messages_to_process)
            engine_log("--- COORDINATOR messages_to_process [END]")

        return messages_to_process, tools

    def _gather_messages_worker(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        engine_log("_gather_messages_worker")
        messages_to_process = []

        msg_w_dc = get_last_message_like(self.get_message_history(), _has_delegated_content)
        if msg_w_dc is None:
            raise ValueError("Delegated content was None.")

        if not DEBUG__USE_FILTER_TOOLS:
            tools = list_all_tool_specs()
        else:
            tool_names = d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerState).delegated_tools  # type: ignore
            tools = list_all_tool_specs(filter_tool_names=tool_names)

        worker_messages = filter_messages_by_agent(self.get_message_history(), agent.agent_type)
        historic_worker_messages, last_worker_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            worker_messages
        )
        if not last_worker_messages:
            raise ValueError("No worker messages found for worker.")

        worker_agent_system_message = last_worker_messages[0]
        worker_agent_system_message.text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(
                    d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerState).delegated_content  # type: ignore
                ),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
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

        historic_worker_messages = self.exclude_system_messages(historic_worker_messages)
        last_worker_messages = self.exclude_system_messages(last_worker_messages)
        # ^ This will also remove the old system message.

        # [system]
        messages_to_process.append(worker_agent_system_message)

        if historic_worker_messages:
            # <separator>
            messages_to_process.append(
                # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["worker"]["record_preamble"],
                    agent="worker",
                )
            )
            # [historic record] - w/o system messages.
            messages_to_process.extend(historic_worker_messages)

        # <separator>
        messages_to_process.append(
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["worker"]["record_continuation"],
                agent="worker",
            )
        )
        # [current conversation]
        messages_to_process.extend(last_worker_messages)

        # Reminder message - to avoid going beyond the subtasks.
        reminder_message_text = update_templates(
            body_text=MESSAGE_OPTIONS["worker"]["reminder"],
            templates={
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(
                    d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerState).delegated_content  # type: ignore
                ),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
            },
        )
        reminder_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            text=reminder_message_text,
            agent="worker",
            visibility="llm_only_ephemeral",
            engine_state=self.session.engine_state,
        )
        self._append_message(reminder_message)  # Add to history.
        messages_to_process.append(reminder_message)  # And add to list of messages to send to LLM straight away too.
        # Reminder message [END].

        if DEBUG__PRINT_MESSAGES_PER_AGENT:
            engine_log("--- WORKER messages_to_process ")
            rich.pretty.pprint(messages_to_process)
            engine_log("--- WORKER messages_to_process [END]")

        return messages_to_process, tools

    def get_current_plan(self) -> List[Dict[str, Any]]:
        msg_w_plan = get_last_message_like(self.get_message_history(), _has_structured_plan)
        if not msg_w_plan:
            return copy.deepcopy(STRUCTURED_PLAN)
        else:
            return d2m(msg_w_plan.engine_state.agent_state["coordinator"], CoordinatorState).current_structured_plan  # type: ignore

    def _dispatch_coordinator(self, agent: EngineAgent) -> EngineState:
        last_message = self.get_last_message()

        if last_message.text is None:
            raise ValueError("Last message text must not be None.")
        if last_message.agent != "coordinator":
            raise ValueError("Last message agent must be coordinator.")
        if last_message.engine_state is None:
            raise ValueError("Last message EngineState must not be None.")
        last_message_coordinator_state = d2m(last_message.engine_state.agent_state["coordinator"], CoordinatorState)

        if last_message_coordinator_state.coordinator_reasoning_stage is None:
            raise ValueError("Last message EngineState must have reasoning stage.")

        if PROJECT_END_MARKER in last_message.text:
            last_message_coordinator_state.whole_project_completed = True

            coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorState)
            coordinator_state.coordinator_reasoning_stage = "done"
            coordinator_state.whole_project_completed = True
            self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)

            self.update_state()

            return self.session.engine_state

        if last_message_coordinator_state.coordinator_reasoning_stage == "update_subtask_status":
            # engine_log("IN: update_subtask_status")
            # engine_log("last_message.text")
            # engine_log(last_message.text)
            try:
                # Content between two triple backticks.
                raw_content = extract_content_between_two_triple_backticks(last_message.text)
                # engine_log("raw_content")
                # engine_log(raw_content)

                # Try to parse it to Python list of dictionaries.
                subtask_status_updates = parse_python_list_of_dicts(raw_content)
                # engine_log("subtask_status_updates")
                # engine_log(subtask_status_updates)

                # Validate the format of the status updates.
                validated_status_updates = validate_status_updates(subtask_status_updates)
                # engine_log("validated_status_updates")
                # engine_log(validated_status_updates)

                # Update the structured plan with the new statuses.
                updated_structured_plan = update_statuses_in_structured_plan(
                    structured_plan=self.get_current_plan(), status_updates=validated_status_updates
                )

                # Update task statuses in the structured plan.
                updated_structured_plan = update_task_statuses_in_structured_plan(
                    structured_plan=updated_structured_plan
                )

                # TODO: This.
                # Check for missed "not_started" subtasks.
                # missed_not_started = gather_missed_not_started_subtasks(self.get_current_plan())
                # if missed_not_started:
                #     missed_not_started_message = dedent(f"""
                #     {PROBLEM_WITH_SUBTASK_SELECTION_MARKER}:
                #     The following subtasks are marked as 'not_started', even though the tasks after them have been worked on.
                #     Have you forgotten to mark these as "skipped"?
                #     {missed_not_started}
                #     """)
                #     # engine_log("missed_not_started")
                #     # rich.pretty.pprint(missed_not_started)
                #     # engine_log("REMINDER_EXTRA")
                #     # engine_log(REMINDER_EXTRA)
                #     self._append_message(
                #         message=Message(
                #             key=KeyGeneration.generate_message_key(),
                #             role="system",
                #             visibility="llm_only",
                #             agent="coordinator",
                #             text=f"{PROBLEM_WITH_SUBTASK_SELECTION_MARKER}:\n{missed_not_started_message}",
                #         )
                #     )
                # else:

                # Update EngineState state.
                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorState)
                coordinator_state.coordinator_reasoning_stage = "dispatch_next_subtasks"
                coordinator_state.current_structured_plan = updated_structured_plan
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
                self.update_state()

            except ValueError as e:
                exc_str = str(e)
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_SUBTASK_STATUS_UPDATES_MARKER}:\n{exc_str}",
                    )
                )

            # import time
            # time.sleep(1)

        elif last_message_coordinator_state.coordinator_reasoning_stage == "dispatch_next_subtasks":
            # engine_log("IN: dispatch_next_subtasks")
            # engine_log("last_message.text")
            # engine_log(last_message.text)
            try:
                # Content between two triple backticks.
                raw_content = extract_content_between_two_triple_backticks(last_message.text)
                # engine_log("raw_content")
                # engine_log(raw_content)

                # Try to parse it to Python list of strings
                subtask_selection = parse_python_list_of_strings(raw_content)
                # engine_log("subtask_selection")
                # engine_log(subtask_selection)

                # Validate the format of the subtask selection.
                validated_subtask_selection = validate_subtask_selection(subtask_selection)
                # engine_log("validated_subtask_selection")
                # engine_log(validated_subtask_selection)

                # Update statuses in the structured plan.
                updated_structured_plan = update_statuses_in_structured_plan(
                    structured_plan=self.get_current_plan(),
                    status_updates=[
                        {"subtask_id": subtask_id, "subtask_status": "in_progress"}
                        for subtask_id in validated_subtask_selection
                    ],
                )

                # Update task statuses in the structured plan.
                updated_structured_plan = update_task_statuses_in_structured_plan(
                    structured_plan=updated_structured_plan,
                )

                # Create the actual task description for the worker.
                try:
                    worker_actual_task = create_worker_actual_task(
                        structured_plan=self.get_current_plan(), subtask_selection=validated_subtask_selection
                    )
                    # Any replacements:
                    # NOTE: If the structured plan features any "replace markers". these must be replaced here!
                    worker_actual_task = update_templates(
                        body_text=worker_actual_task,
                        templates={
                            MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                        },
                    )
                except Exception as e:
                    # engine_log("EXCEPTION IN create_worker_actual_task")
                    # engine_log(e)
                    raise RuntimeError(f"Could not create the worker actual task description:\n{str(e)}") from e
                # engine_log("worker_actual_task")
                # engine_log(worker_actual_task)

                # Update EngineState state.
                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorState)
                coordinator_state.coordinator_reasoning_stage = "done"
                coordinator_state.current_structured_plan = updated_structured_plan
                # PS: current_structured_plan is unchanged at this stage.
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
                self.session.engine_state.agent = "worker"
                self.session.engine_state.agent_switched = True

                # Get the list of relevant tools.
                if DEBUG__USE_FILTER_TOOLS:
                    # Go through each subtask in validated_subtask_selection and get the tools.
                    relevant_tools = []
                    for subtask_id in subtask_selection:
                        task = get_task_from_subtask_id(updated_structured_plan, subtask_id)
                        subtasks_list = task["subtasks"]
                        subtask_dict = [subtask for subtask in subtasks_list if subtask["subtask_id"] == subtask_id][0]
                        if subtask_dict["tools"] is None:
                            relevant_tools = None
                            break
                        relevant_tools.extend(subtask_dict["tools"])
                    delegated_tools = list(set(relevant_tools)) if relevant_tools is not None else None

                # Prepare everything for the worker.
                worker_state = d2m(self.get_state().agent_state["worker"], WorkerState)
                worker_state.delegated_content = worker_actual_task
                if DEBUG__USE_FILTER_TOOLS:
                    worker_state.delegated_tools = delegated_tools
                self.session.engine_state.agent_state["worker"] = m2d(worker_state)
                self._set_initial_messages(agent=self.agents_.worker)

                # Update state.
                self.update_state()

                # engine_log("EngineState at the end of `dispatch_next_subtasks`.")
                # engine_log("*" * 100)
                # engine_log(self.session.engine_state)
                # engine_log("*" * 100)

            except ValueError as e:
                exc_str = str(e)
                # raise
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_SUBTASK_SELECTION_MARKER}:\n{exc_str}",
                    )
                )

            # import time
            # time.sleep(1)

        else:
            raise ValueError(
                f"Invalid coordinator reasoning stage: {last_message_coordinator_state.coordinator_reasoning_stage}"
            )

        return self.session.engine_state

    def _dispatch_worker(self, agent: EngineAgent) -> EngineState:
        last_message = self.get_last_message()

        if TASK_COMPLETED_INDICATOR in str(last_message.text):
            self.session.engine_state.agent_switched = True
            self.session.engine_state.agent = "coordinator"

            # Update coordinator state:
            coordinator_state = d2m(self.session.engine_state.agent_state["coordinator"], CoordinatorState)
            coordinator_state.coordinator_reasoning_stage = "update_subtask_status"
            self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)

            self._set_initial_messages(agent=self.agents_.coordinator)
            self.update_state()

        else:
            self.session.engine_state.user_message_requested = True

            # Exceptions to asking for user input.
            # Control will be handed over back to the worker in these cases.
            if last_message.text is not None:
                if GENERATED_CODE_FORMAT_ERROR_MSG in last_message.text:
                    self.session.engine_state.user_message_requested = False
                if LONG_MESSAGE_SPLIT_INDICATOR in last_message.text:
                    self.session.engine_state.user_message_requested = False

        return self.session.engine_state

    def define_agents(self) -> Dict[Agent, EngineAgent]:
        # AgentStore just to act as a dotdict for convenient access.
        self.agents_ = AgentStore(
            coordinator=EngineAgent(
                "coordinator",
                first_message_content=MESSAGE_OPTIONS["coordinator"]["first_message_content"],
                system_message_template=MESSAGE_OPTIONS["coordinator"]["system_message_template"],
                first_message_role="assistant",
                set_initial_messages=OpenAINextGenEngine._set_initial_messages,  # type: ignore
                gather_messages=OpenAINextGenEngine._gather_messages_coordinator,  # type: ignore
                dispatch=OpenAINextGenEngine._dispatch_coordinator,  # type: ignore
            ),
            worker=EngineAgent(
                "worker",
                system_message_template=MESSAGE_OPTIONS["worker"]["system_message_template"],
                first_message_content=MESSAGE_OPTIONS["worker"]["first_message_content"],
                first_message_role="assistant",
                set_initial_messages=OpenAINextGenEngine._set_initial_messages,  # type: ignore
                gather_messages=OpenAINextGenEngine._gather_messages_worker,  # type: ignore
                dispatch=OpenAINextGenEngine._dispatch_worker,  # type: ignore
            ),
        )
        as_dict = self.agents_.model_dump()  # {"coordinator": coordinator EngineAgent, ...}
        return as_dict  # type: ignore

    def define_initial_agent(self) -> Agent:
        return "coordinator"

    @staticmethod
    def get_engine_name() -> str:
        return "openai_nextgen"

    def project_completed(self) -> bool:
        return d2m(self.get_state().agent_state["coordinator"], CoordinatorState).whole_project_completed

    def get_worker_messages_historic_last(self) -> Tuple[List[Message], List[Message]]:
        # Return [the historic messages] and [the last reasoning cycle messages] for the worker.
        all_worker_messages = self.get_all_worker_messages()
        worker_new_cycle_messages = [m for m in all_worker_messages if m.new_reasoning_cycle is True]
        if not worker_new_cycle_messages:
            raise ValueError("No new reasoning cycle messages found for worker.")
        separating_message = worker_new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
        separating_message_idx = next(i for i, m in enumerate(all_worker_messages) if m.key == separating_message.key)
        return (
            all_worker_messages[:separating_message_idx],  # Historic worker messages.
            all_worker_messages[separating_message_idx:],  # Last reasoning cycle worker messages.
        )

    def get_all_messages_historic_last(self) -> Tuple[List[Message], List[Message]]:
        all_messages = self.get_message_history()
        worker_new_cycle_messages = [m for m in all_messages if m.agent == "worker" and m.new_reasoning_cycle is True]
        if not worker_new_cycle_messages:
            return all_messages, []
        separating_message = worker_new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
        separating_message_idx = next(i for i, m in enumerate(all_messages) if m.key == separating_message.key)
        return (
            all_messages[:separating_message_idx],  # Historic messages up to last worker new cycle system message.
            all_messages[separating_message_idx:],  # All messages from last worker new cycle system message.
        )

    def exclude_system_messages(self, messages: List[Message]) -> List[Message]:
        return [m for m in messages if m.role != "system"]

    def exclude_non_coordinator_system_messages(self, messages: List[Message]) -> List[Message]:
        return [m for m in messages if (m.role != "system" or m.agent == "coordinator")]

    def get_all_worker_messages(self) -> List[Message]:
        return [m for m in self.get_message_history() if m.agent in ("worker", "supervisor")]

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
        completion_kwargs = dict(
            messages=messages_to_send_to_openai,
            stream=True,
        )
        if tools is not None and len(tools) > 0:
            completion_kwargs["tools"] = tools
        stream = self.initialize_completion()(**completion_kwargs)
        if self.supports_streaming_token_count() is False:
            self._count_tokens(messages_in=messages_to_send_to_openai, tools=tools)

        # Process the stream and yield the results.
        message_chunks: List[str] = []
        is_tool_call = False
        tool_call_content: Dict[str, Any] = {"content": None, "role": "assistant", "tool_calls": {}}

        # Reset the engine state.
        worker_state = d2m(self.session.engine_state.agent_state["worker"], WorkerState)
        coordinator_state = d2m(self.session.engine_state.agent_state["coordinator"], CoordinatorState)
        self.session.engine_state = EngineState(
            streaming=True,
            # Do not reset agent:
            agent=self.session.engine_state.agent,
            agent_switched=False,
            agent_state={
                # We do not need to reset anything in the coordinator state.
                "coordinator": m2d(coordinator_state),
                # We do not need to reset anything in the worker state.
                "worker": m2d(worker_state),
            },
            ui_controlled=UIControlledState(),
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

    def _update_system_message(self, message: Message, template: str) -> Message:
        message = copy.deepcopy(message)
        message.text = template.replace("{WD_CONTENTS}", self.describe_working_directory_str())
        return message


class AzureOpenAINextGenEngine(
    AzureOpenAIEngineMixin,  # Mixing needs to come first to override the methods correctly.
    OpenAINextGenEngine,
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
        OpenAINextGenEngine.__init__(
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
        return "azure_openai_nextgen"
