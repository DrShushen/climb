# Developer's Guide

Please read the [Contributing Guide](contributing.md) to contribute to this project.

## Code structure

The code structure of `climb` is summarized below:
```
├── config_examples      -- Folder housing example configurations for deployment
├── docs                 -- Folder for documentation on readthedocs.io
│   ├── _static
│   ├── api
│   └── assets
│       └── quickstart
├── entry                -- Coder for Streamlit user interface
│   └── st               -- Entrypoint for Streamlit under entry/st/app.py
│       └── pages
├── src                  -- Source code for CliMB
│   └── climb
│       ├── common       -- Common functions for CliMB
│       ├── db           -- Code for database. Currently the database is set up as a custom data class using TinyDB
│       ├── engine       -- Engine to interact with Azure / OpenAI for AutoML
│       ├── tool         -- Folder for tools available to the agent
│       │   └── impl
│       └── ui           -- UI code for Streamlit
└── tests                -- Unit tests. Folder structure follows ./src/climb/
    ├── engine
    └── tool
        └── impl
            └── data
```

## Engine

**Engine** (under `./climb/engine`) is the backbone used to interact with a LLM backend as the reasoning unit. Currently CliMB supports Azure and OpenAI. The `EngineBase` class manages sessions, messages, agents, and stores session data to a local database (`./db`). Direct interaction with the LLMs is performed by agents under the `EngineAgent` class. Communication is abstracted as a `Message` in `./climb/common/data_structures`, which stores the details of each interaction step. Messages are structured as a tree to allow branching conversations, managed using `tree_helpers`.

The engine takes in input dataset and user specifications, and performs automated data data processing and analysis on the data by consulting the reasoning unit on actions, and the end user for feedback and confirmation of actions. The reasoning unit then plans a set of tasks to be executed by the action unit. We provide the action unit a set of tools, and also allow it to generate code not within the tool library to be executed (e.g. print a column specific to the user's dataset). The full list of possible tasks to be assigned by the reasoning unit are described in `./engine/engine_openai_nextgen.py`.

## Tools

**Tools** (under `./climb/tool`) are modules that can be called by the actioning unit to execute pre-defined actions on the dataset. `ToolBase` is a wrapper for a function that actions on the data. Tools are executed in a multi-threaded manner via a `ToolThread`. A tool is used when `._execute` is called by the action unit, at which point a `ToolCommunicator` object is initialized to log and display the output generated during execution of the tool function. This allows for background execution of tools and prevents delay during user interaction.

One direction for further contribution to CliMB is the development of more tools for data analysis. To develop a new tool, we first define a function that takes in a `ToolCommunicator`. Then, we define a superclass of `ToolBase` to generate a class that can be initialized and passed onto the action unit. We want to follow the signature of `.execute()` to define the specific behaviour of our tool. If we want to pass any additional argument to our underlying tool function, they will also need to be described under `.specification()`. The specification is expoed to the action unit and will be filled in by it during the tool call. Below is an example of how to initialize a tool function and class:

```
# NB: data_file_path and target_column will need to be defined in specification
def get_col_max(tc: ToolCommunicator, data_file_path: str, target_column: str):
    """Example tool function to get the max value of a column within a dataframe"""

    # Instead of printing to console, we print via the ToolCommunicator
    tc.print(f"Getting max value in column {target_column} from {data_file_path}")

    df = pd.read_csv(data_file_path)

    output_str = "Column max: " + str(df[target_column].max())

    tc.set_returns(output_str)

class ColumnMax(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:

        # NOTE: In general tools work on a separate directory
        data_file_path = os.path.join(self.working_directory, kwargs["data_file_path"])

        thrd, out_stream = execute_tool(
            get_col_max, # Our tool function defined above
            data_file_path=data_file_path,
            target_column=kwargs["target_column"],
        )

        self.tool_thread = thrd
        return out_stream

    # Other properties we want to define for reasoning / action units.
    @property
    def name(self) -> str:
        return "column_max"

    @property
    def description(self) -> str:
        return "This returns the value of a column within a dataframe"

    # Specification defines the parameters for the underlying tool function
    @property
    def specification(self) -> Dict[str, Any]:
        return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_file_path": {"type": "string", "description": "Path to the data file."},
                            "target_column": {"type": "string", "description": "Target column"},
                        },
                        "required": ["data_file_path", "target_column"],
                    },
                },
            }
```

The list of available tools is documented under [Tool Reference](tool.md).

## UI
CliMB utilizes [Streamlit](https://streamlit.io/) as the framework for user interface. Streamlit serves a locally hosted web app with pages under `./entry/st/pages`. Currently, pages include a chat interface under `main.py` and session manager under `research_management.py`. Other helper functions for the UI are found under `./ui/`.

An entrypoint to understand the user experience and data flow would be `main_flow()` under `main.py`. In general, the state of the UI is defined in the underlying engine, where the app could be either processing user input, intermediate reasoning stream from the reasoning unit, or display output. Each interaction within the chat interface is logged as a Message and stored within the Engine. This allows for sessions to be saved and retrieved at a later time.
