import shutil
from typing import Any, List

import pandas as pd
import streamlit as st

from climb.common import create_new_session
from climb.common.plan_files import load_plan_file
from climb.db.tinydb_db import TinyDB_DB
from climb.engine import AZURE_OPENAI_CONFIG_PATH, ENGINE_MAP, EngineBase, load_azure_openai_configs
from climb.ui.st_common import (
    CLIMB_ICON_IMAGE,
    PAGE_TITLES,
    SHOW_ROLES,
    SHOW_VISIBILITIES,
    initialize_common_st_state,
    menu,
)

st.set_page_config(layout="wide", page_title=PAGE_TITLES["research_management_tab_title"], page_icon=CLIMB_ICON_IMAGE)
menu()


db = TinyDB_DB()
sessions = db.get_all_sessions()

initialize_common_st_state(db)

st.markdown(f"### {PAGE_TITLES['research_management_emoji']}")

EDITABLE_COLUMNS = ["Session name", "Select"]
sessions_df = pd.DataFrame(
    [
        {
            "Session key": session.session_key,
            "Session name": session.friendly_name,
            "Started at": session.started_at,
            "Engine name": session.engine_name,
            "# messages": len(
                [m for m in session.messages if m.role in SHOW_ROLES and m.visibility in SHOW_VISIBILITIES]
            ),
            "Active": session.session_key == st.session_state.active_session_key,
        }
        for session in sessions
    ]
)


def delete_sessions(session_keys: List[str]) -> None:
    for session_key in session_keys:  # pylint: disable=redefined-outer-name
        session = db.get_session(session_key)  # pylint: disable=redefined-outer-name
        try:
            shutil.rmtree(session.working_directory)
        except FileNotFoundError:
            pass
        db.delete_session(session_key)

    if st.session_state.active_session_key in session_keys:
        st.session_state.active_session_key = (
            db.get_all_sessions()[0].session_key if len(db.get_all_sessions()) > 0 else None
        )
        st.session_state.session_reload = True

        # Update active session.
        user_settings = db.get_user_settings()  # pylint: disable=redefined-outer-name
        user_settings.active_session = st.session_state.active_session_key
        db.update_user_settings(user_settings)

        if "engine" in st.session_state:
            del st.session_state["engine"]

        print("Active session was deleted, resetting to `None`.")

    st.success("Sessions deleted.")


def start_new_session() -> None:
    # pylint: disable-next=redefined-outer-name
    session = create_new_session(
        session_name=st.session_state.new_session_settings["session_name"],
        engine_name=st.session_state.new_session_settings["engine_name"],
        engine_params=st.session_state.new_session_settings["engine_params"],
        db=db,
    )

    st.session_state.active_session_key = session.session_key
    st.session_state.session_reload = True

    if "engine" in st.session_state:
        del st.session_state["engine"]


# Using a row selection workaround:
# https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
sessions_df.insert(0, "Select", False)

if not sessions:
    st.write("No research history available.")
else:
    de = st.data_editor(
        sessions_df,
        disabled=[c for c in sessions_df.columns if c not in EDITABLE_COLUMNS],
        column_config={
            "Select": st.column_config.CheckboxColumn(required=True, width=5),
            "Session key": None,
            "Session name": st.column_config.TextColumn(width=450),
            "Started at": st.column_config.DatetimeColumn(width=150),
            "Engine name": st.column_config.TextColumn(width=150),
            "# messages": st.column_config.NumberColumn(width=25),
            "Active": st.column_config.CheckboxColumn(width=5),
        },
        hide_index=True,
    )

    selected_rows = de[de.Select]

    active_session_name = None
    ready_to_load = len(selected_rows) == 1
    if ready_to_load:
        selected_session_key = selected_rows["Session key"].tolist()[0]
        active_session_name = sessions_df.loc[
            sessions_df["Session key"] == selected_session_key, "Session name"
        ].values[0]  # type: ignore
    activate_button_name = (
        "🚀 Load selected session ⏵"
        if active_session_name is None
        else f"🚀 Load selected session `{active_session_name}` ⏵"
    )

    col_btn1, col_btn2, col_btn3 = st.columns([0.15, 0.72, 0.18])
    with col_btn1:
        btn_update_session_details = st.button(
            "💾 Update session details", help="Update the details of selected sessions if you edited them above."
        )
    with col_btn2:
        btn_load_selected_session = st.button(
            activate_button_name, disabled=not ready_to_load, help="Select one session to activate."
        )
    with col_btn3:
        btn_delete_selected_sessions = st.button(
            "🗑️ Delete selected session(s) ⏵",
            disabled=selected_rows.empty,
            help="Select at least one session to delete.",
        )

    if btn_update_session_details:
        for _, row in de.iterrows():
            session_key = row["Session key"]
            new_name = row["Session name"]
            session = db.get_session(session_key)
            if session.friendly_name != new_name:
                session.friendly_name = new_name
                db.update_session(session)
                if session_key == st.session_state.active_session_key:
                    st.session_state.session_reload = True
                    st.rerun()

    if btn_load_selected_session:
        st.session_state.active_session_key = selected_session_key
        st.session_state.session_reload = True

        # Update active session.
        user_settings = db.get_user_settings()
        user_settings.active_session = selected_session_key
        db.update_user_settings(user_settings)

        if "engine" in st.session_state:
            del st.session_state["engine"]

        st.switch_page("pages/main.py")

    if btn_delete_selected_sessions:
        with st.expander("Confirm deletion", expanded=True):
            st.markdown("")
            selected_session_keys = selected_rows["Session key"].tolist()
            st.markdown("The following sessions will be **deleted**:")
            st.dataframe(
                selected_rows,
                column_config={"Select": None, "Session key": None},
                hide_index=True,
            )
            st.markdown("> ⚠️ Are you sure you want to delete these sessions? **This action cannot be undone.**")
            st.button(
                "🗑️ Yes, permanently delete " + ("these sessions" if len(selected_rows) > 1 else "this session"),
                type="primary",
                on_click=delete_sessions,
                args=(selected_session_keys,),
            )
            st.markdown("")

st.write("")
st.write("")
st.markdown("#### 🕹️ Start a new session")
st.write("")
st.info(
    """
    **CliMB** currently supports the following classes of OpenAI models:
    - `gpt-4o`, `gpt-4-turbo`: **recommended** as they good reasoning capabilities.
    - `gpt-4o-mini`, `gpt-3.5-turbo`: **not** recommended as they are less capable and are more likely to \
lead to substandard results.
    """
)

with st.container(border=True):
    st.markdown("#### New session:")
    col_new_session_name, col_engine_name = st.columns(2)
    with col_new_session_name:
        new_session_name = st.text_input("Session name", value="", placeholder="Leave empty for auto-generated name")
        st.session_state.new_session_settings["session_name"] = new_session_name if new_session_name != "" else None
    with col_engine_name:
        engine_name = st.selectbox("Select engine", options=ENGINE_MAP.keys())
        st.session_state.new_session_settings["engine_name"] = engine_name

with st.container(border=True):
    st.markdown("""
    #### Session engine parameters:
    Hover over :grey[:material/help:] to see more information about the parameter.
    """)
    engine_params = dict()
    EngineClass: EngineBase = ENGINE_MAP[engine_name]  # type: ignore
    cannot_create = False
    if "azure" in engine_name:
        if load_azure_openai_configs(AZURE_OPENAI_CONFIG_PATH) == []:
            st.markdown(
                "No Azure OpenAI configurations found. Please add a configuration file at `az_openai_config.yml`."
            )
            cannot_create = True
    # kwargs dict for params that are set by static methods. (i.e. params that have a `set_by_static_method` attribute.)
    kwargs_dict = {
        # ... Add any initial kwargs needed ...
    }
    # print(kwargs_dict)
    for param in EngineClass.get_engine_parameters():
        # Values set by static methods. --- --- ---
        # NOTE: we also keep dynamically adding the engine_params defined thus far to potentially be used by
        # a param-setting static method. Hence the order of the parameters matters, if a parameter is set by a
        # static method, it should be defined after the parameters that the static method depends on.
        value_set_dynamically = None
        disabled_set_dynamically = False
        enum_value_set_dynamically = None
        if hasattr(param, "value_set_by_static_method") and param.value_set_by_static_method is not None:
            static_method = getattr(EngineClass, param.value_set_by_static_method)
            kwargs_dict.update(engine_params)
            value_set_dynamically = static_method(**kwargs_dict)
            value_may_be_editable = False
        if hasattr(param, "disabled_set_by_static_method") and param.disabled_set_by_static_method is not None:
            static_method = getattr(EngineClass, param.disabled_set_by_static_method)
            kwargs_dict.update(engine_params)
            disabled_set_dynamically = static_method(**kwargs_dict)
            print(f">>> disabled_set_dynamically ({param.name}): {disabled_set_dynamically}")
        if hasattr(param, "enum_values_set_by_engine_config") and param.enum_values_set_by_engine_config is not None:
            static_method = getattr(EngineClass, param.enum_values_set_by_engine_config)
            kwargs_dict.update(engine_params)
            enum_value_set_dynamically = static_method(**kwargs_dict)
        param_disabled = param.disabled if disabled_set_dynamically is False else True
        # --- --- ---
        if param.kind == "float":
            engine_params[param.name] = st.number_input(  # type: ignore
                param.name,
                help=param.description,
                value=value_set_dynamically if value_set_dynamically is not None else param.default,  # type: ignore
                min_value=param.min_value,
                max_value=param.max_value,
                disabled=param_disabled,
            )
        elif param.kind == "bool":
            engine_params[param.name] = st.checkbox(
                param.name,
                help=param.description,
                value=value_set_dynamically if value_set_dynamically is not None else param.default,  # type: ignore
                disabled=param_disabled,
            )
        elif param.kind == "enum":
            engine_params[param.name] = st.selectbox(
                param.name,
                help=param.description,
                options=param.enum_values if enum_value_set_dynamically is None else enum_value_set_dynamically,  # type: ignore
                index=(
                    param.enum_values.index(
                        value_set_dynamically if value_set_dynamically is not None else param.default
                    )
                    if not enum_value_set_dynamically
                    else enum_value_set_dynamically.index(enum_value_set_dynamically[0])
                ),
                disabled=param_disabled,
            )
        elif param.kind == "records":
            st.markdown(param.name)
            st.caption(f"{param.description}")

            disabled = param.records_disabled_keys or []

            df = pd.DataFrame(value_set_dynamically if value_set_dynamically is not None else param.default)

            # Estimate a reasonable width for all the columns in the dataframe.
            # Convert each column to string, and return the maximum character length of each column.
            max_column_lengths = df.map(lambda x: len(str(x)) if not isinstance(x, bool) else 1).max()
            # Normalize so values add to 1000 and convert to int.
            max_column_lengths = (max_column_lengths / max_column_lengths.sum() * 1000).astype(int)
            max_column_lengths = {col: int(str(max_column_lengths[col])) for col in df.columns}  # int64 --> python int.

            # Automatically get the column config type for column types bool, numeric, and string.
            def _get_col_type(col: str) -> Any:
                if df[col].dtype.name == "bool":
                    return st.column_config.CheckboxColumn
                elif "float" in df[col].dtype.name or "int" in df[col].dtype.name:
                    return st.column_config.NumberColumn
                elif "datetime" in df[col].dtype.name:
                    return st.column_config.DatetimeColumn
                elif "object" in df[col].dtype.name:
                    return st.column_config.TextColumn
                else:
                    raise ValueError(f"Unexpected column type: {df[col].dtype.name}")

            # Create the data editor.
            df_edited = st.data_editor(
                df,
                disabled=disabled,
                hide_index=True,
                column_config={col: _get_col_type(col)(width=max_column_lengths[col]) for col in df.columns},
            )
            engine_params[param.name] = df_edited.to_dict("records")

        else:
            raise ValueError(f"Unexpected parameter kind: {param.kind}")
    st.session_state.new_session_settings["engine_params"] = engine_params

# Any extra validation:
if "plan_file" in engine_params:
    try:
        plan_file_data = load_plan_file(engine_params["plan_file"], relative_path=True)
    except ValueError as e:
        st.error(f"Failed to load plan file: {e}")
        cannot_create = True
    if len(plan_file_data) == 0:
        st.error("No episodes available in the plan file. Please select a different plan file.")
        cannot_create = True

if "possible_episodes" in engine_params:
    if len([ep for ep in engine_params["possible_episodes"] if ep["enabled"]]) == 0:
        st.error("No episodes enabled in the possible episodes parameter. Please select at least one enabled episode.")
        cannot_create = True

if "model_id" in engine_params and "temperature" in engine_params:
    if "gpt-5" in engine_params["model_id"] and engine_params["temperature"] != 1.0:
        st.error(
            "GPT-5 class models only support a temperature of `1.0`. Please set the temperature to `1.0` in the engine parameters."
        )
        cannot_create = True

if st.button(
    "Start new session ⏵",
    on_click=start_new_session,
    type="primary",
    help="This will start a new research session with the selected engine and parameters.",
    disabled=cannot_create,
):
    st.switch_page("pages/main.py")
