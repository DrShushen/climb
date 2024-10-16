import shutil
from typing import List, cast

import pandas as pd
import streamlit as st

from climb.common import create_new_session
from climb.db.tinydb_db import TinyDB_DB
from climb.engine import AZURE_OPENAI_CONFIG_PATH, ENGINE_MAP, load_azure_openai_configs
from climb.ui.st_common import PAGE_TITLES, SHOW_ROLES, SHOW_VISIBILITIES, initialize_common_st_state, menu

st.set_page_config(page_title=PAGE_TITLES["research_management_plain"])
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
                [m for m in session.messages if m.data.role in SHOW_ROLES and m.data.visibility in SHOW_VISIBILITIES]
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
            "Select": st.column_config.CheckboxColumn(required=True),
            "Session key": None,
        },
        hide_index=True,
    )

    selected_rows = de[de.Select]

    if st.button("💾 Update session details", help="Update the details of selected sessions if you edited them above."):
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
    if st.button(
        activate_button_name,
        disabled=not ready_to_load,
        help="Select one session to activate.",
    ):
        st.session_state.active_session_key = selected_session_key
        st.session_state.session_reload = True

        # Update active session.
        user_settings = db.get_user_settings()
        user_settings.active_session = selected_session_key
        db.update_user_settings(user_settings)

        if "engine" in st.session_state:
            del st.session_state["engine"]

        st.switch_page("pages/main.py")

    if st.button(
        "🗑️ Delete selected session(s) ⏵", disabled=selected_rows.empty, help="Select at least one session to delete."
    ):
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
st.markdown("""
> ⛔ Advanced section for early testing only, this will be simplified or removed in the future.
""")
st.write("")

col1b, col2b = st.columns(2)
with col1b:
    st.markdown("New session settings:")
    new_session_name = st.text_input("Session name", value="", placeholder="Leave empty for auto-generated name")
    engine_name = st.selectbox("Select engine", options=ENGINE_MAP.keys())
    st.session_state.new_session_settings["session_name"] = new_session_name if new_session_name != "" else None
    st.session_state.new_session_settings["engine_name"] = engine_name

with col2b:
    st.markdown("Engine parameters:")
    engine_params = dict()
    EngineClass = ENGINE_MAP[engine_name]  # type: ignore
    cannot_create = False
    for param in ENGINE_MAP[cast(str, engine_name)].get_engine_parameters():
        # Values set by static methods. --- --- ---
        if engine_name == "azure_openai":
            if load_azure_openai_configs(AZURE_OPENAI_CONFIG_PATH) == []:
                st.markdown("No Azure OpenAI configurations found.")
                cannot_create = True
                break
        if engine_name == "azure_openai" and param.name == "model_id":
            static_method = getattr(EngineClass, param.set_by_static_method)
            config_item_name = engine_params["config_item_name"]
            value_set = static_method(config_item_name=config_item_name)
        else:
            value_set = None
        # --- --- ---
        if param.kind == "float":
            engine_params[param.name] = st.number_input(  # type: ignore
                param.name,
                help=param.description,
                value=value_set if value_set is not None else param.default,  # type: ignore
                min_value=param.min_value,
                max_value=param.max_value,
                disabled=param.disabled if value_set is None else True,
            )
        elif param.kind == "bool":
            engine_params[param.name] = st.checkbox(
                param.name,
                help=param.description,
                value=value_set if value_set is not None else param.default,  # type: ignore
                disabled=param.disabled if value_set is None else True,
            )
        elif param.kind == "enum":
            engine_params[param.name] = st.selectbox(
                param.name,
                help=param.description,
                options=param.enum_values,  # type: ignore
                index=param.enum_values.index(value_set if value_set is not None else param.default),  # type: ignore
                disabled=param.disabled if value_set is None else True,
            )
        else:
            raise ValueError(f"Unexpected parameter kind: {param.kind}")
    st.session_state.new_session_settings["engine_params"] = engine_params
if st.button(
    "Start new session ⏵",
    on_click=start_new_session,
    type="primary",
    help="This will start a new research session with the selected engine and parameters.",
    disabled=cannot_create,
):
    st.switch_page("pages/main.py")
