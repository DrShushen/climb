import streamlit as st

from climb.db.tinydb_db import TinyDB_DB
from climb.ui.st_common import COMMON_SETTINGS, PAGE_TITLES, menu

st.set_page_config(page_title=PAGE_TITLES["settings_plain"])
menu()

db = TinyDB_DB()

st.markdown(f"## {PAGE_TITLES['settings_emoji']}")

st.markdown("#### Your Details")

# Load the user settings from the database.
user_settings = db.get_user_settings()

# Display the user settings.
user_settings.user_name = st.text_input("User Name", user_settings.user_name)

# For debugging:
# user_settings.disclaimer_shown = st.checkbox("Disclaimer shown", value=user_settings.disclaimer_shown, disabled=False)

st.markdown(
    """
    #### Default conversation view settings
    These settings will be applied by default to your new research sessions.
    """
)
user_settings.default_session_settings.show_tool_call_logs = st.checkbox(
    COMMON_SETTINGS["display"]["show_tool_call_logs"]["name"],
    user_settings.default_session_settings.show_tool_call_logs,
    help=COMMON_SETTINGS["display"]["show_tool_call_logs"]["help"],
)
user_settings.default_session_settings.show_tool_call_return = st.checkbox(
    COMMON_SETTINGS["display"]["show_tool_call_return"]["name"],
    user_settings.default_session_settings.show_tool_call_return,
    help=COMMON_SETTINGS["display"]["show_tool_call_return"]["help"],
)
user_settings.default_session_settings.show_code = st.checkbox(
    COMMON_SETTINGS["display"]["show_code"]["name"],
    user_settings.default_session_settings.show_code,
    help=COMMON_SETTINGS["display"]["show_code"]["help"],
)
user_settings.default_session_settings.show_code_out = st.checkbox(
    COMMON_SETTINGS["display"]["show_code_out"]["name"],
    user_settings.default_session_settings.show_code_out,
    help=COMMON_SETTINGS["display"]["show_code_out"]["help"],
)
user_settings.default_session_settings.show_full_message_history = st.checkbox(
    COMMON_SETTINGS["display"]["show_full_message_history"]["name"],
    user_settings.default_session_settings.show_full_message_history,
    help=COMMON_SETTINGS["display"]["show_full_message_history"]["help"],
)
user_settings.default_session_settings.show_message_history_length = int(
    st.number_input(
        COMMON_SETTINGS["display"]["show_message_history_length"]["name"],
        value=user_settings.default_session_settings.show_message_history_length,
        help=COMMON_SETTINGS["display"]["show_message_history_length"]["help"],
        min_value=COMMON_SETTINGS["display"]["show_message_history_length"]["min_value"],
        max_value=COMMON_SETTINGS["display"]["show_message_history_length"]["max_value"],
        step=COMMON_SETTINGS["display"]["show_message_history_length"]["step"],
    )
)

with st.expander("🛑 Advanced settings", expanded=False):
    st.markdown("""
        **Only modify these settings if you know what you are doing.**
        """)
    user_settings.default_session_settings.show_planning_details = st.checkbox(
        COMMON_SETTINGS["display"]["show_planning_details"]["name"],
        user_settings.default_session_settings.show_planning_details,
        help=COMMON_SETTINGS["display"]["show_planning_details"]["help"],
    )

if st.button("Save"):
    # Save the user settings to the database.
    db.update_user_settings(user_settings)
    st.success("User settings saved successfully.")
