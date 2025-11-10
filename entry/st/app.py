import streamlit as st

import climb.ui.st_common as st_common

st_common.menu()

st.set_page_config(
    page_title=st_common.TITLE,
    page_icon=st_common.CLIMB_ICON_IMAGE,
)

# NOTE: This page should not end up being displayed to the user (we auto-navigate to `main.py` page), but just in case:
st.markdown("⇦ Please select a page from the sidebar.")
st.markdown(
    f"🛈 If this is your first time using **{st_common.TITLE}**, start at the **{st_common.PAGE_TITLES['research_management_emoji']}** page."
)
