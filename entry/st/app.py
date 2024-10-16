import streamlit as st

import climb.ui.st_common as st_common

st_common.menu()

# NOTE: This page should not end up being displayed to the user (we auto-navigate to `main.py` page), but just in case:
st.markdown("⇦ Please select a page from the sidebar.")
