import copy
import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import streamlit as st
from streamlit_sortables import sort_items

import climb.ui.st_common as st_common
from climb.common.plan_files import (
    PLAN_FILES_DIR,
    PLAN_FILES_DIR_RELATIVE_STR,
    TEMPLATES_DIR,
    TEMPLATES_DIR_RELATIVE_STR,
    load_plan_and_template_files,
)
from climb.tool import list_all_tool_names

st.set_page_config(
    layout="wide", page_title=st_common.PAGE_TITLES["plan_editor_tab_title"], page_icon=st_common.CLIMB_ICON_IMAGE
)
st_common.menu()

st.markdown(f"## {st_common.PAGE_TITLES['plan_editor_emoji']}")

# -------------------- CSS Hack --------------------
# Disable the header anchors:
st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")


# ----------------------------
# 1) CONFIGURATION
# ----------------------------

# The allowed tool names. Only these can be chosen in the UI.
TOOL_NAMES: List[str] = list_all_tool_names()

# Fixed schema (users cannot add/remove these keys)
SCHEMA_KEYS: List[str] = [
    "episode_id",
    "selection_condition",
    "episode_name",
    "episode_details",
    "coordinator_guidance",
    "worker_guidance",
    "tools",
]

# Default item for "Add item"
TEMPLATE_ITEM: Dict[str, Any] = {
    "episode_id": "",
    "selection_condition": None,  # optional
    "episode_name": "",
    "episode_details": "",  # REQUIRED (non-empty)
    "coordinator_guidance": None,  # optional
    "worker_guidance": None,  # optional
    "tools": [],  # default new item => NO tools until changed
}


# -------------------- Session State --------------------

if "loaded_file" not in st.session_state:
    st.session_state.loaded_file: Path | None = None
if "plan_items" not in st.session_state:
    st.session_state.plan_items: List[Dict[str, Any]] = []
if "plan_sequence" not in st.session_state:
    st.session_state.plan_sequence: List[str] = []
if "last_serialized" not in st.session_state:
    st.session_state.last_serialized = None
if "manual_open_state" not in st.session_state:
    st.session_state.manual_open_state: Dict[str, bool] = {}
if "manual_open_state_override" not in st.session_state:
    st.session_state.manual_open_state_override: bool = False
if "is_new_from_template" not in st.session_state:
    st.session_state.is_new_from_template: bool = False
if "template_source_file" not in st.session_state:
    st.session_state.template_source_file: Path | None = None
if "selected_source" not in st.session_state:
    st.session_state.selected_source: str = "My plans"
if "new_plan_filename" not in st.session_state:
    st.session_state.new_plan_filename: str = ""


# -------------------- Utilities --------------------


def ensure_data_dir() -> None:
    PLAN_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _clone_template() -> Dict[str, Any]:
    d = copy.deepcopy(TEMPLATE_ITEM)
    d["_uid"] = str(uuid.uuid4())[:8]  # stable UI identity across re-renders
    d["_extras"] = {}  # non-schema keys preserved here (read-only)
    return d


def ensure_str_is_valid_filename(filename: str) -> str:
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Remove or replace invalid characters for Windows, macOS, and Unix
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove control characters (0-31) and DEL (127)
    filename = "".join(char for char in filename if ord(char) > 31 and ord(char) != 127)

    # Remove leading/trailing dots and spaces (after other replacements)
    filename = filename.strip(". ")

    # Ensure it has .json extension
    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    return filename


def _read_plan_file(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read plan file supporting both legacy (list of episodes) and new ({plan, episode_db}) formats.
    Returns (items, plan_sequence).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        episode_db = data.get("episode_db", [])
        plan_seq = data.get("plan", []) or []
        if not isinstance(episode_db, list):
            raise ValueError("Invalid plan file: 'episode_db' must be a list.")
        if not isinstance(plan_seq, list):
            raise ValueError("Invalid plan file: 'plan' must be a list.")
    else:
        raise ValueError("Invalid plan file: expected list or dict with 'episode_db' and 'plan'.")
    items: List[Dict[str, Any]] = []
    for i, obj in enumerate(episode_db):
        if not isinstance(obj, dict):
            raise ValueError(f"Item {i} is not an object.")
        base = _clone_template()
        # Fill known keys (respecting explicit nulls)
        for k in TEMPLATE_ITEM.keys():
            base[k] = obj.get(k, copy.deepcopy(TEMPLATE_ITEM[k]))
        # episode_details must be editable as text (not None)
        if base.get("episode_details") is None:
            base["episode_details"] = ""
        # Preserve unknown keys read-only
        extras = {k: obj[k] for k in obj.keys() if k not in SCHEMA_KEYS}
        base["_extras"] = {k: obj[k] for k in extras}
        items.append(base)
    # Normalize plan sequence to strings
    plan_seq_str = [str(x) for x in plan_seq if isinstance(x, (str, int, float))]
    return items, plan_seq_str


def _ordered_for_save(item: Dict[str, Any]) -> OrderedDict:
    """Return an OrderedDict with schema keys first (in fixed order), then extras.
    Honors 'tools' semantics:
      - None  -> save as null   (use ALL tools)
      - []    -> save as []     (NO tools)
      - [..]  -> save unique subset filtered to TOOL_NAMES
    """
    od = OrderedDict()
    od["episode_id"] = item.get("episode_id", "")
    od["selection_condition"] = item.get("selection_condition", None)
    od["episode_name"] = item.get("episode_name", "")
    od["episode_details"] = item.get("episode_details", "")
    od["coordinator_guidance"] = item.get("coordinator_guidance", None)
    od["worker_guidance"] = item.get("worker_guidance", None)

    tools_val = item.get("tools", [])
    if tools_val is None:
        od["tools"] = None
    else:
        cleaned: List[str] = []
        seen = set()
        if isinstance(tools_val, list):
            for t in tools_val:
                if t in TOOL_NAMES and t not in seen:
                    seen.add(t)
                    cleaned.append(t)
        od["tools"] = cleaned

    # Put preserved extras at the end, unchanged
    for k, v in (item.get("_extras") or {}).items():
        if k not in SCHEMA_KEYS:  # ensure extras can't override schema keys
            od[k] = v
    return od


def _serialize_for_compare(items: List[Dict[str, Any]], plan_seq: List[str]) -> str:
    """Stable serialization for 'dirty' detection (ignores _uid, but includes extras) of both plan and episodes."""
    to_save = OrderedDict()
    to_save["plan"] = list(plan_seq or [])
    to_save["episode_db"] = [_ordered_for_save(it) for it in items]
    return json.dumps(to_save, ensure_ascii=False, sort_keys=True)


def _validate(items: List[Dict[str, Any]], plan_seq: List[str]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Save is blocked if errors. Validates episodes and plan."""
    errors: List[str] = []
    warnings: List[str] = []

    # no episodes in the database
    if len(items) == 0:
        errors.append("No episodes in the database. Please add at least one episode to the database.")

    # episode_id checks
    ids = [str(it.get("episode_id") or "").strip() for it in items]
    empty_idx = [i for i, eid in enumerate(ids) if eid == ""]
    if empty_idx:
        errors.append("Empty episode_id in item index(es): " + ", ".join(str(i + 1) for i in empty_idx))
    counts = {}
    for eid in ids:
        if eid:
            counts[eid] = counts.get(eid, 0) + 1
    dupes = [eid for eid, n in counts.items() if n > 1]
    if dupes:
        errors.append("Duplicate episode_id(s): " + ", ".join(dupes))

    # required: episode_name and episode_details must be non-empty strings
    for idx, it in enumerate(items):
        name = (it.get("episode_name") or "").strip()
        if not name:
            errors.append(f"Item {idx + 1} ({it.get('episode_id') or 'UNNAMED'}): episode_name is required.")
        details = it.get("episode_details")
        if not isinstance(details, str) or len(details.strip()) == 0:
            errors.append(
                f"Item {idx + 1} ({it.get('episode_id') or 'UNNAMED'}): episode_details is required and cannot be empty."
            )

    # tools warnings (if list has any unknowns; UI prevents, but keep for safety)
    for idx, it in enumerate(items):
        tval = it.get("tools", [])
        if isinstance(tval, list):
            bad = [t for t in tval if t not in TOOL_NAMES]
            if bad:
                warnings.append(
                    f"Item {idx + 1} ({it.get('episode_id') or 'UNNAMED'}): unknown tools {bad} will be removed on save."
                )

    # ensure all schema keys exist
    for idx, it in enumerate(items):
        missing = [k for k in SCHEMA_KEYS if k not in it]
        if missing:
            errors.append(f"Item {idx + 1} missing required key(s): {missing}")

    # plan validation
    valid_ids = set([i for i in ids if i])
    invalid_in_plan = [eid for eid in (plan_seq or []) if eid not in valid_ids]
    if invalid_in_plan:
        warnings.append(f"Plan contains episode_id(s) not present in episode_db: {invalid_in_plan}")
    # duplicate plan ids
    seen_plan = set()
    dup_plan = []
    for eid in plan_seq or []:
        if eid in seen_plan:
            dup_plan.append(eid)
        else:
            seen_plan.add(eid)
    if dup_plan:
        errors.append(f"Plan has duplicate episode_id(s): {sorted(set(dup_plan))}")
    # empty plan
    if not plan_seq or len(plan_seq) == 0:
        errors.append("Plan is empty. Please add at least one episode to the plan.")

    return errors, warnings


def _save_plan(path: Path, items: List[Dict[str, Any]], plan_seq: List[str]) -> None:
    # backup if file exists
    # if path.exists():
    #     ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     backup = path.with_suffix(path.suffix + f".bak_{ts}")
    #     backup.write_bytes(path.read_bytes())

    # write plan to file:
    ordered_items = [_ordered_for_save(it) for it in items]
    top_level = OrderedDict()
    top_level["plan"] = list(plan_seq or [])
    top_level["episode_db"] = ordered_items
    with path.open("w", encoding="utf-8") as f:
        json.dump(top_level, f, ensure_ascii=False, indent=4)


def _new_episode_id(existing_ids: List[str]) -> str:
    while (new_id := f"NEW_{str(uuid4())[:4].upper()}") in existing_ids:
        pass
    return new_id


# -------- Helpers: item actions, open-state, and field editors --------

OPEN_STATE_PREFIX = "item_open_"


def _reset_manual_expand_states() -> None:
    """Reset manual expander states and remove their widget keys.
    Called when opening or reloading a file to start with all-collapsed UI.
    """
    st.session_state.manual_open_state = {}
    # Remove any prior checkbox widget states
    keys_to_delete = [
        k for k in list(st.session_state.keys()) if isinstance(k, str) and k.startswith(OPEN_STATE_PREFIX)
    ]
    for k in keys_to_delete:
        del st.session_state[k]


def move_item(idx: int, direction: int):
    j = idx + direction
    if 0 <= j < len(st.session_state.plan_items):
        st.session_state.plan_items[idx], st.session_state.plan_items[j] = (
            st.session_state.plan_items[j],
            st.session_state.plan_items[idx],
        )


def delete_item(idx: int):
    # Clean up manual open state and its checkbox key for the item being deleted
    try:
        it = st.session_state.plan_items[idx]
        uid = it.get("_uid", f"row{idx}")
        if isinstance(st.session_state.get("manual_open_state"), dict) and uid in st.session_state.manual_open_state:
            del st.session_state.manual_open_state[uid]
        widget_key = f"{OPEN_STATE_PREFIX}{uid}"
        if widget_key in st.session_state:
            del st.session_state[widget_key]
    except Exception:
        pass
    # Remove the item and re-run
    del st.session_state.plan_items[idx]


def plan_add_items(new_ids: List[str]):
    existing = set(st.session_state.plan_sequence or [])
    to_add = [e for e in new_ids if e not in existing]
    if to_add:
        st.session_state.plan_sequence = list(st.session_state.plan_sequence or []) + to_add


def optional_singleline(it: Dict[str, Any], field_key: str, label: str, widget_key_prefix: str) -> str | None:
    """Single-line text with a 'Set to None' toggle (for optional short fields)."""
    is_none = it.get(field_key) is None
    c1_optional_singleline, c2_optional_singleline = st.columns([0.84, 0.16])
    with c2_optional_singleline:
        st.markdown("")
        st.markdown("")
        set_none = st.checkbox("None", value=is_none, key=f"{widget_key_prefix}_{field_key}_none")
    with c1_optional_singleline:
        val = st.text_input(
            label,
            value=it.get(field_key) or "",
            key=f"{widget_key_prefix}_{field_key}_txt",
            disabled=set_none,
            help="To edit, uncheck 'None'." if set_none else "",
        )
    return None if set_none else val


def required_longtext(it: Dict[str, Any], field_key: str, label: str, widget_key_prefix: str, height: int = 220) -> str:
    """Required long text with Edit/Preview tabs (no None allowed)."""
    tabs = st.tabs(["✏️ Edit", "🔍 Preview"])
    with tabs[0]:
        val = st.text_area(
            label, value=it.get(field_key) or "", height=height, key=f"{widget_key_prefix}_{field_key}_ta"
        )
    with tabs[1]:
        st.markdown(val or "_(empty — fill before saving)_")
    return val


def optional_longtext(
    it: Dict[str, Any], field_key: str, label: str, widget_key_prefix: str, height: int = 200
) -> str | None:
    """Optional long text with a None toggle + Edit/Preview."""
    is_none = it.get(field_key) is None
    tabs = st.tabs(["✏️ Edit", "🔍 Preview"])
    with tabs[0]:
        set_none = st.checkbox(
            f"None (no `{field_key}` set)", value=is_none, key=f"{widget_key_prefix}_{field_key}_none"
        )
        val = st.text_area(
            label,
            value=it.get(field_key) or "",
            height=height,
            key=f"{widget_key_prefix}_{field_key}_ta",
            disabled=set_none,
            help="To edit, uncheck 'None'." if set_none else "",
        )
    with tabs[1]:
        st.markdown(val or "_(empty - enter text to see preview)_")
    return None if set_none else val


def tools_editor(it: Dict[str, Any], uid: str):
    """Three-state tools editor:
    - All tools (save as None)
    - No tools (save as [])
    - Select specific tools (unique subset of TOOL_NAMES)
    """
    current = it.get("tools", [])
    if current is None:
        mode_idx = 0  # All
    elif isinstance(current, list) and len(current) == 0:
        mode_idx = 1  # None
    else:
        mode_idx = 2  # Custom

    mode = st.radio(
        "Tools mode",
        ("All tools", "No tools", "Select tools"),
        index=mode_idx,
        key=f"tools_mode_{uid}",
        horizontal=True,
        help=(
            "'All tools' will allow this episode to use all available tools. "
            "'No tools' will prevent this episode from using any tools (code generation will still be available). "
            "'Select tools' will allow you to select specific tools from the list of "
            "[available tools](https://climb-ai.readthedocs.io/en/latest/tool.html)."
        ),
    )

    if mode.startswith("All tools"):
        it["tools"] = None
        st.info("All available tools will be used for this episode. (JSON: tools = null).")
    elif mode.startswith("No tools"):
        it["tools"] = []
        st.info("This episode will not use any tools. Code generation will still be available. (JSON: tools = []).")
    else:
        defaults = [t for t in (current or []) if isinstance(current, list) and t in TOOL_NAMES]
        chosen = st.multiselect(
            "Select tools (from [available tools](https://climb-ai.readthedocs.io/en/latest/tool.html))",
            options=TOOL_NAMES,
            default=defaults,
            key=f"tools_select_{uid}",
            placeholder="Select tools",
            help="Only names of [available tools](https://climb-ai.readthedocs.io/en/latest/tool.html) are allowed. Duplicates are removed.",
        )
        seen = set()
        deduped: List[str] = []
        for t in chosen:
            if t in TOOL_NAMES and t not in seen:
                seen.add(t)
                deduped.append(t)
        it["tools"] = deduped


# -------------------- UI: Directory & File selection --------------------

ensure_data_dir()

with st.container(border=True):
    column_infopanel_notes, col_loader_left, col_loader_right = st.columns([0.35, 0.4, 0.25])
    with column_infopanel_notes:
        st.caption(
            "**Instructions:**\n"
            "- Open an existing plan file to edit it, or create a new plan from a template.\n"
            "- Add, edit, and delete episodes in :green-background[Episode Database] to create your plan.\n"
            "- Modify the :green-background[Plan Sequence] to control the suggested order of episodes in the plan.\n"
            "- More instructions on editing are available in the corresponding sections below.\n"
            "- Save your plan by clicking the :grey-background[💾 Save changes]."
        )
    with col_loader_left:
        source_options = ("My plans", "Templates")
        selected_source = st.radio(
            "Load plan file from:",
            options=source_options,
            index=(
                source_options.index(st.session_state.selected_source)
                if st.session_state.selected_source in source_options
                else 0
            ),
            key="selected_source_radio",
            horizontal=True,
        )
        st.session_state.selected_source = selected_source
        browse_dir = PLAN_FILES_DIR if selected_source == "My plans" else TEMPLATES_DIR
        plan_and_template_files = load_plan_and_template_files()
        if selected_source == "My plans":
            files_available_names = plan_and_template_files["plan_files"]
        else:
            files_available_names = plan_and_template_files["template_plan_files"]
        select_key = "file_select_my" if selected_source == "My plans" else "file_select_tpl"
        selected_name = st.selectbox(
            (
                "Select a plan file to open then click `📂 Load`. Files are in "
                f"`{PLAN_FILES_DIR_RELATIVE_STR if selected_source == 'My plans' else TEMPLATES_DIR_RELATIVE_STR}/`."
            ),
            options=files_available_names if files_available_names else ["(no JSON files found)"],
            index=0 if not files_available_names else 0,
            key=select_key,
            disabled=not bool(files_available_names),
        )
    with col_loader_right:
        # st.caption("")
        refresh = st.button("↻ Refresh files", use_container_width=True, help="Refresh the list of available files.")
        load_btn = st.button("📂 Load", use_container_width=True, help="Load the selected file into the plan editor.")
        load_status_container = st.container()

    # st_common.horizontal_rule()
    st.markdown("")

# Explicit open action
if files_available_names and load_btn:
    path_to_open = browse_dir / selected_name
    if path_to_open.exists():
        try:
            items_loaded, plan_loaded = _read_plan_file(path_to_open)
            st.session_state.plan_items = items_loaded
            st.session_state.plan_sequence = plan_loaded
            st.session_state.loaded_file = path_to_open
            st.session_state.last_serialized = _serialize_for_compare(
                st.session_state.plan_items, st.session_state.plan_sequence
            )
            st.session_state.is_new_from_template = selected_source == "Templates"
            st.session_state.template_source_file = path_to_open if st.session_state.is_new_from_template else None
            if st.session_state.is_new_from_template:
                # When opening a template, clear any prior "Save As" filename
                st.session_state.new_plan_filename = ""
            _reset_manual_expand_states()
            with load_status_container:
                src_label = "template" if st.session_state.is_new_from_template else "plan"
                st.success(f"Opened {src_label}: `{path_to_open.name}`")
        except Exception as e:
            with load_status_container:
                st.error(f"Failed to open file: {e}")

current_file: Path | None = st.session_state.loaded_file
items: List[Dict[str, Any]] = st.session_state.plan_items

# -------------------- Toolbar --------------------

with st.container():
    col_menu_current_file, col_menu_reset, col_menu_delete, col_menu_save = st.columns([0.6, 0.1, 0.1, 0.2])
    with col_menu_current_file:
        if current_file is None:
            st.markdown(":red-background[No plan loaded, load it above]")
    with col_menu_reset:
        reload_btn = st.button(
            "⟲ Reset",
            use_container_width=True,
            disabled=current_file is None,
            help="Reset all edits and reload from source file.",
        )
    # c4: Has the save button, and due to streamlit operation ordering, this must go after the editable elements list
    # as otherwise the changes will not be correctly reflected.

    col_statuses_left, column_infopanel_statuses = st.columns([0.3, 0.7])
    with col_statuses_left:
        st.markdown("Any messages about the status of your plan will appear here ➡")


if reload_btn and current_file is not None:
    try:
        items_loaded, plan_loaded = _read_plan_file(current_file)
        st.session_state.plan_items = items_loaded
        st.session_state.plan_sequence = plan_loaded
        st.session_state.last_serialized = _serialize_for_compare(
            st.session_state.plan_items, st.session_state.plan_sequence
        )
        _reset_manual_expand_states()
        with column_infopanel_statuses:
            st.info(f"Reloaded from file: {current_file.name}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to reload: {e}")

with col_menu_delete:
    delete_btn = st.button(
        "🗑️ Delete",
        use_container_width=True,
        type="secondary",
        disabled=st.session_state.is_new_from_template or current_file is None,
        help="Delete the current plan file.",
    )


# Explicit delete action (only for 'My plans')
if current_file is not None and not st.session_state.is_new_from_template and delete_btn:
    path_to_delete = current_file.resolve()
    try:
        if path_to_delete.exists() and path_to_delete.is_file():
            # If the file being deleted is currently loaded, clear the editor state
            if st.session_state.loaded_file and st.session_state.loaded_file.resolve() == path_to_delete.resolve():
                st.session_state.loaded_file = None
                st.session_state.plan_items = []
                st.session_state.plan_sequence = []
                st.session_state.last_serialized = None
                st.session_state.is_new_from_template = False
                st.session_state.template_source_file = None
                _reset_manual_expand_states()
            # Delete the file
            path_to_delete.unlink()
            with load_status_container:
                st.success(f"Deleted plan: `{path_to_delete.name}`")
            st.rerun()
        else:
            with load_status_container:
                st.error("Selected file does not exist or is not a file.")
    except Exception as e:
        with load_status_container:
            st.error(f"Failed to delete file: {e}")

if current_file is not None:
    with column_infopanel_statuses:
        if not items:
            st.info("This file currently contains an empty list. Use **Add item** to begin.")


st_common.horizontal_rule()
st.markdown("")

# -------------------- Plan (Episode Ordered List) Editor --------------------

if current_file is None:
    st.stop()

plan_editor_container = st.container(border=False)

st_common.horizontal_rule()
st.markdown("")

# -------------------- Items Editor --------------------


def on_expand_collapse_all_toggle_change():
    value = st.session_state.get("expand_collapse_all_toggle", False)
    if value:
        for it in items:
            st.session_state.manual_open_state[it.get("_uid")] = True
    else:
        for it in items:
            st.session_state.manual_open_state[it.get("_uid")] = False
    st.session_state.manual_open_state_override = True
    # ^ Allows to temporarily ignore the UI open state of the episodes, and force open/close all of them.


col_title_episodes_database, col_expand_collapse_all_episodes, col_add_episode_to_db = st.columns([0.7, 0.2, 0.1])
with col_title_episodes_database:
    st.markdown("#### Episode database:")
with col_expand_collapse_all_episodes:
    expand_collapse_all_toggle = st.toggle(
        "Expand/collapse all episodes",
        value=False,
        key="expand_collapse_all_toggle",
        on_change=on_expand_collapse_all_toggle_change,
        help="Expand/collapse all episodes below.",
    )
with col_add_episode_to_db:
    add_btn = st.button(
        "➕ Add item",
        use_container_width=True,
        disabled=current_file is None,
        help="Add a new item to the episode database.",
    )

with st.expander(":material/info: Episode database explanation"):
    col_episodes_db_explanation_1, col_episodes_db_explanation_2 = st.columns([0.4, 0.6])
    with col_episodes_db_explanation_1:
        st.caption("""
        **General instructions:**
        * The episode database contains all the possible episodes that can be used by the CliMB engine.
        * You can add new episodes to the database by clicking the `➕ Add item` button above.
        * You can delete episodes from the database by clicking the `🗑️ Delete` button next to the episode.
        * You can move episodes up and down in the database by clicking the `↑ Move up` and `↓ Move down` buttons next to the episode. This is only for your convenience, the engine will not use this order.
        * You can edit the different fields for each episode by expanding that episode's panel (use the checkbox on the left of the episode title).
        """)
    with col_episodes_db_explanation_2:
        st.caption("""
        **Episode fields:**
        * `episode_id`: must be unique and is used to identify the episode in the plan.
        * `episode_name`: is the name of the episode, make it clear and concise.
        * `episode_details`: is the detailed description of the episode, it should contain the specific task that the episode is designed to complete, for the worker agent.
        * `coordinator_guidance` (optional): is the guidance for the coordinator (planner) agent about the episode.
        * `worker_guidance` (optional): is the guidance for the worker agent about the episode (to supplement the `episode_details`).
        * `tools`: configures the selection of tools that the worker can use to complete the episode.
        * `selection_condition` (optional): is the (concise) condition under which the episode should be selected, for the coordinator agent.
        """)

# Toolbar actions
if add_btn and current_file is not None:
    new_item = _clone_template()
    existing_ids = [str(it.get("episode_id") or "").strip() for it in items]
    new_item["episode_id"] = _new_episode_id(existing_ids)
    new_item["episode_name"] = "(Enter episode name)"
    new_item["episode_details"] = "(Enter episode details)"
    items.append(new_item)
    st.session_state.plan_items = items
    st.rerun()

for idx, it in enumerate(items):
    uid = it.get("_uid", f"row{idx}")

    # Header with manual toggle next to title
    with st.container(border=True):
        title = f"`{it.get('episode_id') or 'NEW'}` {it.get('episode_name') or ''}"
        col_episode_1, col_episode_2, col_episode_btns = st.columns([0.06, 0.6, 0.34])
        with col_episode_1:
            if st.session_state.manual_open_state_override:
                # Manually set the checkbox state when the override (expand/collapse all episodes toggle) is active.
                st.session_state[f"{OPEN_STATE_PREFIX}{uid}"] = st.session_state.manual_open_state.get(uid, False)
            is_open = st.checkbox(
                "❯",
                value=bool(st.session_state.manual_open_state.get(uid, False)),
                key=f"{OPEN_STATE_PREFIX}{uid}",
                # help="Use the checkbox to open/close the editor for each item",
            )
            if not st.session_state.manual_open_state_override:
                # React to the checkbox state change when the override (expand/collapse all episodes toggle)
                # is not active.
                st.session_state.manual_open_state[uid] = bool(is_open)
        with col_episode_2:
            st.markdown(f"##### {title}")
        with col_episode_btns:
            # Row actions
            col_episode_btns_up, col_episode_btns_down, col_episode_btns_delete = st.columns([1, 1, 1])
            with col_episode_btns_up:
                st.button(
                    "↑ Move up",
                    key=f"up_{uid}",
                    on_click=move_item,
                    args=(idx, -1),
                    disabled=(idx == 0),
                    use_container_width=True,
                )
            with col_episode_btns_down:
                st.button(
                    "↓ Move down",
                    key=f"down_{uid}",
                    on_click=move_item,
                    args=(idx, +1),
                    disabled=(idx == len(items) - 1),
                    use_container_width=True,
                )
            with col_episode_btns_delete:
                st.button("🗑️ Delete", key=f"del_{uid}", on_click=delete_item, args=(idx,), use_container_width=True)

        if st.session_state.manual_open_state.get(uid, False):
            st_common.horizontal_rule()

            # Top fields
            col_episode_fields_id, col_episode_fields_name, col_episode_fields_sel = st.columns([0.3, 0.35, 0.35])
            with col_episode_fields_id:
                it["episode_id"] = st.text_input(
                    "episode_id (must be unique)", value=it.get("episode_id") or "", key=f"eid_{uid}"
                ).strip()
            with col_episode_fields_name:
                it["episode_name"] = st.text_input(
                    "episode_name (required)", value=it.get("episode_name") or "", key=f"name_{uid}"
                )
            with col_episode_fields_sel:
                it["selection_condition"] = optional_singleline(
                    it, "selection_condition", "selection_condition (optional)", widget_key_prefix=f"sel_{uid}"
                )

            st.markdown("")

            # Required long text with preview
            it["episode_details"] = required_longtext(
                it,
                "episode_details",
                "episode_details **(Markdown supported)**",
                widget_key_prefix=f"details_{uid}",
                height=220,
            )

            st.markdown("")

            # Optional long texts with preview + None toggle
            col_episode_fields_cg, col_episode_fields_wg = st.columns(2)
            with col_episode_fields_cg:
                it["coordinator_guidance"] = optional_longtext(
                    it,
                    "coordinator_guidance",
                    "coordinator_guidance **(Markdown supported)**",
                    widget_key_prefix=f"cg_{uid}",
                    height=200,
                )
            with col_episode_fields_wg:
                it["worker_guidance"] = optional_longtext(
                    it,
                    "worker_guidance",
                    "worker_guidance **(Markdown supported)**",
                    widget_key_prefix=f"wg_{uid}",
                    height=200,
                )

            st.markdown("")

            # Tools editor (supports None / [] / explicit list)
            tools_editor(it, uid)

            # Read-only unknown keys (preserved as-is)
            extras = it.get("_extras") or {}
            if extras:
                with st.container():
                    st.markdown("_Read-only extra keys (preserved on save)_")
                    st.json(extras)

if current_file is not None:
    with plan_editor_container:
        col_plan_left, col_plan_right = st.columns([0.4, 0.6])
        with col_plan_left:
            st.markdown("#### Plan sequence:")
            st.caption("""
            * The plan sequence is the default order of the episodes from the episode database) given to the CliMB engine.
            * The engine may review and modify the plan sequence, it will only use this as a starting point.
            * Edit the plan sequence in this section.
            * You can add, remove, and reorder episodes by their `episode_id` in this section.
            """)

            # Sync plan with current episode IDs
            current_ids = [
                str(it.get("episode_id") or "").strip() for it in st.session_state.plan_items if it.get("episode_id")
            ]
            valid_id_set = set(current_ids)
            plan_before = list(st.session_state.plan_sequence or [])
            missing_in_db = [eid for eid in plan_before if eid not in valid_id_set]
            if missing_in_db:
                # Auto-remove missing to keep plan consistent
                st.session_state.plan_sequence = [eid for eid in plan_before if eid in valid_id_set]
                st.warning(f"Removed non-existent episode_id(s) from plan: {missing_in_db}")

            # Duplicates warning (allow but warn)
            seen_tmp = set()
            dup_tmp = []
            for eid in st.session_state.plan_sequence:
                if eid in seen_tmp:
                    dup_tmp.append(eid)
                seen_tmp.add(eid)
            if dup_tmp:
                st.warning(f"Plan has duplicate episode_id(s): {sorted(set(dup_tmp))}")

        with col_plan_right:
            # col_plan_add_episodes, col_remove_episodes, col_clear_plan = st.columns([0.45, 0.45, 0.1])
            # with col_plan_add_episodes:
            # Add to plan
            available_to_add = [eid for eid in current_ids if eid not in set(st.session_state.plan_sequence or [])]
            add_cols = st.columns([0.7, 0.3])
            with add_cols[0]:
                add_select = st.multiselect(
                    "Add episodes to the end of the plan",
                    options=available_to_add,
                    key="plan_add_select",
                    placeholder="Select episode_id(s) to add",
                )
            with add_cols[1]:
                st.markdown("")
                if st.button(
                    "➕ Add to plan",
                    use_container_width=True,
                    disabled=len(add_select) == 0,
                    help="Selected episode_id(s) will be added to the end of the plan.",
                ):
                    plan_add_items(add_select)

            st_common.horizontal_rule()

            # Current plan rows
            if not st.session_state.plan_sequence:
                st.info("Plan is currently empty. Add episode_id(s) above.")
            else:
                # Removal of episodes from the plan:
                # with col_remove_episodes:
                rem_cols = st.columns([0.7, 0.3])
                with rem_cols[0]:
                    remove_select = st.multiselect(
                        "Remove episodes from the plan",
                        options=st.session_state.plan_sequence,
                        key="plan_remove_select",
                        placeholder="Select episode_id(s) to remove",
                    )
                with rem_cols[1]:
                    st.markdown("")
                    if st.button(
                        "🗑️ Remove selected",
                        use_container_width=True,
                        disabled=len(remove_select) == 0,
                        help="Selected episode_id(s) will be removed from the plan.",
                    ):
                        st.session_state.plan_sequence = [
                            eid for eid in st.session_state.plan_sequence if eid not in set(remove_select)
                        ]

                    # Clearing the plan:
                    # with col_clear_plan:
                    if st.button(
                        "Clear plan",
                        use_container_width=True,
                        type="secondary",
                        disabled=not bool(st.session_state.plan_sequence),
                        help="All episode_id(s) will be removed from the plan.",
                    ):
                        st.session_state.plan_sequence = []

        # Reordering the plan:
        col_plan_reordering_caption, col_plan_reordering_clear_btn = st.columns([0.8, 0.2])
        st.caption(
            "Drag episode IDs below to reorder. The order here (left to right, may go onto multiple lines) is the current plan order."
        )
        sorted_items = sort_items(st.session_state.plan_sequence)
        if sorted_items != st.session_state.plan_sequence:
            st.session_state.plan_sequence = sorted_items

if current_file is not None:
    dirty = st.session_state.last_serialized != _serialize_for_compare(items, st.session_state.plan_sequence)

    with column_infopanel_statuses:
        if dirty:
            st.info("There are unsaved changes.")

        errs, warns = _validate(items, st.session_state.plan_sequence)

        if errs:
            st.error("Problem with the plan:\n\n- " + "\n- ".join(errs))
        elif warns:
            st.warning("Warnings:\n\n- " + "\n- ".join(warns))

    with col_menu_save:
        errs, warns = _validate(items, st.session_state.plan_sequence)
        # For 'new from template', prompt for a filename under PLAN_FILES_DIR
        save_disabled_extra = False
        dest_path_preview_str = ""
        if st.session_state.is_new_from_template:
            st.session_state.new_plan_filename = st.text_input(
                f"Save as filename (under `{PLAN_FILES_DIR_RELATIVE_STR}/`)",
                value=st.session_state.new_plan_filename,
                key="new_plan_filename_input",
                placeholder="Enter plan name, e.g. 'my_plan.json'",
                help=f"Enter a filename to save this template as a new plan in `{PLAN_FILES_DIR_RELATIVE_STR}/`.",
            ).strip()
            pan_file_being_edited_filename = (
                ensure_str_is_valid_filename(st.session_state.new_plan_filename)
                if st.session_state.new_plan_filename
                else ""
            )
            save_disabled_extra = not bool(pan_file_being_edited_filename)
            dest_path_preview_str = (
                ("./" + str(PLAN_FILES_DIR / pan_file_being_edited_filename)) if pan_file_being_edited_filename else ""
            )

        if errs:
            save_helper_text = "Fix the problems with the plan first."
        elif not dirty:
            save_helper_text = "No changes to save."
        elif st.session_state.is_new_from_template:
            if dest_path_preview_str:
                save_helper_text = f"Save to {dest_path_preview_str}"
            else:
                save_helper_text = "Enter a name for the new plan to be able to save it"
        else:
            save_helper_text = "Save to disk."
        save_btn = st.button(
            "💾 Save changes",
            type="primary",
            use_container_width=True,
            disabled=bool(errs) or not dirty or save_disabled_extra,
            help=save_helper_text,
        )

    with col_menu_current_file:
        if st.session_state.is_new_from_template:
            pan_file_being_edited_filename = (
                ensure_str_is_valid_filename(st.session_state.new_plan_filename)
                if st.session_state.new_plan_filename
                else "..."
            )
            pan_file_being_edited_path = (PLAN_FILES_DIR / pan_file_being_edited_filename).resolve()
            st.markdown(
                f"### Editing template: `{current_file.name}` → will save to `{os.path.join(PLAN_FILES_DIR_RELATIVE_STR, pan_file_being_edited_filename)}`"
            )
        else:
            st.markdown(f"### Editing plan file: `{current_file.name}`")

    if save_btn:
        if st.session_state.is_new_from_template:
            _save_plan(pan_file_being_edited_path, items, st.session_state.plan_sequence)
            st.session_state.loaded_file = pan_file_being_edited_path
            st.session_state.last_serialized = _serialize_for_compare(items, st.session_state.plan_sequence)
            st.session_state.is_new_from_template = False
            st.session_state.template_source_file = None
            # Switch UI selection to the newly saved plan
            st.session_state.selected_source = "My plans"
            with column_infopanel_statuses:
                st.success(f"Saved new plan: {pan_file_being_edited_path.name}")
        else:
            _save_plan(current_file, items, st.session_state.plan_sequence)
            st.session_state.last_serialized = _serialize_for_compare(items, st.session_state.plan_sequence)
            with column_infopanel_statuses:
                st.success(f"Saved: {current_file.name}")
        st.rerun()

# Reset the manual open state override (for expand/collapse all episodes toggle).
st.session_state.manual_open_state_override = False
