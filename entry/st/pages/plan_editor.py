import copy
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import streamlit as st

import climb.ui.st_common as st_common
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

PLANS_DIR = Path("./plans")
TEMPLATES_DIR = Path("./plans/defaults")

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
if "last_serialized" not in st.session_state:
    st.session_state.last_serialized = None
if "manual_open_state" not in st.session_state:
    st.session_state.manual_open_state: Dict[str, bool] = {}
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
    PLANS_DIR.mkdir(parents=True, exist_ok=True)


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


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.glob("*.json") if p.is_file()])


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of objects.")
    items: List[Dict[str, Any]] = []
    for i, obj in enumerate(data):
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
    return items


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


def _serialize_items_for_compare(items: List[Dict[str, Any]]) -> str:
    """Stable serialization for 'dirty' detection (ignores _uid, but includes extras)."""
    to_save = []
    for it in items:
        to_save.append(_ordered_for_save(it))
    return json.dumps(to_save, ensure_ascii=False, sort_keys=True)


def _validate(items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Save is blocked if errors."""
    errors: List[str] = []
    warnings: List[str] = []

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

    return errors, warnings


def _save_plan(path: Path, items: List[Dict[str, Any]]) -> None:
    # backup if file exists
    # if path.exists():
    #     ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     backup = path.with_suffix(path.suffix + f".bak_{ts}")
    #     backup.write_bytes(path.read_bytes())

    # write plan to file:
    ordered_items = [_ordered_for_save(it) for it in items]
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered_items, f, ensure_ascii=False, indent=4)


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
    repo_path = Path(__file__).parent.parent.parent.parent
    col_loader_left, col_loader_right = st.columns([0.7, 0.3])
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
        browse_dir = PLANS_DIR if selected_source == "My plans" else TEMPLATES_DIR
        browse_dir_relative = browse_dir.resolve().relative_to(repo_path)
        files = list_json_files(browse_dir)
        files_available_names = [p.name for p in files]
        select_key = "file_select_my" if selected_source == "My plans" else "file_select_tpl"
        selected_name = st.selectbox(
            f"Select a plan file to open then click `📂 Load`. Files are in `./{browse_dir_relative}/`.",
            options=files_available_names if files_available_names else ["(no JSON files found)"],
            index=0 if not files_available_names else 0,
            key=select_key,
            disabled=not bool(files_available_names),
        )
    with col_loader_right:
        # st.caption("")
        refresh = st.button("↻ Refresh files", use_container_width=True)
        load_btn = st.button("📂 Load", use_container_width=True)
        load_status_container = st.container()

    # st_common.horizontal_rule()
    st.markdown("")

# Explicit open action
if files_available_names and load_btn:
    path_to_open = browse_dir / selected_name
    if path_to_open.exists():
        try:
            st.session_state.plan_items = _read_json(path_to_open)
            st.session_state.loaded_file = path_to_open
            st.session_state.last_serialized = _serialize_items_for_compare(st.session_state.plan_items)
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
    col_menu_current_file, col_menu_add_item, col_menu_reset, col_menu_save = st.columns([0.6, 0.1, 0.1, 0.2])
    with col_menu_current_file:
        if current_file is None:
            st.markdown("_No file open_")
    with col_menu_add_item:
        add_btn = st.button(
            "➕ Add item",
            use_container_width=True,
            disabled=current_file is None,
            help="Add a new item to the episode list.",
        )
    with col_menu_reset:
        reload_btn = st.button(
            "⟲ Reset",
            use_container_width=True,
            disabled=current_file is None,
            help="Reset all edits and reload from source file.",
        )
    # c4: Has the save button, and due to streamlit operation ordering, this must go after the editable elements list
    # as otherwise the changes will not be correctly reflected.

    column_infopanel_notes, column_infopanel_statuses = st.columns([0.5, 0.5])
    with column_infopanel_notes:
        st.caption(
            "- Open an existing plan file to edit it, or create a new plan from a template.\n"
            "- Add, edit, and delete episodes to create your plan.\n"
            "- Save your plan to a file in the `./plans/` directory."
        )


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

if reload_btn and current_file is not None:
    try:
        st.session_state.plan_items = _read_json(current_file)
        st.session_state.last_serialized = _serialize_items_for_compare(st.session_state.plan_items)
        _reset_manual_expand_states()
        with column_infopanel_statuses:
            st.info(f"Reloaded from file: {current_file.name}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to reload: {e}")


if current_file is not None:
    with column_infopanel_statuses:
        if not items:
            st.info("This file currently contains an empty list. Use **Add item** to begin.")


st_common.horizontal_rule()
st.markdown("")

# -------------------- Items Editor --------------------

if current_file is None:
    st.stop()

for idx, it in enumerate(items):
    uid = it.get("_uid", f"row{idx}")

    # Header with manual toggle next to title
    with st.container(border=True):
        title = f"`{it.get('episode_id') or 'NEW'}` {it.get('episode_name') or ''}"
        col_episode_1, col_episode_2, col_episode_btns = st.columns([0.06, 0.6, 0.34])
        with col_episode_1:
            is_open = st.checkbox(
                "❯",
                value=bool(st.session_state.manual_open_state.get(uid, False)),
                key=f"{OPEN_STATE_PREFIX}{uid}",
                # help="Use the checkbox to open/close the editor for each item",
            )
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
    dirty = st.session_state.last_serialized != _serialize_items_for_compare(items)

    with column_infopanel_statuses:
        # DEBUG:
        # import rich.pretty
        # print("items serialized:")
        # rich.pretty.pprint(_serialize_items_for_compare(items))
        # print("st.session_state.last_serialized:")
        # rich.pretty.pprint(st.session_state.last_serialized)
        # print("items serialized == st.session_state.last_serialized:")
        # print(_serialize_items_for_compare(items) == st.session_state.last_serialized)

        if dirty:
            st.info("There are unsaved changes.")

        errs, warns = _validate(items)

        if errs:
            st.error("Validation errors:\n\n- " + "\n- ".join(errs))
        elif warns:
            st.warning("Warnings:\n\n- " + "\n- ".join(warns))

    with col_menu_save:
        errs, warns = _validate(items)
        # For 'new from template', prompt for a filename under PLANS_DIR
        save_disabled_extra = False
        dest_path_preview_str = ""
        if st.session_state.is_new_from_template:
            st.session_state.new_plan_filename = st.text_input(
                "Save as filename (under `./plans/`)",
                value=st.session_state.new_plan_filename,
                key="new_plan_filename_input",
                placeholder="Enter plan name, e.g. 'my_plan.json'",
                help="Enter a filename to save this template as a new plan in `./plans/`.",
            ).strip()
            dest_name = (
                ensure_str_is_valid_filename(st.session_state.new_plan_filename)
                if st.session_state.new_plan_filename
                else ""
            )
            save_disabled_extra = not bool(dest_name)
            dest_path_preview_str = ("./" + str(PLANS_DIR / dest_name)) if dest_name else ""

        save_btn = st.button(
            "💾 Save changes",
            type="primary",
            use_container_width=True,
            disabled=bool(errs) or not dirty or save_disabled_extra,
            help=(
                "Fix validation errors first."
                if errs
                else (
                    "No changes to save."
                    if not dirty
                    else (
                        f"Save to {dest_path_preview_str}"
                        if st.session_state.is_new_from_template and dest_path_preview_str
                        else "Save to disk."
                    )
                )
            ),
        )

    with col_menu_current_file:
        if st.session_state.is_new_from_template:
            dest_name = (
                ensure_str_is_valid_filename(st.session_state.new_plan_filename)
                if st.session_state.new_plan_filename
                else "..."
            )
            st.markdown(f"##### Editing template: `{current_file.name}` → will save to `./{PLANS_DIR}/{dest_name}`")
        else:
            st.markdown(f"##### Editing plan file: `{current_file.name}`")

    if save_btn:
        if st.session_state.is_new_from_template:
            # Determine destination path within PLANS_DIR
            dest_path = (PLANS_DIR / dest_name).resolve()
            _save_plan(dest_path, items)
            st.session_state.loaded_file = dest_path
            st.session_state.last_serialized = _serialize_items_for_compare(items)
            st.session_state.is_new_from_template = False
            st.session_state.template_source_file = None
            # Switch UI selection to the newly saved plan
            st.session_state.selected_source = "My plans"
            with column_infopanel_statuses:
                st.success(f"Saved new plan: {dest_path.name}")
            # st.rerun()
        else:
            _save_plan(current_file, items)
            st.session_state.last_serialized = _serialize_items_for_compare(items)
            with column_infopanel_statuses:
                st.success(f"Saved: {current_file.name}")
