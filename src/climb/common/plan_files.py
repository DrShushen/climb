from pathlib import Path

import rich.pretty

REPO_PATH = Path(__file__).parent.parent.parent.parent

# User plan files directory:
PLAN_FILES_DIR = REPO_PATH / "plans"
# Template / default plan files directory:
TEMPLATES_DIR = PLAN_FILES_DIR / "defaults"

TEMPLATES_DIR_STR = str(TEMPLATES_DIR)
PLAN_FILES_DIR_RELATIVE = PLAN_FILES_DIR.resolve().relative_to(REPO_PATH)
TEMPLATES_DIR_RELATIVE = TEMPLATES_DIR.resolve().relative_to(REPO_PATH)

# And string versions of each (for UI display convenience):
REPO_PATH_STR = str(REPO_PATH)
PLAN_FILES_DIR_STR = str(PLAN_FILES_DIR)
PLAN_FILES_DIR_RELATIVE_STR = "./" + str(PLAN_FILES_DIR_RELATIVE)
TEMPLATES_DIR_RELATIVE_STR = "./" + str(TEMPLATES_DIR_RELATIVE)


def load_plan_and_template_files() -> dict[str, list[str]]:
    """Load all .json files from TEMPLATES_DIR and PLAN_FILES_DIR.

    Returns:
        dict with keys "plan_files" and "template_plan_files", each containing
        a list of .json filenames (without path, just the filename).
    """
    plan_files = []
    template_plan_files = []

    # Load plan files
    if PLAN_FILES_DIR.exists():
        plan_files = [f.name for f in PLAN_FILES_DIR.glob("*.json") if f.is_file()]
        plan_files.sort()

    # Load template files
    if TEMPLATES_DIR.exists():
        template_plan_files = [f.name for f in TEMPLATES_DIR.glob("*.json") if f.is_file()]
        template_plan_files.sort()

    return {"plan_files": plan_files, "template_plan_files": template_plan_files}


if __name__ == "__main__":
    # Check by running: python src/climb/common/plan_files.py

    print("Repo path:")
    # STR
    print(f"REPO_PATH_STR:\t{REPO_PATH_STR}")
    # NON-STR
    print(f"REPO_PATH:\t{REPO_PATH}")
    print()

    print("User plan files directory:")
    # STR
    print(f"PLAN_FILES_DIR_STR:\t{PLAN_FILES_DIR_STR}")
    # NON-STR
    print(f"PLAN_FILES_DIR:\t\t{PLAN_FILES_DIR}")
    print()

    print("Templates directory:")
    # STR
    print(f"TEMPLATES_DIR_STR:\t{TEMPLATES_DIR_STR}")
    # NON-STR
    print(f"TEMPLATES_DIR:\t\t{TEMPLATES_DIR}")
    print()

    print("User plan files directory relative to repo path:")
    # STR
    print(f"PLAN_FILES_DIR_RELATIVE_STR:\t{PLAN_FILES_DIR_RELATIVE_STR}")
    # NON-STR
    print(f"PLAN_FILES_DIR_RELATIVE:\t{PLAN_FILES_DIR_RELATIVE}")
    print()

    print("Templates directory relative to repo path:")
    # STR
    print(f"TEMPLATES_DIR_RELATIVE_STR:\t{TEMPLATES_DIR_RELATIVE_STR}")
    # NON-STR
    print(f"TEMPLATES_DIR_RELATIVE:\t\t{TEMPLATES_DIR_RELATIVE}")
    print()

    print("Plan and template files:")
    plan_and_template_files = load_plan_and_template_files()
    rich.pretty.pprint(plan_and_template_files)
    print()
