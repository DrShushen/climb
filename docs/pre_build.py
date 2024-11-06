"""A docs building code that must be ran when building CliMB docs.

The code will:
* Update ``docs/overview.md`` based on the main ``README.md`` (carries out some string substitutions etc.).
* Create the PyPI version of ``README.md``, ``pypi.md``, with image links fixed.
"""

import os
import re

print("=" * 80)
print("Running `pre_build.py`...")
print("=" * 80)

REPO_URL_ROOT = "https://github.com/vanderschaarlab/climb/"
REPO_URL_TREE = f"{REPO_URL_ROOT}tree/main/"

# -- Convert `README.md` into `overview.md`.
print("Working on `docs/overview.md`...")

README_PATH = os.path.join(os.path.dirname(__file__), "../README.md")
OVERVIEW_PATH = os.path.join(os.path.dirname(__file__), "overview.md")

REPLACE = {
    # Add more as necessary.
    # Uncomment docs-only sections.
    "<!-- include_docs": "",
    "include_docs_end -->": "",
    # Fix links (we are in `docs/` now).
    "./docs/": "",
    "docs/": "",
    # Fix images.
    "./#-": "#",
    "./": REPO_URL_TREE,
    # Fix the automatic HTML anchors.
    "(#1--": "(#",
    "(#2--": "(#",
    "(#3--": "(#",
    "(#-": "(#",
}

with open(README_PATH, "r", encoding="utf8") as file:
    readme_content = file.read()

# Replace:
for k, v in REPLACE.items():
    readme_content = readme_content.replace(k, v)

# Remove parts that should only be in repo `README.md`.
readme_content = re.sub(r"\n<!-- exclude_docs -->.*?<!-- exclude_docs_end -->", "", readme_content, flags=re.DOTALL)

with open(OVERVIEW_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)


# -- Convert `README.md` into `pypi.md`.
print("Working on `pypi.md`...")

PYPI_README_PATH = os.path.join(os.path.dirname(__file__), "../pypi.md")

REPLACE = {
    # Add more as necessary.
    "<!-- include_pypi": "",
    "include_pypi_end -->": "",
    "./": REPO_URL_TREE,
}

with open(README_PATH, "r", encoding="utf8") as file:
    readme_content = file.read()

# Replace:
for k, v in REPLACE.items():
    readme_content = readme_content.replace(k, v)

# Remove parts that should only be in repo `README.md`.
readme_content = re.sub(r"\n<!-- exclude_pypi -->.*?<!-- exclude_pypi_end -->", "", readme_content, flags=re.DOTALL)

# Fix images:
convert = {
    r"\"docs/assets/(.*?\..*?)\"": r"'https://raw.githubusercontent.com/vanderschaarlab/climb/main/docs/assets/\1'",
    r"\[docs/assets/(.*?\..*?)\]": r"[https://raw.githubusercontent.com/vanderschaarlab/climb/main/docs/assets/\1]",
}
for source, destination in convert.items():
    readme_content = re.sub(source, destination, readme_content, flags=re.DOTALL)

with open(PYPI_README_PATH, "w", encoding="utf8") as file:
    file.write(readme_content)

print("=" * 80)
print("Running `pre_build.py` complete.")
print("=" * 80)
