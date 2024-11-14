from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "climb-ai"  # i.e. the PyPI name.
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import common, db, engine, tool, ui

print(common.disclaimer.DISCLAIMER_SECTION)

__all__ = [
    "common",
    "db",
    "engine",
    "tool",
    "ui",
]
