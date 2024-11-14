"""CliMB exceptions."""

DOCS_BASE_URL = "https://climb-ai.readthedocs.io/en/latest/"

EXC_DOCS_REFS = {
    "troubleshooting_win_pangoft": f"{DOCS_BASE_URL}troubleshooting.html#windows-pangoft2",
    "troubleshooting_conda_not_founc": f"{DOCS_BASE_URL}troubleshooting.html#conda-not-found",
}


class ClimbConfigurationError(Exception):
    """CliMB configuration-related error."""

    pass
