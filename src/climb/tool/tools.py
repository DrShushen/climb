import abc
from typing import Any, Dict, List, Optional

from climb.common.data_structures import UserInputRequest

from .tool_comms import ToolReturnIter, ToolThread

# TODO: Success/fail
# TODO: Streaming of STDOUT/STDERR


def get_str_up_to_marker(s: str, marker: str) -> str:
    if marker in s:
        return s[: s.index(marker)]
    return s


class ToolBase(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.user_input: Optional[UserInputRequest] = None
        self.working_directory: str = "TO_BE_SET_BY_ENGINE"
        self.tool_thread: Optional[ToolThread] = None

    def execute(self, **kwargs: Any) -> ToolReturnIter:
        # For now this is just a "pass-through".
        if "session" not in kwargs:
            raise ValueError("Expected 'session' in kwargs.")
        if "additional_kwargs_required" not in kwargs:
            raise ValueError("Expected 'additional_kwargs_required' in kwargs.")
        yield from self._execute(**kwargs)

    def stop_execution(self, timeout: Optional[int] = 1) -> None:
        if self.tool_thread is not None:
            self.tool_thread.kill(timeout=timeout)
            print("ToolBase.stop_execution(): Tool thread killed.")

    @abc.abstractmethod
    def _execute(self, **kwargs: Any) -> ToolReturnIter: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def description(self) -> str: ...

    @property
    def logs_useful(self) -> bool:
        """Return `True` if the logs of this tool are *especially* useful for the LLM to understand what has been done.

        This will be used by the engine to determine whether to shorten the logs if needed for token reasons etc. This
        is up to the engine's discretion, this property just provides a hint.

        The user will always be able to see the full logs.

        Returns:
            bool: `True` if the logs of this tool are *especially* useful for the LLM to understand what has been done.
        """
        return False

    # TODO: Factor this out to be engine-specific!
    @property
    @abc.abstractmethod
    def specification(self) -> Dict[str, Any]: ...

    @property
    def user_input_requested(self) -> List[UserInputRequest]:
        return []

    def receive_user_inputs_requested(self, user_input: Optional[UserInputRequest]) -> None:
        self.user_input = user_input

    def receive_working_directory(self, working_directory: str) -> None:
        self.working_directory = working_directory

    # NOTE: Should make sense in the context "This tool <description_for_user>"
    @property
    @abc.abstractmethod
    def description_for_user(self) -> str:
        """A description of what this tool does, for the user. Should make sense in the context:
        "This tool `<description_for_user>`."
        """
        ...
