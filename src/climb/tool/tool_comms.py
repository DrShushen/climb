import copy
import io
import os
import queue
import sys
import threading
import traceback
from functools import partial
from typing import Any, Callable, Iterable, List, NoReturn, Optional, Tuple, Union

from climb.common import ToolUserReportSeq
from climb.common.utils import filter_out_lines

BACKUP_OUTPUT_FILE = "progress.txt"


class ToolOutput:
    def __init__(self) -> None:
        self._tool_return: str = ""
        self._user_report_outputs: ToolUserReportSeq = []
        self.success: bool = True

        self.files_in: List[str] = []
        self.files_out: List[str] = []

    @property
    def tool_return(self) -> str:
        return self._tool_return

    @tool_return.setter
    def tool_return(self, value: str) -> None:
        self._tool_return = value

    @property
    def user_report_outputs(self) -> ToolUserReportSeq:
        return self._user_report_outputs

    @user_report_outputs.setter
    def user_report_outputs(self, value: ToolUserReportSeq) -> None:
        self._user_report_outputs = value

    def set_empty(self) -> None:
        self._tool_return = ""
        self._user_report_outputs = []

        self.files_in = []
        self.files_out = []


ToolReturnIter = Iterable[Union[str, ToolOutput]]


class ToolCommunicator:
    def __init__(self) -> None:
        self.comm_queue: queue.Queue = queue.Queue()
        self.exc_queue: queue.Queue = queue.Queue()
        self.std_queue: queue.Queue = queue.Queue()
        self.return_set = False

    def print(self, *args: Any) -> None:
        if self.return_set:
            raise ValueError("Cannot print after return value has been set")
        as_str = "".join([str(s) for s in args])
        self.comm_queue.put(f"{as_str}\n")

    def set_returns(
        self,
        tool_return: str,
        user_report: Optional[ToolUserReportSeq] = None,
        files_in: Optional[List[str]] = None,
        files_out: Optional[List[str]] = None,
    ) -> None:
        to = ToolOutput()

        # Output for LLM:
        to.tool_return = tool_return

        # Output for user:
        if user_report:
            to.user_report_outputs = user_report

        # Files in and out:
        if files_in:
            to.files_in = files_in
        if files_out:
            to.files_out = files_out

        self.comm_queue.put(to)
        self.return_set = True


class ToolException(Exception):
    pass


def except_hook(args: Any, exc_queue: queue.Queue) -> NoReturn:
    string_io = io.StringIO()
    # print(args.exc_type, args.exc_value)
    # print("args.exc_type is SystemExit", args.exc_type is SystemExit)
    # print("args.exc_value == 'ThreadWithTrace killed'", str(args.exc_value) == "ThreadWithTrace killed")
    if args.exc_type is SystemExit and str(args.exc_value) == "ThreadWithTrace killed":
        # Do not print the exception.
        # Only put it on the queue.
        exc_queue.put("ERROR: Tool terminated by system or user.")
        raise
    else:
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=string_io)
        exc_queue.put(string_io.getvalue())
        raise ToolException(f"\nException from thread: {args.thread}" f"\n{string_io.getvalue()}")


def process_stream_chunk(s: str) -> Optional[str]:
    if s.strip() == "":
        return None
    if not s.endswith("\n"):
        return s + "\n"
    return s


class StreamRedirector:
    def __init__(self, q: queue.Queue) -> None:
        self.q = q

    def write(self, text: str) -> None:
        text_ = process_stream_chunk(text)
        if text_ is not None:
            self.q.put(text_)

    def flush(self) -> None:
        pass


class ToolThread(threading.Thread):
    # Source for the trace approach: https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/

    SYSTEM_EXIT_MSG = "ThreadWithTrace killed"

    def __init__(self, *args: Any, **keywords: Any) -> None:
        threading.Thread.__init__(self, *args, **keywords)
        self.killed: bool = False

    def start(self, std_q: queue.Queue, exc_q: queue.Queue, backup_output_file_path: str) -> None:
        self.sys_stdout_bak = sys.stdout
        self.sys_stderr_bak = sys.stderr
        self.threading_excepthook_bak = copy.copy(threading.excepthook)
        self.backup_output_file_path = backup_output_file_path
        sys.stdout = StreamRedirector(std_q)  # type: ignore
        sys.stderr = StreamRedirector(std_q)  # type: ignore
        threading.excepthook = partial(except_hook, exc_queue=exc_q)

        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def _reset_key_resources(self) -> None:
        threading.excepthook = self.threading_excepthook_bak
        sys.stdout = self.sys_stdout_bak
        sys.stderr = self.sys_stderr_bak
        if os.path.exists(self.backup_output_file_path):
            os.remove(self.backup_output_file_path)

    def __run(self) -> None:
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame: Any, event: str, arg: Any) -> Optional[Callable]:
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame: Any, event: str, arg: Any) -> Optional[Callable]:
        if self.killed:
            if event == "line":
                raise SystemExit(self.SYSTEM_EXIT_MSG)
        return self.localtrace

    def join(self, timeout: Optional[int] = None) -> None:
        self._reset_key_resources()
        threading.Thread.join(self, timeout=timeout)

    def kill(self, timeout: Optional[int] = None) -> None:
        self.killed = True
        self.join(timeout=timeout)


def live_output_iterable(
    thread: ToolThread,
    comm_q: queue.Queue,
    exc_q: queue.Queue,
    std_q: queue.Queue,
    return_holder: ToolOutput,
    # stdout_bak: TextIO,
    # stderr_bak: TextIO,
    wd: str,
) -> ToolReturnIter:
    timeout = 1

    thread.start(std_q, exc_q, os.path.join(wd, BACKUP_OUTPUT_FILE))

    return_was_obtained = False

    backup_output_file_contents = ""

    while thread.is_alive() and not thread.killed:
        # 1. Follow the STDOUT/STDERR streams.
        try:
            output = std_q.get(block=False)
            thread.sys_stdout_bak.write(output)
            thread.sys_stdout_bak.flush()
            yield filter_out_lines(output)
        except queue.Empty:
            pass

        # 2. Follow the backup output stream file BACKUP_OUTPUT_FILE, and yield any new lines.
        # Why is this here? It may be impossible to capture the output of a tool's STDOUT/STDERR streams if it uses
        # multiprocessing, e.g. Autoprognosis does this. In such cases, we can use a backup file to capture the output.
        # The tool will have to be directed to print occasional output to this file, though. In AutoPrognosis, this is
        # achievable with the heartbeat hook.
        backup_file_path = os.path.join(wd, BACKUP_OUTPUT_FILE)
        if os.path.exists(backup_file_path):
            with open(backup_file_path, "r") as f:  # pylint: disable=unspecified-encoding
                backup_output_file_contents_current = f.read()
            if backup_output_file_contents_current != backup_output_file_contents:
                new_lines = backup_output_file_contents_current.replace(backup_output_file_contents, "")
                backup_output_file_contents = backup_output_file_contents_current
                yield filter_out_lines(new_lines)

        # 3. Follow the explicit communication queue.
        try:
            message = comm_q.get(timeout=timeout)
        except queue.Empty:
            if thread.is_alive():
                continue
            else:
                # 4. Follow the exception queue.
                # If an exception occurred in the thread, it should have been saved in the exception queue.
                if not exc_q.empty():
                    # If there is an exception in the queue, we should return its string representation.
                    exc = exc_q.get()
                    return_holder.success = False
                    return_holder.tool_return = exc
                    break
                else:
                    # If somehow exception was not caught, we return an empty output.
                    return_holder.set_empty()
                    break

        # 5. Handle output from the tool, if reached.
        if isinstance(message, ToolOutput):
            return_was_obtained = True
            return_holder.tool_return = message.tool_return
            return_holder.user_report_outputs = message.user_report_outputs
            return_holder.success = message.success
            return_holder.files_in = message.files_in
            return_holder.files_out = message.files_out
            break

        # Empty message edge case.
        if message is None:
            return_holder.set_empty()
            break

        yield filter_out_lines(message)

    # Join the thread if it is still alive.
    thread.join()

    # 1. Spit out any STDOUT/STDERR streams if there are such left
    if not thread.killed:
        while True:
            try:
                output = std_q.get(block=False)
                thread.sys_stdout_bak.write(output)
                thread.sys_stdout_bak.flush()
                yield filter_out_lines(output)
            except queue.Empty:
                break

    # 2. Attempt to catch the return value from the tool, if missed.
    #
    # In certain cases, likely due to a "race condition"-like scenario, we end up not capturing the return value in the
    # above loop. In order to "catch" the return value, we will try to get it from the queue here, by iterating over the
    # queue until we find it (or we get to the end without finding it).
    if not return_was_obtained:
        try:
            # Keep looping through the comm_q to see if we can find the return value.
            while True:
                message = comm_q.get(timeout=timeout)
                if isinstance(message, ToolOutput):
                    # Return value found, save it.
                    return_was_obtained = True
                    return_holder.tool_return = message.tool_return
                    return_holder.user_report_outputs = message.user_report_outputs
                    return_holder.success = message.success
                    return_holder.files_in = message.files_in
                    return_holder.files_out = message.files_out
                    break
        except queue.Empty:
            # If we get here, we did not find the return value.
            pass

    yield return_holder


def execute_tool(tool_func: Callable, wd: str, **kwargs: Any) -> Tuple[ToolThread, ToolReturnIter]:
    tc = ToolCommunicator()

    # Start message generation in a separate thread
    thread = ToolThread(target=tool_func, args=(tc,), kwargs=kwargs)

    return_holder = ToolOutput()
    return (
        thread,
        live_output_iterable(
            thread,
            tc.comm_queue,
            tc.exc_queue,
            tc.std_queue,
            return_holder=return_holder,
            wd=wd,
        ),
    )
