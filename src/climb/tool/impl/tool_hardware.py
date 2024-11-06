import time
from typing import Any, Dict

import psutil

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase, get_str_up_to_marker

try:
    import GPUtil

    gpus_available = True
except ImportError:
    gpus_available = False

import torch


def get_cpu_info() -> Dict[str, Any]:
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "max_frequency": psutil.cpu_freq().max,
    }
    return cpu_info


def get_ram_info() -> Dict[str, int]:
    ram_info = {
        "total_memory": psutil.virtual_memory().total,
        "available_memory": psutil.virtual_memory().available,
    }
    return ram_info


def get_gpu_info() -> Any:
    if not gpus_available:
        return "GPUtil library not available. GPU info cannot be retrieved."
    gpus = GPUtil.getGPUs()
    gpu_info = [{"name": gpu.name, "total_memory": gpu.memoryTotal} for gpu in gpus]
    return gpu_info


def generate_report() -> str:
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    gpu_info = get_gpu_info()

    report = (
        f"CPU Information:\n- Physical Cores: {cpu_info['physical_cores']}\n"
        f"- Total Cores: {cpu_info['total_cores']}\n- Max Frequency: {cpu_info['max_frequency']} MHz\n\n"
        f"RAM Information:\n- Total Memory: {ram_info['total_memory'] / (1024 ** 3):.2f} GB\n"
        f"- Available Memory: {ram_info['available_memory'] / (1024 ** 3):.2f} GB\n\n"
        "GPU Information:\n"
    )

    if isinstance(gpu_info, str):
        report += gpu_info
    else:
        for idx, gpu in enumerate(gpu_info, start=1):
            report += f"- GPU {idx}: {gpu['name']} with {gpu['total_memory']}MB of memory\n"

    torch_cuda_is_available = torch.cuda.is_available()
    torch_cuda_device_count = torch.cuda.device_count()
    report += (
        f"\nPyTorch CUDA Information:\n- CUDA is available: {torch_cuda_is_available}\n"
        f"- Number of CUDA devices: {torch_cuda_device_count}\n"
    )

    return report


def check_user_hardware(tc: ToolCommunicator) -> None:
    """Gather information about the user's CPU, RAM, and GPU (if available).

    The report will be as follows:
    ```
    CPU Information:
    - Physical Cores: <value>
    - Total Cores: <value>
    - Max Frequency: <value> MHz

    RAM Information:
    - Total Memory: <value> GB
    - Available Memory: <value> GB

    GPU Information:
    - GPU 1: <model> with <value>MB of memory

    PyTorch CUDA Information:
    - CUDA is available: <True/False>
    - Number of CUDA devices: <value>
    ```

    Args:
        tc (ToolCommunicator): tool communicator object.
    """
    analysis_summary = ""

    # Dataset basic info
    tc.print("Gathering information about your hardware...")
    time.sleep(0.4)
    # For DEBUG:
    # time.sleep(2)
    # tc.print("This may take a few seconds...")
    # time.sleep(15)  # For testing purposes.

    analysis_summary += generate_report()

    tc.set_returns(analysis_summary)


class HardwareInfo(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        thrd, out_stream = execute_tool(check_user_hardware, wd=self.working_directory)
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "hardware_info"

    @property
    def description(self) -> str:
        return get_str_up_to_marker(check_user_hardware.__doc__, "Args")  # type: ignore

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "gather information about your hardware (CPU, RAM, and GPU)."
