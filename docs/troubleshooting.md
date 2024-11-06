# 🛠️ Troubleshooting

This section explains common troubleshooting steps.

## Windows: `OSError: cannot load library 'pangoft2...'`
There is a known issue with the `pango` dependency (that is used by the [WeasyPrint](https://github.com/Kozea/WeasyPrint) dependency we currently use for creating a PDF report):
* https://github.com/conda-forge/weasyprint-feedstock/issues/31
* https://github.com/conda-forge/pango-feedstock/issues/77

which can cause an error like this:
```
OSError: cannot load library 'pangoft2-1.0-0': error 0x7e. Additionally, ctypes.util.find_library() did not manage to locate a library called 'pangoft2-1.0-0'
```
when launching CliMB UI on Windows systems.

To fix this error, you can follow the following steps:
1. Locate your `climb` environment's directory. You can find it by e.g. running `conda env list`. Then, locate the binary libraries directory - e.g. if your environment is at `C:\Users\<username>\miniconda3\envs\climb`, the binary libraries directory will be `C:\Users\<username>\miniconda3\envs\climb\Libraries\bin`.
2. Download the missing `pangoft2-1.0` DLL file from [DLLme](https://www.dllme.com/dll/files/pangoft2-1_0-0/versions?sort=version&arch=0x8664).
    * It is best practice to do a security scan on this file, e.g. with Windows Defender.
3. Place this downloaded `pangoft2-1.0-0.dll` file into your CliMB environment's binary libraries directory (typically `C:\Users\<username>\miniconda3\envs\climb\Libraries\bin`).

## `conda` not found (`OSError: file not found`)
If whenever CliMB tries to execute a tool or generate code, you see an error similar to `OSError: file not found`, this most likely indicates that it cannot find the location of your `conda` executable. if you know that `conda` is installed (see [📦 Installation](installation.md#set-up-the-conda-environments)) on your system and this issue is happening, follow the below steps to resolve this.

1. Determine the path to your `conda` installation. You can use a command like: `which conda` (Linux, OSX) or
`where conda` (Windows, in Anaconda PowerShell Prompt). Make note of the full conda path shown.
2. Set the full conda path in your `.env`/`keys.env` file. Add a line like so:
    ```bash
    CONDA_PATH="/path/to/your/conda"
    ```
    Replace `/path/to/your/conda` with the path you found in step 1.
3. Restart the tool and try running the generated code again.
