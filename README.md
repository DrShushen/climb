[![Documentation Status](https://readthedocs.org/projects/climb-ai/badge/?version=latest)](https://climb-ai.readthedocs.io/en/latest/?badge=latest)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE.txt)
<!-- [![PyPI-Server](https://img.shields.io/pypi/v/climb-ai?color=blue)](https://pypi.org/project/climb-ai/) -->
<!-- [![Downloads](https://static.pepy.tech/badge/climb-ai)](https://pepy.tech/project/climb-ai) -->

[![arXiv](https://img.shields.io/badge/arXiv-2301.12260-b31b1b.svg)](http://arxiv.org/abs/2410.03736)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=76XuR0K3F5Y)


# <img src="docs/assets/climb-logo-no-text.png" height=25> CliMB

> **CliMB**: **Cli**nical **M**achine learning **B**uilder

This repository is the implementation of the system as described in the preprint [CliMB: An AI-enabled Partner for Clinical Predictive Modeling](http://arxiv.org/abs/2410.03736).

[<img src="docs/assets/play.svg" height=12> Watch the demo](https://www.youtube.com/watch?v=76XuR0K3F5Y)

[![Demo Video](docs/assets/video-demo.gif)](https://www.youtube.com/watch?v=76XuR0K3F5Y)



## 🏥 Overview
CliMB is an AI-enabled partner designed to empower clinician scientists to create predictive models from real-world clinical data, all within a single conversation. With its no-code, natural language interface, CliMB guides you through the entire data science pipeline, from data exploration and engineering to model building and interpretation. The intuitive interface combines an interactive chat with a dashboard that displays project progress, data transformations, and visualizations, making it easy to follow along. Leveraging state-of-the-art methods in AutoML, data-centric AI, and interpretability tools, CliMB offers a streamlined solution for developing robust, clinically relevant predictive models.

<img src="docs/assets/climb-fig-clinical.png" width=45% alt="CliMB Clinical Figure"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="docs/assets/climb-fig-main.png" width=45% alt="CliMB Conceptual Figure">

Our vision is for CliMB to integrate seamlessly into the clinician's workflow, supporting the complete cycle of clinical predictive modeling, and ultimately democratizing machine learning and AI utilization in healthcare.



## 🔏 Data Privacy
> [!WARNING]  
> It is crucial to understand the data privacy and confidentiality implications of using CliMB. Please ensure to read this section prior to using the system.

When using CliMB with real-world clinical data, you as the clinician scientist act as the data steward, and are responsible for ensuring that the use of the data complies with all relevant laws and regulations, as well as ethical considerations. CliMB aims to provide a secure and privacy-preserving environment for data exploration and model building, while balancing this with leveraging the capabilities of the most advanced large language models (LLMs). This section summarizes the data privacy fundamentals of CliMB and should allow you to make an informed decision about using the system with your data.

### CliMB's Privacy-Preserving Features
1. **Local storage of data.** All dataset files (original or modified in any way by CliMB) are
stored locally on your machine. Hence, the data files are never uploaded to any third-party servers.
2. **Local code execution.** All code execution performed by CliMB, either through code generation or predefined tool invocation occurs locally on the your machine. Hence, no working directory files of any kind (including saved predictive models, image files, tool output artifacts
etc.) leave the machine that you are using CliMB on.

### Privacy Implications of Using Third-party LLMs
CliMB currently supports the following third-party LLMs providers:
* [OpenAI](https://platform.openai.com/),
* [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview).

This allows for making use of more powerful LLMs (GPT-4 and beyond). Integration with locally-deployable LLMs (e.g., the [Hugging Face](https://huggingface.co/) ecosystem) is under development, but not yet available.

In order to use third-party, proprietary LLMs, CliMB uses their API ([What's an API?](https://www.contentful.com/api/)). This means that:
* The **message data** is transferred, encrypted, via the internet to the LLM provider's (cloud) servers, which then generate a response message.
* The **message data** may be stored by the LLM provider for some limited time (e.g., often 30 days) in order to detect and prevent abuse of the API.

> [!NOTE]
> **Message data** in CliMB is all the content you see in the chat interface, including the text you type, the text the system generates, and the output of code execution and tool invocations. This is also know as "prompts" and "completions", or the "context". This data *may* contain sensitive information, such as variable names, categorical values, and other data that you are working with in your predictive modeling project. It is unlikely to contain any patient records in full, as this is not required in the CliMB workflow, however this is **not guaranteed**.

It is critical that you understand the terms of service of the LLM provider you choose to use with CliMB. Below are links to the overviews of how each provider uses your data (but a detailed review of the terms of service is highly recommended):
* **OpenAI**:
     * [OpenAI Platform - How we use your data](https://platform.openai.com/docs/models/how-we-use-your-data)
     * [Privacy Policy](https://openai.com/policies/row-privacy-policy/)
* **Azure OpenAI Service**:
     * [Data, privacy, and security for Azure OpenAI Service](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy?tabs=azure-portal)
     * [Privacy in Azure](https://azure.microsoft.com/en-gb/explore/trusted-cloud/privacy)

For instance, in case of **Azure OpenAI Service**, the following applies:
> Your prompts (inputs) and completions (outputs), your embeddings, and your training data:
> * are NOT available to other customers.
> * are NOT available to OpenAI.
> * are NOT used to improve OpenAI models.
> * are NOT used to train, retrain, or improve Azure OpenAI Service foundation models.
> * are NOT used to improve any Microsoft or 3rd party products or services without your permission or instruction.
> Your fine-tuned Azure OpenAI models are available exclusively for your use.

However, the following points regarding data storage and human review for purposes of abuse prevention, and the process of obtaining an exemption should also be read and understood:
* [Data storage for Azure OpenAI Service features](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy?tabs=azure-portal#data-storage-for-azure-openai-service-features)
* [Preventing abuse and harmful content generation](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy?tabs=azure-portal#preventing-abuse-and-harmful-content-generation)
* [How can customers get an exemption from abuse monitoring and human review?](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy?tabs=azure-portal#how-can-customers-get-an-exemption-from-abuse-monitoring-and-human-review)

If using **OpenAI** as the LLM provider, the corresponding terms of service should be reviewed in detail.

A useful additional resource for understanding the privacy implications of specific LLM providers is PhysioNet's [Responsible use of MIMIC data with online services like GPT](https://physionet.org/news/post/gpt-responsible-use). PhysioNet is the provider of the MIMIC datasets, a set of widely-used open access datasets in clinical research.

> [!TIP]
> Data [anonymization and pseudonymization](https://www.ucl.ac.uk/data-protection/guidance-staff-students-and-researchers/practical-data-protection-guidance-notices/anonymisation-and) are important techniques for maintaining compatibility with GDPR and similar regulations, and these are generally advised when using CliMB with clinical data.



## 📦 Installation
In order to use CliMB, you need to accomplish the following three steps:
1. [🐍 Set up the `conda` environments](#1--set-up-the-conda-environments)
2. [🔑 Obtain the API keys for the third-party LLM](#2--obtain-the-api-keys-for-the-third-party-llm) ([OpenAI](https://platform.openai.com/) or[Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview))
3. [📈 Install the CliMB package](#3--install-the-climb-package)

### 1. 🐍 Set up the `conda` environments
CliMB uses [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the Python environments. Before installing CliMB, you need to set up two `conda` environments as follows.
1. If you do not have `conda` installed on your system, follow [these instructions](https://docs.anaconda.com/free/miniconda/) to install `miniconda`.
2. Create the *main* conda environment for CliMB:
     ```bash
     conda create -n climb python=3.9 -y
     ```
     Python `3.9` or newer should be set.
3. Create a *separate* `conda` environment that will be used for *code execution*:

     > [!CAUTION]
     > The exact environment name `climb-code` must be used.

     ```bash
     # Create and activate the environment, Python `3.9` or newer should be set:
     conda create -n climb-code python=3.9 -y

     # Activate the environment:
     conda activate climb-code
     # Install some standard packages in the environment. If more packages are needed by generated code, those will be automatically installed by the tool.
     conda install pandas numpy matplotlib seaborn scikit-learn shap -y
     # Exit this environment:
     conda deactivate
     ```

     CliMB will automatically use this environment when executing the generated code.

### 2. 🔑 Obtain the API keys for the third-party LLM
> [!WARNING]  
> Please read the [🔏 Data Privacy](#-data-privacy) section before proceeding with this step, in order to make an informed decision about which LLM provider is suitable for your use case.

#### Option 1: OpenAI
1. Sign up for OpenAI platform [here](https://platform.openai.com/signup).
2. Fund your account by following [this guide](https://help.openai.com/en/articles/8264644-what-is-prepaid-billing).
3. Follow [this guide](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key) to get your API key.
     * ⚠️ Never share your API key with anyone and treat it as a "password". A reminder to developers to to never commit your API keys to a public repository!
     * Make note of this **key** as it is needed later.

#### Option 2: Azure OpenAI Service
1. Create an Azure account [here](https://azure.microsoft.com/en-gb/pricing/purchase-options/azure-account?icid=azurefreeaccount).
2. Create an Azure OpenAI Service resource by following [this guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource).
     * At the final **"Deploy a model"** step, we recommend selecting `gpt-4` (one of the versions `1106-preview`, `0125-preview`, or `turbo-<DATE>`), or `gpt-4o` (any version).
     * When you are deploying the model, make note of the **① deployment name** that you use as it is needed later.
3. In [Azure OpenAI Studio](https://oai.azure.com/), click the resource name at the top right of the screen to find: **② endpoint** and **③ key**, make note of these as they are needed later.
     
     <img src="docs/assets/installation-az-info.png" height=450 alt="CliMB Clinical Figure">



### 3. 📈 Install the CliMB package
1. Clone the CliMB repository and navigate to the directory (we will call this the **repo directory**)
    ```bash
    # Clone the repository:
    git clone <get the URL from github>

    # Navigate inside the repo directory:
    cd climb
    ```
2. Activate the *main* `conda` environment and install the package itself (this will install all the dependencies as well):
    ```bash
    # Activate the main environment:
    conda activate climb

    # Install the CliMB package:
    pip install -e .
    ```
3. Finally, you need to set up the configuration file for the LLM provider you chose.
     * Copy the [Example `.env`](./config/.env) file to the **repo directory**.
     On Windows you may wish to rename it to `keys.env` to avoid the file being hidden / extension confusion.
     * **Option 1: OpenAI**:
          * Open the `.env`/`keys.env` file in the **repo directory** and replace the value of
               ```ini
               OPENAI_API_KEY="API_KEY_FOR_OPENAI"
               ```
               with the **key** you obtained.
     * **Option 2: Azure OpenAI Service**:
          * Open the `.env`/`keys.env` file in the **repo directory**.
               ```ini
               AZURE_OPENAI_API_KEY__my-endpoint-1="API_KEY_FOR_AZURE_ENDPOINT_1"
               ```
               * Update the value `"API_KEY_FOR_AZURE_ENDPOINT_1"` with the **③ key** you obtained.
               * Replace `my-endpoint-1` template with the ID of the endpoint you are actually using. For example, if your endpoint is `https://my-clinic.openai.azure.com/`, use the `my-clinic` part. In this example case, the line would look like:
                    ```ini
                    AZURE_OPENAI_API_KEY__my-clinic="your actual ③ key"
                    ```
          * Copy the [Example `az_openai_config.yml`](./config/az_openai_config.yml) file to the **repo directory**.
          * Open the `az_openai_config.yml` file in the **repo directory**:
               ```yaml
               models:
                 - name: "your-custom-name"
                   # ^ This is to identify the model in the UI, it can be anything.
                   endpoint: "https://my-endpoint-1.openai.azure.com/"
                   # ^ The endpoint of azure openai service you are using.
                   deployment_name: "your-deployment-name"
                   # ^ The deployment name of the model you are using.
                   api_version: "2024-02-01"
                   # ^ The api version, see https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs
                   model: "gpt-4-0125-preview"
                   # ^ This needs to match the model type you set up in the Azure OpenAI Portal.
                   # Currently the options are:
                   #  - "gpt-4-1106-preview"
                   #  - "gpt-4-0125-preview"
                   #  - "gpt-4o"
               ```
               * You need to set the value of `endpoint` to **② endpoint** and `deployment_name` to **① deployment name**.
               * Make sure th `model` field matches the model type you deployed in the Azure OpenAI Portal.
               * Make sure the `api_version` field matches one of the [versions available](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs), it is best practice to use the latest version available.
               * The `name` field can be anything you want, it is used to identify the model in the UI.




## 🚀 Usage
To launch CliMB UI, from the **repo directory**, run:
```bash
streamlit run entry/st/app.py
```

This will show the output like:
```txt
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.68:8501
```

Navigate to the `Local URL` in your browser ([Chrome](https://www.google.com/intl/en_uk/chrome/) is recommended for UI compatibility reasons) to start using CliMB.

The **working directory** of CliMB will be:
```bash
<repo_directory>/wd/
```

CliMB will put all the various data and model files there. Each research session will have its own subdirectory inside the `wd` directory. If you delete a session, the corresponding subdirectory will be deleted as well.



## ✍️ Citing

If you use CliMB in your work, please cite the associated paper:
```bibtex
@article{saveliev2024climb,
  title={CliMB: An AI-enabled Partner for Clinical Predictive Modeling},
  author={Saveliev, Evgeny and Schubert, Tim and Pouplin, Thomas and Kosmoliaptsis, Vasilis and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2410.03736},
  year={2024}
}
```
