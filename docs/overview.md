<!-- CliMB README.md -->



# <img src="assets/climb-logo-no-text.png" height=25> CliMB

> **CliMB**: **Cli**nical **M**achine learning **B**uilder

This repository is the implementation of the system as described in the preprint [CliMB: An AI-enabled Partner for Clinical Predictive Modeling](http://arxiv.org/abs/2410.03736).

[<img src="assets/play.svg" height=12> Watch the demo](https://www.youtube.com/watch?v=76XuR0K3F5Y)

[![Demo Video](assets/video-demo.gif)](https://www.youtube.com/watch?v=76XuR0K3F5Y)



## 🏥 Overview
CliMB is an AI-enabled partner designed to empower clinician scientists to create predictive models from real-world clinical data, all within a single conversation. With its no-code, natural language interface, CliMB guides you through the entire data science pipeline, from data exploration and engineering to model building and interpretation. The intuitive interface combines an interactive chat with a dashboard that displays project progress, data transformations, and visualizations, making it easy to follow along. Leveraging state-of-the-art methods in AutoML, data-centric AI, and interpretability tools, CliMB offers a streamlined solution for developing robust, clinically relevant predictive models.

<img src="assets/climb-fig-clinical.png" width=45% alt="CliMB Clinical Figure"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="assets/climb-fig-main.png" width=45% alt="CliMB Conceptual Figure">

Our vision is for CliMB to integrate seamlessly into the clinician's workflow, supporting the complete cycle of clinical predictive modeling, and ultimately democratizing machine learning and AI utilization in healthcare.


```{admonition} Target audience
:class: note

The intended target audience of CliMB is a clinician scientists with some basic Python and AI knowledge.
If you do not consider yourself "tech-savvy", the installation and configuration steps in particular may require assistance from your IT department.
```




## 🔏 Data Privacy

```{admonition} Warning
:class: attention

It is crucial to understand the data privacy and confidentiality implications of using CliMB. Please ensure to read and understand the [Data Privacy documentation section](dataprivacy.md) in full prior to installing and using the system.
```


When using CliMB with real-world clinical data, you as the clinician scientist act as the data steward, and are responsible for ensuring that the use of the data complies with all the relevant laws and regulations, as well as ethical considerations. CliMB aims to provide a secure and privacy-preserving environment for data exploration and model building, while balancing this with leveraging the capabilities of the most advanced large language models (LLMs).

We provide a detailed section in the documentation, which summarizes the data privacy fundamentals of CliMB and should allow you to make an informed decision about using the system with your data. It is essential that you read and understand this prior to using CliMB, please find the link below:


#### [📕 **Must-read:** Data Privacy documentation](dataprivacy.md)




## 📦 Installation


```{admonition} Warning
:class: attention

Please read the [🔏 Data Privacy](dataprivacy.md) section before proceeding with this step, in order to understand whether CliMB is compatible with your data and use case.
```

Please follow the steps in [📦 Installation](installation.md) section in the documentation to install CliMB.

To update to the latest version of CliMB, please follow [📦⬆️ Updating CliMB](installation.md#updating-climb)




## 🚀 Usage
First, navigate to the the CliMB **repo directory** in the terminal.

```{admonition} Repo directory
:class: tip

The location of the **repo directory** is explained in the [📈 Install the CliMB package](installation.md#install-the-climb-package) section of the documentation. Don't forget to run `cd climb` to change to the repo directory.
```


To launch CliMB UI, run the command:
```bash
streamlit run entry/st/app.py
```

This will show the output like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.68:8501
```


The best way to get started with CliMB is to follow the [**🚀 Quickstart Guide**](quickstart.md) in the documentation.

If you encounter errors or problems when running CliMB for the first time, please check out the [🛠️ Troubleshooting](troubleshooting.md) section, as it has the resoulutions steps for some common intallation and set up problems. For any other problems, please submit a GitHub issue [here](https://github.com/vanderschaarlab/climb/issues), or ask us on [Slack](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1u2rmhw06-sHS5nQDMN3Ka2Zer6sAU6Q).







## 📝 Disclaimer

By accessing and using this software, you acknowledge and agree that you do so at your own risk. The copyright holders and contributors of this software and its associated web-based tools disclaim any liability for inaccuracies, errors, or omissions in the analyses generated or for any actions taken based on these analyses. 

This software is provided on an "as-is" basis without any warranties, expressed or implied, including but not limited to accuracy, fitness for a particular purpose, or compatibility with regulatory requirements. The copyright holders and contributors assume no responsibility for any clinical or non-clinical outcomes that may arise from use of this software or its results.

**Data Privacy and Confidentiality:**  
You are solely responsible for ensuring that any data entered into this system complies with relevant confidentiality and data privacy regulations, including HIPAA, GDPR, or any other applicable standards. As this software utilizes third-party, proprietary large language model (LLM) APIs, the copyright holders and contributors are not responsible for data security or regulatory compliance in relation to the use of these external APIs. It is the user's responsibility to anonymize data as required and to ensure that data-sharing practices align with the applicable privacy laws and institutional policies.

By proceeding to use this software, you agree to these terms and accept full responsibility for your use and management of any data within this system.



## ✍️ Citing

If you use CliMB in your work, please cite the [associated paper](http://arxiv.org/abs/2410.03736):
```bibtex
@article{saveliev2024climb,
  title={CliMB: An AI-enabled Partner for Clinical Predictive Modeling},
  author={Saveliev, Evgeny and Schubert, Tim and Pouplin, Thomas and Kosmoliaptsis, Vasilis and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2410.03736},
  year={2024}
}
```
