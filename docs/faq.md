# ❓ FAQ

* **Q:** I have a clinical dataset with some patient data. Can I use this data with CliMB?
    * **A:** This depends on the nature and requirements of your data. Reading the [🔏 Data Privacy](dataprivacy.md)
    section of this documentation should allow you to make an informed decision on this.

* **Q:** My dataset has so-and-so many variables and so-and-so many records (e.g. patients). Will a dataset of this size work with CliMB?
    * **A:** This depends on your hardware configuration and the details of your dataset(note: for minimal hardware requirements, please refer to the [📦 Installation](installation.md#system-requirements)).
        * If your dataset has more than a few 10s of features, the data exploration step may produce too much output, and use up the LLM context window. We recommend using datasets with no more than ~50 features. 
        * A larger number of records (rows of data) will lead to a longer execution time of various tools, especially the predictive modelling steps and the feature importance steps. For a modern workstation with a GPU, please use the following very rough guide for time estimate: *5 minutes per 1,000 records* for the predictive modelling step, and up to 5 times slower for the feature importance step.

* **Q:** I have problems when installing or running CliMB. Where can I get help?
    * **A:** Please submit a GitHub issue [here](https://github.com/vanderschaarlab/climb/issues), or ask us on [Slack](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1u2rmhw06-sHS5nQDMN3Ka2Zer6sAU6Q), `#climb` channel.
