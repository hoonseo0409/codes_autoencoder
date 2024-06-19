# Semi-Supervised Attention Autoencoder
This repository contains the implementation of the Semi-Supervised Attention Autoencoder (SAE), designed to enrich time-series or multi-instance data into new representations. SAE learns the attention relationships between instances and extracts deep features in a lower-dimensional space. This repository includes code to process datasets, conduct enrichment and prediction, and visualize and save the enrichment results. Currently, six datasets are supported for enrichment learning: [Chest X-ray dataset](https://github.com/ieee8023/covid-chestxray-dataset), [COVID-19](https://github.com/HAIRLAB/Pre_Surv_COVID_19/tree/master/data), [Physionet Challenge](https://physionet.org/content/challenge-2012/1.0.0/), [Alzheimer's Dataset](https://adni.loni.usc.edu), [Texas Austin Traffic Dataset](https://data.mobility.austin.gov), and [Colorado Traffic Dataset](https://dtdapps.coloradodot.info/otis/TrafficData).

## Project Description
* `main.py`: Entry point of this project. This file imports the enrichment and prediction models and runs the experiments. It loads the dataset and conducts a grid search to tune the hyperparameters of the enrichment learning model. The Dataset object is passed into the Experiment object to generate the experimental results.
* `data_prep.py`: Code to preprocess the raw dataset. The processed dataset is formatted in the Dataset object with a consistent format, readily available for the subsequent experiments.
* `group_finder.py`: Code to find the groups of snips. This code is used only in data_prep.py.
* `estimator.py`: Implementations of classification models used after enrichment.
* `autoencoder.py`: Implementation of the semi-supervised autoencoder for dynamic data.
* `utils.py`: Code to conduct the experiments. The Experiment object takes (1) a list of Estimators and (2) a Dataset object. Then, the Experiment object conducts the prediction task on the Dataset object using the given Estimators. It also contains utility and visualization codes to plot the important features in enrichment.

## Getting Started
### Requirements
* Python 3.8
* Python packages listed in requirements.txt

### Installation and Run
To set up the project, start by cloning the repository to your local machine. 
* Install Python 3.8 (https://www.python.org).
* It is highly recommended to create an isolated Python environment via conda. For example:
```
conda create -n "SAE" python=3.8
```
* Then, activate the created environment:
```
conda activate "SAE"
```
* The installation also requires installing the custom Python library [utilities-python](https://github.com/hoonseo0409/utilities-python). `utilities-python` repository hosts the utility and visualization codes used across many projects, including this one.
* Install the required packages for this project. The packages are listed under `./requirements.txt`.
    * The public Python packages can be simply installed via `pip install requirements.py` which will install all the packages listed under `./requirements.txt`.
    * [utilities-python](https://github.com/hoonseo0409/utilities-python) is a custom package and can be installed via `pip install -e "path/to/utilities-python"`. Here, "path/to/utilities-python" is the local path to cloned [utilities-python](https://github.com/hoonseo0409/utilities-python).
* If everything is set up correctly, you will be able to run main.py:
```
python3 main.py
```
A folder will be created under the `./outputs`, containing all the experimental results.

### Trouble shooting for m1 silicon mac
* Install brew, miniforge (ref: https://towardsdatascience.com/how-to-easily-set-up-python-on-any-m1-mac-5ea885b73fab)
* install Python 3.8.13 with conda, and create virtual env.
* install packages under requirements.txt
* install tensorflow-macos (ref: https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/)
* If `pip install` not works, try `conda install`.

## Data
The datasets are publicly available at the following links:
* [Chest X-ray dataset](https://github.com/ieee8023/covid-chestxray-dataset): This is a public open dataset of chest X-ray and CT images of patients who are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS). Data is collected from public sources as well as through indirect collection from hospitals and physicians. 
* [COVID-19](https://github.com/HAIRLAB/Pre_Surv_COVID_19/tree/master/data): This dataset contains the medical records collected from 375 COVID-19 patients.
* [Physionet Challenge](https://physionet.org/content/challenge-2012/1.0.0/): The data used for the challenge consists of records from 12,000 ICU stays. All patients were adults admitted for a wide variety of reasons to cardiac, medical, surgical, and trauma ICUs. ICU stays of less than 48 hours have been excluded.
* [Alzheimer's Dataset](https://adni.loni.usc.edu): It consists of neuroimaging and genetic information of participants, some of whom have Alzheimer's disease.
* [Texas Austin Traffic Dataset](https://data.mobility.austin.gov): City and municipality governments deploy and collect data from a variety of traffic infrastructures. Traffic sensing infrastructures commonly include devices to measure speed and traffic flow or to capture images. This dataset consists of speed measurements and traffic camera images from the Austin Department of Transportation.
* [Colorado Traffic Dataset](https://dtdapps.coloradodot.info/otis/TrafficData): This dataset is collected by the Colorado Department of Transportation (CDOT) from January 2021 to January 2023 and consists of hourly traffic counts.

## Contributing
We welcome contributions to improve the framework and extend its functionalities. Please feel free to fork the repository, make your changes, and submit a pull request.