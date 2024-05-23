## Requirements
* pipenv: To create Python virtual environment.
* Python 3.7
* The required Python packages are listed under "./Pipfile".

## Installation and Run
* Install Python 3.7.3 (https://www.python.org) (For macos user, I recommend to install Python through brew).
* Install pipenv (https://pipenv.pypa.io/en/latest/) (For macos user, again I recommend to install pipenv through brew). Pipenv is environment manager, which will create independent Python dedicated to the specific project and manage the package dependencies for that project (similar to virtualenv).
* Create the pipenv virtual environment through 'pipenv shell'.
* Pull (1) this (predict-mortality) and (2) utilsforminds repositories. 'utilsforminds' repository hosts the utility and visualization codes used across many projects including this project.
* Install the required packages for this project. The packages are listed under ./Pipfile. When you install packages, don't forget to create and activate (both can be done by 'pipenv shell') your local pipenv virtual environment by checking whether 'pipenv --py' points to the correct Python path.
    * The public Python packages can be simply installed via 'pipenv install' which will install all the packages listed under ./Pipfile.
    * utilsforminds is a custom package and should be installed via 'pipenv install -e "path-to-utilsforminds"'.
* (Optional) You may need to install programming editor to work conveniently. I recommend to use vscode (https://code.visualstudio.com).
    * After installing vscode, you can link the pipenv Python to your project directory. At the left down corner, click the Python and set the path to pipenv local Python executable.
* If everythings are setup correctly, you will be able to run main.py.

## Trouble shooting for m1 silicon mac
* Install brew, miniforge (ref: https://towardsdatascience.com/how-to-easily-set-up-python-on-any-m1-mac-5ea885b73fab)
* install Python 3.8.13 with conda, and create virtual env.
* install packages under requirements.txt
* install tensorflow-macos (ref: https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/)
* If 'pip install' not works, try 'conda install'.

## Project Structure
main.py : Entry point of this project. This file imports the prediction models and run the experiments.
data_prep.py : Code to preprocess the raw dataset. The processed dataset is formatted in the Dataset object.
group_finder.py : Code to find the groups of snips. This code is used only in data_prep.py.
estimator.py : Implementations of baseline prediction models.
autoencoder.py : Implementation of semi-supervised autoencoder for dynamic data.
utils.py : Code to conduct the experiments. The Experiment object will take (1) list of Estimators (2) Dataset object. Then Experiment object will conduct the prediction task on Dataset object using Estimators given.

