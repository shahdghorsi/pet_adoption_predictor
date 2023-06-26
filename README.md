# Pet adoption prediction

This project trains an Xboost model to predict whether a pet is going to be adopted or not

## Table of Contents
- `data_processing.py`: File containing the code for data processing.
- `MLOpsTechTest.md`: Markdown document with the MLOps technical test instructions or description.
- `model_training.ipynb`: Jupyter Notebook containing the code for model training and EDA.
- `output`: Folder where the output CSV is  stored.
- `artifactory`: Folder where the output model is  stored.
- `predictor_test.py`: File containing the unit test code to test the predictor.
- `predictor.py`: File containing the code to load the model, dataset, and perform prediction.
- `run_predictor.py`: File containing the code to run the predictor.

- [Usage](#usage)

## Usage
Run the model_training.ipynb notebook to train the model, explore the data, and eventually save the trained model in the artifactory folder.
To get the predictions using the trained model, run the predictor using the run_predictor.py file as a result you should see the output result CSV in the output folder.
To test the process of loading the model, dataset, and prediction you can run the predictor_test.py.
