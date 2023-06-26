# Pet Adoption Prediction

This project trains an XGBoost model to predict whether a pet is going to be adopted or not.

## Table of Contents

- [Introduction](#introduction)
- [Structure](#structure)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)

## Introduction

The main objective of this project is to develop a machine-learning model that can predict the likelihood of a pet being adopted. The model is trained using the XGBoost algorithm, a popular gradient-boosting framework.
I tried different preprocessing approaches and compared the different evaluation metrics required and chose the best possible trained model to be used.


## Structure

List of artifacts included in the project:

- `data_processing.py`: Python script for data processing, including cleaning, preprocessing, and feature engineering.
- `MLOpsTechTest.md`: Markdown document with the MLOps technical test instructions or description.
- `model_training.ipynb`: Jupyter Notebook containing the code for model training and EDA.
- `output`: Folder where the output CSV is stored.
- `artifactory`: Folder where the output model is stored.
- `predictor_test.py`: File containing the unit test code to test the predictor.
- `predictor.py`: File containing the code to load the model, dataset, and perform prediction.
- `run_predictor.py`: File containing the code to run the predictor.

## Dataset

The dataset used for training and evaluation can be accessed at [dataset_url](gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv). Please download the dataset and place it in the appropriate location as specified in the code or documentation.

## Evaluation Metrics

The performance of the model has been evaluated using the following metrics:

- F1 Score: 0.8310249307479225
- Accuracy: 0.7436974789915967
- Recall: 0.847457627118644

The model has been evaluated using various data preprocessing techniques and feature transformations. Here are the evaluation metrics for different scenarios:

1. **Downsampling**:
   - F1 Score: 0.7745173745173746
   - Accuracy: 0.6932773109243697
   - Recall: 0.7083333333333334

2. **Normalized Age**:
   - F1 Score: 0.8345575376005596
   - Accuracy: 0.7515756302521008
   - Recall: 0.8425141242937854

3. **Normalized Age and PhotoAmt**:
   - F1 Score: 0.8386873920552678
   - Accuracy: 0.7547268907563025
   - Recall: 0.8573446327683616

4. **Dropped Fee**:
   - F1 Score: 0.834257975034674
   - Accuracy: 0.7489495798319328
   - Recall: 0.8495762711864406

5. **Normalized Fee**:
   - F1 Score: 0.8326983027364047
   - Accuracy: 0.7463235294117647
   - Recall: 0.8488700564971752

6. **Removed Age Outliers and Normalized**:
   - F1 Score: 0.8434752022511431
   - Accuracy: 0.7601078167115903
   - Recall: 0.8750918442321822
