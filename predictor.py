import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from data_processing import preprocess_csv

def load_data(data_url: str) -> pd.DataFrame:
    '''
    Load the data from a CSV file located at the given URL.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    '''
    df = pd.read_csv(data_url)
    return df

def load_model(model_path: str) -> xgb.Booster:
    '''
    Load the XGBoost model from the specified path.
    Returns:
        xgb.Booster: The loaded XGBoost model.
    '''
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def preprocess_data(df: pd.DataFrame) -> tuple:
    '''
    Preprocess the data by performing encoding and feature selection.
    Args:
        df (pd.DataFrame): The input data as a pandas DataFrame.
    Returns:
        tuple: A tuple containing the target variable, selected features, and the encoded DataFrame.
    '''
    target, features, encoded_df = preprocess_csv(df)
    return target, features, encoded_df

def predict(model: xgb.Booster, encoded_df: pd.DataFrame) -> list:
    '''
    Make predictions using the trained XGBoost model.
    Args:
        model (xgb.Booster): The trained XGBoost model.
        encoded_df (pd.DataFrame): The encoded DataFrame used for prediction.
    Returns:
        list: The predicted values.
    '''
    encoded_df.drop('Adopted', axis=1, inplace=True)
    dtest = xgb.DMatrix(encoded_df)
    predictions = model.predict(dtest)
    return predictions

def map_predictions(predictions: list) -> list:
    '''
    Map the predicted values to the corresponding labels.
    Args:
        predictions (list): The predicted values.
    Returns:
        list: The mapped predictions as "Yes" or "No".
    '''
    mapped_predictions = ["Yes" if x >= 0.5 else "No" for x in predictions]
    return mapped_predictions

def merge_data(df: pd.DataFrame, predictions: list) -> pd.DataFrame:
    '''
    Merge the original DataFrame with the predicted values.
    Args:
        df (pd.DataFrame): The original DataFrame.
        predictions (list): The predicted values.
    Returns:
        pd.DataFrame: The merged DataFrame.
    '''
    prediction_df = pd.DataFrame(predictions, columns=["Adopted_prediction"])
    mapped_predictions = map_predictions(predictions)
    prediction_df["Adopted_prediction"] = mapped_predictions[:len(df)]  # Take subset of mapped predictions
    output_df = pd.concat([df.reset_index(drop=True), prediction_df], axis=1)
    return output_df

def save_output(output_df: pd.DataFrame, output_path: str) -> None:
    '''
    Save the DataFrame as a CSV file at the specified output path.
    Args:
        output_df (pd.DataFrame): The DataFrame to be saved.
        output_path (str): The output path for saving the CSV file.
    Returns:
        None
    '''
    output_df.to_csv(output_path, index=False)
