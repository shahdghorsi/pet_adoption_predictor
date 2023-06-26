from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, normalize
import pandas as pd
import numpy as np

def preprocess_csv(data_frame):
    '''
    Preprocesses the given DataFrame by performing various data transformations.
    Args:
        data_frame (pd.DataFrame): The input DataFrame to preprocess.
    Returns:
        tuple: A tuple containing the features, target, and preprocessed DataFrame.
    '''
    # Replace "Not Sure" values with "None" in the 'Vaccinated' column
    data_frame['Vaccinated'].replace('Not Sure', 'None', inplace=True)
    data_frame['Sterilized'].replace('Not Sure', 'None', inplace=True)
    # Drop rows with missing values and the string "None" in any column
    data_frame = data_frame[~data_frame.isin(['None']).any(axis=1)]
    # Drop the Color2, mostly missing and it doesn't play a crucial role in the prediction
    data_frame.drop('Color2', axis=1, inplace=True)

    label_encoder = LabelEncoder()
    data_frame["Adopted"] = label_encoder.fit_transform(data_frame["Adopted"])

    # Step 3: Encode categorical data
    categorical_features = ["Type", "Breed1", "Gender", "Color1", "MaturitySize", "FurLength",
                            "Vaccinated", "Sterilized", "Health"]
    encoded_df = pd.get_dummies(data_frame, columns=categorical_features)
    encoded_df = encoded_df.astype(int)  # Convert encoded features to int

    # Remove outliers and normalize Age and PhotoAmt features
    scaler = MinMaxScaler()
    threshold = 3  # Set the threshold for outliers (e.g., 3 standard deviations)

    feature_age_ph = encoded_df[['Age', 'PhotoAmt']]
    mean_age_ph = feature_age_ph.mean()
    std_age_ph = feature_age_ph.std()
    outliers_age_ph = feature_age_ph[(feature_age_ph - mean_age_ph) > threshold * std_age_ph]
    encoded_df = encoded_df[~encoded_df.isin(outliers_age_ph)].dropna()

    normalized_age_ph = scaler.fit_transform(encoded_df[['Age', 'PhotoAmt']])
    encoded_df[['Age', 'PhotoAmt']] = normalized_age_ph


    # Step 4: Split the dataset into train, validation, and test sets
    features = encoded_df.columns.drop("Adopted")
    target = encoded_df["Adopted"]
    return features, target, encoded_df