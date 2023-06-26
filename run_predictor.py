import predictor

data_url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
model_path = "artifacts/model_final.xgb"
output_path = "output/output.csv"

# Load data
data = predictor.load_data(data_url)

# Preprocess data
target, features, encoded_data = predictor.preprocess_data(data)

# Load model
model = predictor.load_model(model_path)

# Perform predictions
predictions = predictor.predict(model, encoded_data)

# Merge predictions with original data
output_df = predictor.merge_data(data, predictions)

# Save output
predictor.save_output(output_df, output_path)
