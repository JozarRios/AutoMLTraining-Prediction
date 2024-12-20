from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
import os
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Initialize FastAPI app
app = FastAPI()

# Initialize H2O
h2o.init()

# Define directory structure
DATASET_DIR = "app/datasets"
MODEL_DIR = "app/models"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_model(model_dir, model_id):
    model_path = f"{model_dir}/{model_id}"
    return h2o.load_model(model_path)

@app.post("/train")
async def train_model(file: UploadFile, target_column: str = Form(...)):
    # Save the uploaded dataset to the datasets directory
    dataset_path = os.path.join(DATASET_DIR, file.filename)
    with open(dataset_path, "wb") as f:
        f.write(file.file.read())

    # Load the dataset as a pandas DataFrame
    train_df = pd.read_csv(dataset_path)
    train_df.dropna(inplace=True)

    # Convert the DataFrame into an H2OFrame
    hf = H2OFrame(train_df)
    target = target_column
    features = [col for col in hf.columns if col != target]

    # Train an H2O AutoML model
    aml = H2OAutoML(max_models=10, seed=42, balance_classes=True)
    aml.train(x=features, y=target, training_frame=hf)

    # Save the best model to the models directory
    best_model = aml.leader
    model_path = h2o.save_model(model=best_model, path=MODEL_DIR, force=True)

    # We will explaing the Variable Importamce of the best model
    varimp = best_model.varimp()


    return JSONResponse({
        "message": "Training completed successfully!",
        "model_id": os.path.basename(model_path), # Return only the model file name
        "variance_importance": varimp
    })

@app.post("/predict/results")
async def predict_results(file: UploadFile, model_id: str = Form(...), threshold: float = Form(...)):
    # Save and load dataset
    dataset_path = f"app/datasets/{file.filename}"
    with open(dataset_path, "wb") as f:
        f.write(file.file.read())
    hf_test = h2o.import_file(dataset_path)

    # Convert H2OFrame to pandas DataFrame
    test_df = hf_test.as_data_frame()
    test_df.dropna(inplace=True)

    # Load the trained model
    model = load_model(MODEL_DIR, model_id)

    # Predict with the model
    predictions = model.predict(hf_test)
    predictions_df = predictions.as_data_frame()

    # Add predictions to the test DataFrame
    test_df["Predicted_Probability"] = predictions_df["Y"]  # Use 'Y' for approval probability
    test_df["Recommendation"] = test_df["Predicted_Probability"].apply(
        lambda x: "Approve" if x >= threshold else "Reject"
    )

    # Prepare results
    result_dict = test_df.to_dict(orient="records")

    # Compute SHAP values using a background frame
    # Use a subset of the test data as the background frame
    try:
        background_frame, _ = hf_test.split_frame(ratios=[0.1], seed=123)  # Use 10% of the data

        # Ensure the model supports SHAP calculation with the background frame
        shap_values = model.predict_contributions(hf_test, background_frame=background_frame)
        shap_values_df = shap_values.as_data_frame()

        # Drop the "BiasTerm" column (intercept) as it's not a feature
        shap_values_df = shap_values_df.drop("BiasTerm", axis=1)

        # Aggregate SHAP values to get feature importance
        mean_abs_shap = shap_values_df.abs().mean().sort_values(ascending=False)
        feature_importance = mean_abs_shap.to_dict()
    except Exception as e:
        # Handle models that don't support SHAP calculations
        shap_values_df = None
        feature_importance = {"error": str(e)}

    # Include SHAP values and feature importance in the response
    return JSONResponse(content={
        "results": result_dict,
        "feature_importance": feature_importance
    })

