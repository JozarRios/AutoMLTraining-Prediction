# Updated streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import altair as alt

from io import StringIO
from PIL import Image
import base64

# Define FastAPI base URL
API_URL = "http://127.0.0.1:8000"  # Update if your FastAPI server runs on a different port or IP

st.title("AutoML Training and Prediction")

# Tabs for Training and Prediction
tab1, tab2= st.tabs(["Train Model", "Make Predictions"])

# Train Model Tab
with tab1:
    st.header("Train a New Model")
    uploaded_train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"], key="train_file")
    target_column = st.text_input("Enter Target Column Name (e.g., Loan_Status)", key="target_col")

    if st.button("Train Model"):
        if uploaded_train_file and target_column:
            # Send request to Train API
            files = {"file": uploaded_train_file.getvalue()}
            data = {"target_column": target_column}
            response = requests.post(f"{API_URL}/train", files=files, data=data)

            if response.status_code == 200:
                train_response = response.json()
                st.success("Model Trained Successfully!")
                st.json(train_response)
                # Check if 'variance_importance' exists in the response
                if "variance_importance" in train_response:
                    st.subheader("Variance Importance Chart")

                    # Convert variance_importance to DataFrame
                    variance_importance = train_response["variance_importance"]
                    df = pd.DataFrame(variance_importance,
                                      columns=["Feature", "Importance1", "Importance2", "VarianceImportance"])

                    # Filter features with non-zero variance importance
                    filtered_df = df[df["VarianceImportance"] > 0]

                    # Sort by VarianceImportance for better visualization
                    filtered_df = filtered_df.sort_values(by="VarianceImportance", ascending=True)

                    # Create a horizontal bar chart using Altair
                    chart = alt.Chart(filtered_df).mark_bar().encode(
                        x=alt.X("VarianceImportance:Q", title="Variance Importance"),
                        y=alt.Y("Feature:N", sort='-x', title="Feature"),
                        color=alt.Color("VarianceImportance:Q", scale=alt.Scale(scheme="blues"))
                    ).properties(
                        title="Variance Importance",
                        width=600,
                        height=400
                    )

                    # Display the chart
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Variance importance data not available in the response.")
            else:
                st.error("Failed to train model. Check your input or API logs.")
        else:
            st.warning("Please upload a training dataset and specify the target column.")
#prediction tab
with tab2:
    st.header("Make Predictions")
    uploaded_test_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"], key="test_file")
    threshold = st.slider("Set Prediction Threshold", min_value=0.0, max_value=1.0, value=0.5)
    model_id = st.text_input("Enter Model ID")

    if st.button("Make Prediction"):
        if uploaded_test_file:
            # Send request to Predict API
            files = {"file": uploaded_test_file.getvalue()}
            data = {"threshold": threshold}
            if model_id:
                data["model_id"] = model_id
            response = requests.post(f"{API_URL}/predict/results", files=files, data=data)

            if response.status_code == 200:
                predict_response = response.json()
                results = predict_response.get("results", [])
                feature_importance = predict_response.get("feature_importance", [])

                if results:
                    st.success("Predictions Made Successfully!")
                    st.dataframe(pd.DataFrame(results))

                    # Display Feature Importance Chart
                    if feature_importance:
                        st.subheader("SHAP Feature Importance Chart")

                        # Convert feature importance dictionary to DataFrame
                        importance_df = pd.DataFrame(
                            [{"Feature": k, "Importance": v} for k, v in feature_importance.items()]
                        )

                        # Sort by Importance for better visualization
                        importance_df = importance_df.sort_values(by="Importance", ascending=False)

                        # Create a horizontal bar chart using Altair
                        chart = alt.Chart(importance_df).mark_bar().encode(
                            x=alt.X("Importance:Q", title="Feature Importance"),
                            y=alt.Y("Feature:N", sort='-x', title="Feature"),
                            color=alt.Color("Importance:Q", scale=alt.Scale(scheme="viridis"))
                        ).properties(
                            title="Feature Importance",
                            width=600,
                            height=400
                        )

                        # Display the chart
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("Feature importance data not available.")
                else:
                    st.error(f"Failed to make predictions. Error: {response.text}")
            else:
                st.warning("Please upload a test dataset.")