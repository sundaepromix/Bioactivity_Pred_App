import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Jewelry Price Prediction",
    page_icon="💎",
    layout="wide"
)

# Set MLflow tracking URI for deployment
mlflow.set_tracking_uri("file:///mount/src/jewelry-price-optimization/mlruns")

def load_best_model():
    """
    Load the best model based on the highest test_r2 metric.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_name = "jewelry_price_optimization"
        
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            st.error("Experiment not found. Please check the experiment name.")
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )
        if not runs:
            st.error("No runs found in the experiment.")
            return None

        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/catboost_model"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    """
    Main Streamlit application function for jewelry price prediction.
    """
    st.title("💎 Jewelry Price Prediction")
    st.write("Enter product details to predict the optimal price.")

    # Load the model
    model = load_best_model()
    if not model:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        # Input fields
        category = st.selectbox(
            "Category",
            [
                "jewelry.pendant",
                "jewelry.necklace",
                "jewelry.earring",
                "jewelry.ring",
                "jewelry.brooch",
                "jewelry.bracelet",
                "jewelry.souvenir",
                "jewelry.stud"
            ]
        )
        brand_id = st.radio("Brand ID", [0, 1], format_func=lambda x: f"Brand {x}")
        target_gender = st.selectbox("Target Gender", ["f", "m", None])

    with col2:
        main_color = st.selectbox("Main Color", ["yellow", "white", "red", None])
        main_metal = st.selectbox("Main Metal", ["gold", None])
        main_gem = st.selectbox("Main Gem", ["sapphire", "diamond",