import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import os 
from pathlib import Path

# Set MLflow tracking URI
tracking_uri = os.path.join(Path.home(), "Documents", "mlruns")
mlflow.set_tracking_uri(f"file://{tracking_uri}")

def load_best_model():
    """
    Load the best model based on the highest test_r2 metric.
    """
    try:
        # Access the MLflow client
        client = mlflow.tracking.MlflowClient()
        experiment_name = "jewelry_price_optimization"  
        
        # Get the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError("Experiment not found. Check the experiment name.")

        # Search for the best run (highest test_r2)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError("No runs found in the experiment.")

        # Get the best run's artifact path
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/catboost_model" 
        
        # Load the model
        model = mlflow.sklearn.load_model(model_uri)
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    """
    Main Streamlit application function for jewelry price prediction.
    """
    st.title("Jewelry Price Prediction")

    # Load the model
    model = load_best_model()
    if not model:
        return

    st.header("Enter Product Details")

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
    main_color = st.selectbox("Main Color", ["yellow", "white", "red", None])
    main_metal = st.selectbox("Main Metal", ["gold", None])
    main_gem = st.selectbox("Main Gem", ["sapphire", "diamond", "amethyst", None])

    if st.button("Predict Price"):
        # Create input dataframe
        input_data = pd.DataFrame({
            "Category": [category],
            "Brand_ID": [brand_id],
            "Target_Gender": [target_gender],
            "Main_Color": [main_color],
            "Main_Metal": [main_metal],
            "Main_Gem": [main_gem]
        })

        try:
            # Predict the price
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Price: ${prediction:.2f}")
            st.info(f"Suggested Price Range: ${prediction * 0.9:.2f} - ${prediction * 1.1:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    main()
