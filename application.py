"""Streamlit UI to pick squads and predict top 11 performers."""
import streamlit as st
import pandas as pd
from pathlib import Path
from src import predict
from src.data_loader import load_dataset
from src.config import DATA_DIR

st.set_page_config(page_title="Top 11 Player Predictor", layout="wide")

st.title("🏏 Top 11 Cricket Player Predictor")

st.sidebar.header("Input Options")

sample_data_path = DATA_DIR / "cricket_top11_multinational.csv"

data_source = st.sidebar.radio(
    "Choose input data:",
    ("Use sample historical dataset", "Upload your own squad CSV"),
)

if data_source == "Use sample historical dataset":
    df_input = load_dataset(sample_data_path)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

st.write("### Input Data Preview")
st.dataframe(df_input.head())

if st.button("Predict Top 11"):
    with st.spinner("Predicting..."):
        result_df = predict.predict(df_input)
        st.success("Done!")
        st.write("### 🏆 Predicted Top 11 Players")
        st.table(result_df[['player_name', 'team', 'pred_proba_top']])
