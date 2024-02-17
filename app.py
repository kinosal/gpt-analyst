"""Streamlit app to analyze CSV files with GTPs."""

import os
import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
import matplotlib.pyplot as plt
import seaborn as sns


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_token=OPENAI_API_KEY)

# Configure Streamlit page and state
st.set_page_config(page_title="GPT Analyst", page_icon="🤖")
if "result" not in st.session_state:
    st.session_state.result = ""

# Render Streamlit page
st.title("Analyze data with natural language")
st.markdown("""Upload a CSV file and write your prompt.""")

csv_file = st.file_uploader(
    label="Upload CSV",
    type=["csv"],
    accept_multiple_files=False,
)

prompt = st.text_input(
    label="Enter prompt",
    value="How many records are in the dataset?",
)

if st.button(label="Run", type="primary"):
    if not csv_file:
        st.warning("Please upload a CSV file.")

    if not prompt:
        st.warning("Please enter a prompt.")

    if csv_file and prompt:
        with st.spinner("Analyzing..."):
            df = SmartDataframe(
                pd.read_csv(csv_file),
                config={
                    "llm": OpenAI(),
                    "verbose": True,
                    "conversational": False,
                    "chart_dir": "exports/charts",
                },
            )
            response = df.chat(prompt)
            if os.path.exists("exports/charts/temp_chart.png"):
                st.image("exports/charts/temp_chart.png")
                os.remove("exports/charts/temp_chart.png")
            else:
                st.write(response)

            # st.header("Logs")
            # st.write(df.logger.logs)
