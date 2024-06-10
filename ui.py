import streamlit as st
import pandas as pd
import os

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

def save_uploaded_file(uploaded_file):
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def main():
    st.set_page_config(page_title="Relevancy Predictor", layout="wide")
    
    st.title("Relevancy Predictor")

    st.sidebar.header("Upload and Select Data")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Save the uploaded file
        save_uploaded_file(uploaded_file)
        st.sidebar.success("File uploaded and saved successfully!")
        
        pd.ExcelFile(uploaded_file)
        
        # Query input box
        query = st.text_input("Enter the statement for predicting its relevancy with all the patents")
        if st.button("Predict Relevancy"):
            pass  # Placeholder for future functionality

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
