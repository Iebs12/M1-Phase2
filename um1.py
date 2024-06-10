import streamlit as st
import pandas as pd
import os
import requests

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
    

def main():
    st.set_page_config(page_title="Relevancy Predictor", layout="wide")
    st.title("Relevancy Predictor")

    st.sidebar.header("Upload and Select Data")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        
        query = st.text_input("Enter the statement for predicting its relevancy with all the patents")



        col1, col2 = st.columns([2, 1])
        with col1:
            predict_button = st.button("Predict Relevancy")

        if predict_button:
            if query:
                response = requests.post("http://localhost:5000/", json={"query": query,
                                                                        "file_path": file_path})
                if response.status_code == 200:
                    newFilePath = response.json().get('Path')
                    st.write(f"Result: {newFilePath}")
                else:
                    st.write("Error: Could not process the query")
            else:
                st.write("Please enter a query")

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
