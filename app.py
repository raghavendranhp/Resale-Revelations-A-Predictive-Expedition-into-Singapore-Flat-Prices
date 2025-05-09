import pandas as pd
import joblib
import numpy as np
from datetime import datetime, time
import streamlit as st
from streamlit_option_menu import option_menu
#Data visualization libraries
import seaborn as sns
sns.set()
import json

import matplotlib.pyplot as plt

#Machine learning libraries

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

#Ignore FutureWarnings to avoid clutter in the output
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

# Get the absolute path to the directory containing the current script
path = Path(__file__).parent.resolve()

# Add the path to sys.path if needed
import sys
sys.path.append(str(path))






# Load model and scaler
model_lr = joblib.load("Models/linear_regression_model.pkl")
scaler = joblib.load("Models/scaler.pkl")
#model_gb=joblib.load('Models/gradient_boosting_model.pkl')
##model_rf=joblib.load("Models/random_forest_model.pkl")

# Load feature encoding dictionary from JSON
with open("Datas/feat_dict.json", "r") as f:
    feat_dict = json.load(f)

def about_page():
    st.title(":blue[**Singapore Resale Flat Prices Prediction App**]")

    col1, col2 = st.columns([4, 2], gap='medium')
    
    with col1:
        st.subheader(":violet[Problem Statement:]")
        st.write("""
            The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application 
            that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat 
            transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.
        """)

        
        st.subheader(":violet[Scope:]")
        st.write("""
            This project involves:
            - Collecting and preprocessing data of resale flat transactions from 1990 to the present from the HDB.
            - Performing feature engineering to extract meaningful attributes like town, flat type, storey range, floor area, etc.
            - Selecting and training machine learning regression models such as Linear Regression, Decision Trees, and Random Forests.
            - Evaluating the models using metrics like MAE, MSE, RMSE, and R² score.
            - Deploying the model through a Streamlit web app to allow users to input flat details and get resale price predictions.
        """)

        st.subheader(":violet[Tools and Technologies Used:]")
        st.write("""
            - **Streamlit**: For building the web interface.
            - **Scikit-learn**: For training and evaluating machine learning models.
            - **Pandas & NumPy**: For data wrangling and numerical operations.
            - **Matplotlib & Seaborn**: For Exploratory Data Analysis and visualizations.
            - **Joblib**: For saving and loading ML models.
        """)

        st.subheader(":violet[Business Domain:]")
        st.write("""
            - **Domain**: Real Estate
            - The app is aimed at real estate stakeholders, including flat owners, buyers, agents, and government policymakers, 
              to assist in making data-informed decisions related to resale pricing.
        """)

    with col2:
        col2.markdown("#   ")
        col2.markdown("## <style>h2 {font-size: 18px;}</style> :orange[Skills Takeaway]:", unsafe_allow_html=True)
        col2.markdown("""
            ## <style>h2 {font-size: 16px;}</style> 
            - Data Wrangling
            - Exploratory Data Analysis (EDA)
            - Model Building
            - Model Deployment
        """, unsafe_allow_html=True)
        col2.markdown("#   ")
        col2.markdown("#   ")
        col2.markdown("#   ")
        col2.markdown("""
            ## <span style="font-size: 18px;">:orange[Created By]:</span><br>
            Raghavendran S,<br>
            Data Scientist Aspirant,<br>
            email-id: [raghavendranhp@gmail.com](mailto:raghavendranhp@gmail.com)<br>
            [LinkedIn-Profile](https://www.linkedin.com/in/raghavendransundararajan/),<br>
            [GitHub-Link](https://github.com/raghavendranhp)
        """, unsafe_allow_html=True)

def main_page():
    # Extract encoding maps
    flat_model_map = feat_dict["flat_model_dict"]
    flat_type_map = feat_dict["flat_type_dict"]
    town_map = feat_dict["town_dict"]
    street_map = feat_dict["street_dict"]

    st.title(" Singapore Flat Resale Price Predictor")

    # --- User Inputs ---
    user_input = {}

    # Select categorical features (dropdowns)
    flat_model_key = st.selectbox("Flat Model", options=list(flat_model_map.keys()))
    user_input["flat_model"] = flat_model_map[flat_model_key]

    flat_type_key = st.selectbox("Flat Type", options=list(flat_type_map.keys()))
    user_input["flat_type"] = flat_type_map[flat_type_key]

    town_key = st.selectbox("Town", options=list(town_map.keys()))
    user_input["town"] = town_map[town_key]

    street_key = st.selectbox("Street Name", options=list(street_map.keys()))
    user_input["street_name"] = street_map[street_key]

    # Numeric Inputs
    user_input["floor_area_sqm"] =int( st.number_input("Floor Area (sqm)", value=90, step=1))
    user_input["year"] =int(st.number_input("Year of Transaction", value=2023, step=1))
    user_input["storey_avg"] = int(st.number_input("Average Storey", value=8, step=1))
    user_input["Current_remaining_lease_years"] = int(st.number_input("Remaining Lease (years)", value=70, step=1))

    # --- Prediction ---
    if st.button("Predict Resale Price"):
        # Define column order to match training data
        feature_order = [
            'town', 'flat_type', 'street_name', 'floor_area_sqm', 'flat_model',
            'year', 'storey_avg', 'Current_remaining_lease_years'
        ]

        input_df = pd.DataFrame([user_input])[feature_order]

        # Scale numerical features (if scaler was trained)
        input_scaled = scaler.transform(input_df)

        # Predict resale price
        price_lr = model_lr.predict(input_scaled)[0]
        #price_gb=model_gb.predict(input_df)
        #price_rf=model_rf.predict(input_df)
        st.success(f"💰 Estimated Resale Price: SGD {price_lr:,.2f}")

def main():
    st.set_page_config(page_title="Singapore Flat Resale Price Predictor | By Raghavendran",layout="wide",page_icon="🏠")
    selected = option_menu('',["App","About"],
                           icons=["house","house"],
                           menu_icon="menu-button-wide",
                           default_index=0,orientation="horizontal",
                           styles={"nav-link": {"font-size": "18px", "text-align": "left", "margin": "-2px", "--hover-color": "#FF5A5F"},
                                   "nav-link-selected": {"background-color": "#6495ED"}}
                           )
   

    #App Page
    if selected == "App":
        st.header(":violet[**Singapore Flat Resale Prediction App**] -created by Raghav")
        main_page()
    
    #Aboutpage
    elif selected == "About":
        about_page()
    
   

#Run the Streamlit app
if __name__ == "__main__":
    main()

