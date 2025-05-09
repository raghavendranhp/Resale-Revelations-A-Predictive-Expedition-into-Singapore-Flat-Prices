

---

# Singapore Resale Flat Prices Prediction App

## Problem Statement

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. The predictive model is based on historical data from resale flat transactions, assisting both potential buyers and sellers in estimating the resale value of a flat.



## Scope

The project involves the following steps:

1. **Data Collection and Preprocessing**: Gathering historical resale flat transaction data from the Singapore Housing and Development Board (HDB) from 1990 to the present. Data is cleaned and structured for use in machine learning models.
2. **Feature Engineering**: Extracting relevant features such as town, flat type, storey range, floor area, and lease commence date to enhance the prediction accuracy.
3. **Model Selection and Training**: Implementing regression models (e.g., Linear Regression, Decision Trees, and Random Forests) and training them on historical data.
4. **Model Evaluation**: Evaluating models using regression metrics like MAE, MSE, RMSE, and R² score to determine the best-performing model.
5. **Streamlit Web Application**: Developing a web application with Streamlit that allows users to input flat details (town, flat type, storey range, etc.) and receive predictions of resale prices.
6. **Deployment on Render**: Deploying the web app on the Render platform for public access.
7. **Testing and Validation**: Thorough testing of the deployed application to ensure accuracy and usability.

## Tools and Technologies Used

* **Streamlit**: For building the web interface.
* **Scikit-learn**: For training and evaluating machine learning models.
* **Pandas & NumPy**: For data wrangling and numerical operations.
* **Matplotlib & Seaborn**: For exploratory data analysis and visualizations.
* **Joblib**: For saving and loading trained machine learning models.
* **Render**: For deploying the web application.

## Business Domain

* **Domain**: Real Estate
  The app is designed for real estate stakeholders, including flat owners, buyers, agents, and policymakers, to help them make informed decisions regarding resale flat pricing.

## Skills Acquired

* Data wrangling and preprocessing
* Exploratory data analysis (EDA)
* Model building using machine learning algorithms
* Model evaluation and optimization
* Web application deployment

## Project Structure

The project directory is organized as follows:

```
├── Datas/
│   ├── Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv
│   ├── Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv
│   ├── Resale Flat Prices (Based on Registration Date), From Jan 2015 to....csv
│   ├── Resale Flat Prices (Based on Registration Date), From Mar 2012 t...csv
│   ├── Resale flat prices based on registration date from Jan-2017 onward.csv
│   ├── feat_dict.json
├── Models/
│   ├── gradient_boosting_model.pkl
│   ├── linear_regression_model.pkl
│   ├── scaler.pkl
├── flat_project_env/
│   ├── Lib/
│   ├── Scripts/
│   ├── etc/
│   ├── pyvenv.cfg
├── .gitignore
├── Pre_process_EDAipynb
├── Readme.md
├── app.py
├── packages.txt
├── requirements.txt
```

* **Datas/**: Contains the dataset of resale flat transactions for training the model.
* **Models/**: Includes the trained machine learning models (e.g., Linear Regression, Gradient Boosting) and scaler used for normalization.
* **flat\_project\_env/**: Virtual environment with necessary packages installed.
* **app.py**: The main script that runs the Streamlit web app.
* **Pre\_process\_EDAipynb**: Jupyter notebook for data preprocessing and exploratory data analysis.
* **requirements.txt**: Python dependencies required for the project.
* **packages.txt**: List of installed packages in the project environment.

## How to Run the Project Locally

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   ```

2. **Set up a virtual environment**:
   Create a virtual environment to install the required dependencies:

   ```bash
   python -m venv flat_project_env
   ```

3. **Activate the virtual environment**:

   * On Windows:

     ```bash
     .\flat_project_env\Scripts\activate
     ```
   * On macOS/Linux:

     ```bash
     source flat_project_env/bin/activate
     ```

4. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**:

   ```bash
   streamlit run app.py
   ```

6. **Access the app**: Open the provided URL (usually `http://localhost:8501`) in your browser to interact with the app.

## Deployed Application

The app is deployed and can be accessed via the following URL:

[**Singapore Resale Flat Prices Prediction App**](https://resale-revelations-a-predictive.onrender.com)

## Acknowledgements

* Singapore Housing and Development Board (HDB) for providing the dataset.
* Streamlit and Scikit-learn for their easy-to-use frameworks and libraries for deployment and model building.

## Created By

**Raghavendran S**
Data Scientist Aspirant
Email: [raghavendranhp@gmail.com](mailto:raghavendranhp@gmail.com)
[LinkedIn Profile](LinkedIn URL)
[GitHub](GitHub URL)

---



