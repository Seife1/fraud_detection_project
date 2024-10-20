# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview
This project aims to build a robust fraud detection system for e-commerce and bank transactions. The project focuses on creating machine learning models to detect fraudulent activities, utilizing geolocation data and transaction patterns. Additionally, the models are deployed using Flask, containerized with Docker, and visualized through an interactive dashboard using Dash.

### Project Structure
- **data/**: Contains the raw, processed, and external datasets used for the project.
- **notebooks/**: Jupyter notebooks for Exploratory Data Analysis (EDA), data preprocessing, and model training.
- **src/**: Python scripts for data preprocessing, model training, evaluation, and explainability.
- **deployment/**: Scripts for deploying the machine learning model using Flask and Docker.
- **dashboard/**: Contains the Dash app for visualizing fraud trends and insights.
- **tests/**: Contains unit tests for model training and deployment.
  
### Technologies Used
- **Python**: Main programming language.
- **Flask**: Used to build REST APIs for model serving.
- **Dash**: For interactive data visualization.
- **Docker**: Containerizing the application.
- **Pandas, Scikit-learn, TensorFlow**: For data preprocessing and machine learning.
- **SHAP & LIME**: For model explainability.

## Datasets
- **Fraud_Data.csv**: E-commerce transaction data used for fraud detection.
- **creditcard.csv**: Bank transaction data used for fraud detection.
- **IpAddress_to_Country.csv**: Mapping IP addresses to countries for geolocation analysis.

## Setup Instructions