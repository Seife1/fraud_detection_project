# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview
This project aims to build a robust fraud detection system for e-commerce and bank transactions. The project focuses on creating machine learning models to detect fraudulent activities, utilizing geolocation data and transaction patterns. Additionally, the models are deployed using Flask, containerized with Docker, and visualized through an interactive dashboard using Dash.

### Project Structure
- **data/**: Contains the raw, processed, and external datasets used for the project.
- **notebooks/**: Jupyter notebooks for Exploratory Data Analysis (EDA), data preprocessing, and model training.
- **src/**: Python scripts for data preprocessing, model training, evaluation, and explainability.
- **deployment/**: Scripts for deploying the machine learning model using Flask and Docker.

#### Future Work
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

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fraud_detection_project.git
   cd fraud_detection_project
   ```
2. **Install Dependencies**:
    Create a virtual environment and install required packages:
    ```bash
    conda create --name week_89 python=3.11
    conda activate week_89
    pip install -r deployment/requirements.txt
    ```
3. **Run the Model Training**: You can run model training scripts directly from the src/ folder:
```bash
python src/model_training.py
```
I have used the script in the model training notebook.

4. **Run the Flask API:** To serve the trained model via a Flask API:
```bash
cd deployment/app
python serve_model.py
```

5. **Model Explainability**
Model explainability is implemented using SHAP and LIME to interpret the machine learning models. You can run the script in src/`explainability.py` to generate feature importance plots.

