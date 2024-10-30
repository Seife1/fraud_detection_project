import shap
import joblib
from lime import lime_tabular
import matplotlib.pyplot as plt

# Initialize SHAP JavaScript for interactive plots (useful in Jupyter Notebooks)
shap.initjs()

class FraudDetectionExplainer:
    """
    Explains predictions of a fraud detection model using SHAP and LIME.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file (.pkl).
    X_test : DataFrame
        Test dataset as a pandas DataFrame.
    """

    def __init__(self, model_path, X_test):
        # Load model and test data
        self.model = joblib.load(model_path)
        self.X_test = X_test

        # If the model is in a pipeline, extract the final model step
        if hasattr(self.model, 'steps'):
            self.model = self.model.steps[-1][1]

    def shap_explanation(self, instance_idx=0):
        """Generate SHAP plots: summary, force, and dependence plots for fraud detection."""

        # Create SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_test)  # Direct call to get an Explanation object

        # SHAP Summary Plot: Show feature importance across all test instances
        shap.summary_plot(shap_values.values, self.X_test)
        plt.title('SHAP Summary Plot for Fraud Detection')

        # SHAP Force Plot: Visualize feature impact on prediction for one instance
        shap.force_plot(
            shap_values.base_values[instance_idx], 
            shap_values.values[instance_idx], 
            self.X_test.iloc[instance_idx, :]
        )
        plt.title(f'SHAP Force Plot for Fraud Detection Instance {instance_idx}')

        # SHAP Dependence Plot: Shows how one feature affects model output
        shap.dependence_plot(self.X_test.columns[0], shap_values.values, self.X_test)
        plt.title(f'SHAP Dependence Plot for Feature: {self.X_test.columns[0]}')


    def lime_explanation(self, instance_idx=0):
        """Generate LIME feature importance plot for a single fraud detection instance."""
        

        # Create LIME explainer for fraud detection model
        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=self.X_test.values, 
            feature_names=self.X_test.columns, 
            mode='classification'
        )

        # Explain a single instance using LIME
        instance = self.X_test.iloc[instance_idx].values.flatten()
        explanation = explainer_lime.explain_instance(instance, self.model.predict_proba)

        # Display LIME Feature Importance Plot
        explanation.as_pyplot_figure()
        plt.title(f'LIME Explanation for Fraud Detection Instance {instance_idx}')
        plt.show()

    def explain(self, instance_idx=0):
        """Run SHAP and LIME explanations for the selected instance in fraud detection."""
        self.shap_explanation(instance_idx)
        self.lime_explanation(instance_idx)
