\# Customer Churn Prediction System



An intermediate machine learning project that analyzes telecom customer churn, compares classification models, and estimates the probability that a customer may leave a service.



\## Features



\- Customer churn analysis and business overview

\- Interactive customer filtering and CSV export

\- Individual customer churn prediction

\- Comparison of Logistic Regression, Random Forest, and Gradient Boosting

\- Evaluation using accuracy, precision, recall, F1 score, and ROC-AUC

\- Confusion matrix and classification report



\## Tech Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Matplotlib

\- Streamlit

\- Joblib



\## Machine Learning Workflow



1\. Load and validate the customer dataset.

2\. Handle missing numeric values.

3\. Encode categorical features and scale numeric features using a preprocessing pipeline.

4\. Train three classification models.

5\. Evaluate each model on a held-out test set.

6\. Select the best model based on F1 score.

7\. Use the saved model in the Streamlit dashboard.



\## Run Locally



Clone this repository and open its folder in a terminal.



```bash

python -m venv .venv

```



On Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python train\_model.py

streamlit run app.py

```



\## Dataset



The project uses the IBM Telco Customer Churn dataset, which contains customer account information, subscribed services, billing details, and a churn label.



Original dataset source: \[IBM Watsonx AI Samples](https://github.com/IBM/watsonx-ai-samples)



\## Limitations



\- Predictions are estimates based on historical data, not guarantees about future customer behavior.

\- Model performance may change on data from other telecom companies or later periods.

\- This project is a portfolio demonstration, not a production customer-retention system.



\## Future Improvements



\- Threshold tuning based on business costs

\- Cross-validation and model calibration

\- Feature importance and explainability

\- Monitoring for data drift

\- Deployment with automated testing



\## Author



Built as a machine learning portfolio project.



