# Credit Card Churn Prediction
A complete end-to-end machine learning project to identify potential customer churn in credit card usage. The project covers data preprocessing, model training, SHAP-based feature selection, fairness evaluation, and deployment on a cloud platform.

# 1. Objective
To predict which customers are likely to churn from a credit card service using structured customer data. The model is designed to be accurate, interpretable, and fair across demographic groups.

# 2. Data Processing and Feature Engineering
The dataset was first cleaned by removing irrelevant columns and encoding categorical features. Exploratory Data Analysis was conducted using boxplots, histograms, and heatmaps to understand feature relationships. Based on these insights, new features were created to capture customer behavior more effectively.

# 3. Baseline Modeling
Initial models including logistic regression, decision tree, and random forest were trained to set performance benchmarks.
The best baseline (Random Forest) achieved:
Accuracy: 95.6%
ROC-AUC: 0.88

# 4. Advanced Modeling with XGBoost
An XGBoost model was trained and tuned using GridSearchCV with parameters:
max_depth=3, n_estimators=300, learning_rate=0.2.
After hyperparameter tuning and SHAP-based feature selection, the model achieved:
Accuracy: 97%
ROC-AUC: 0.94
This significantly improved model interpretability while maintaining high performance.

# 5. Fairness Evaluation
The fairness of the model was assessed across the Gender feature using two metrics:
Demographic Parity Difference (DPD): 0.0034
Equalized Odds Difference (EOD): 0.0295
These results show that the model makes fair predictions across gender groups with minimal bias.

# 6. Deployment
The final model was saved using Pickle and deployed to the Render platform.
A hosted API was created to serve real-time churn predictions, making the solution production-ready.

# 7. Tech Stack
Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
Tools: Pickle (model serialization), Render (deployment)

# Summary
This project demonstrates the full lifecycle of a machine learning solution — from raw data and model building to fairness evaluation and real-time deployment — with a strong focus on performance, interpretability, and responsible AI.
