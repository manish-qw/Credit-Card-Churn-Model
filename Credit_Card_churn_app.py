# Credit Card Churn Prediction Project - Step 1 to 7: Streamlit App Integration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import shap
import joblib
import os
import streamlit as st

# Set visualization style
sns.set(style='whitegrid')

# === 2. LOAD DATA ===
df = pd.read_csv('BankChurners.csv')

# Drop irrelevant columns (e.g., customer ID, unused columns)
df = df.drop(['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon',
         'Naive_Bayes_Classifier_Attrition_Flag_Income_Category_Months_Inactive_12_mon', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1, errors='ignore')

# === 3. BASIC CLEANING ===
if 'Attrition_Flag' in df.columns:
    df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
df.drop('Attrition_Flag', axis=1, inplace=True)


# === 5. FEATURE ENGINEERING ===
if 'Total_Revolving_Bal' in df.columns and 'Credit_Limit' in df.columns:
    df['Utilization_Rate'] = df['Total_Revolving_Bal'] / df['Credit_Limit']
if 'Months_on_book' in df.columns:
    df['Tenure_Group'] = pd.cut(df['Months_on_book'], bins=[0, 24, 36, 48, 60], labels=['0-24','25-36','37-48','49-60'])
if 'Total_Trans_Ct' in df.columns and 'Total_Relationship_Count' in df.columns:
    df['Transaction_Relationship_Interaction'] = df['Total_Trans_Ct'] * df['Total_Relationship_Count']

cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === 6. TRAIN-TEST SPLIT ===
X = df.drop(['Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
df.to_csv("feature_engineered_churn_data.csv", index=False)

# === 7. BASELINE & BEST MODEL ===
best_model = joblib.load("xgboost_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# === STREAMLIT APP ===
st.title("Credit Card Churn Prediction")

st.write("Upload a CSV file with customer information or manually input features below.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input():
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=0.0)
            input_data[col] = val
        return pd.DataFrame([input_data])
    input_df = user_input()

# Apply encoders if needed
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure all expected columns are present
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Make prediction
prediction = best_model.predict(input_df)
pred_proba = best_model.predict_proba(input_df)

st.subheader("Prediction")
result_text = "Churn" if prediction[0] == 1 else "Not Churn"
st.write(f"Customer is likely to: **{result_text}**")
st.write(f"Probability of Churn: {pred_proba[0][1]:.2f}")