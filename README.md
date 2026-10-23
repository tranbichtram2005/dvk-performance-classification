# Business Unit Performance Classification

A machine learning project that classifies the performance levels of business units (*Đơn vị kinh doanh – ĐVKD*) using operational survey data.

## Overview

This project explores whether operational survey data can help classify business units into three performance categories:

- **Not Completed** (`Chưa hoàn thành`)
- **Completed** (`Hoàn thành`)
- **Excellently Completed** (`Hoàn thành xuất sắc`)

The workflow covers data preprocessing, feature engineering, model training, and evaluation.

## Objectives

- Preprocess and standardize operational survey data.
- Group business processes into meaningful categories.
- Engineer features describing transaction volume, self-handling rates, IT support requirements, processing time, and error rates.
- Aggregate survey responses at the business-unit level and merge them with performance labels.
- Compare classification models using stratified cross-validation.
- Evaluate model performance using multiple metrics and visualizations.

## Methodology

### 1. Data Preprocessing

- Standardize column names and business-unit identifiers.
- Remove duplicate employee survey records according to the implemented data-cleaning rules.
- Identify and group operational process variables.
- Handle missing feature values within the machine learning pipeline.

### 2. Feature Engineering

Operational processes are organized into five groups:

- Customer Information File and Accounts (`CIF & Accounts`)
- Money Transactions (`Money Transactions`)
- Funds (`Funds`)
- Collateral (`Collateral`)
- Credit (`Credit`)

Features are aggregated at the business-unit level to represent operational characteristics.

### 3. Machine Learning Models

The project compares:

- **Random Forest**
- **Logistic Regression**
- **Majority-class Baseline**

The machine learning pipeline uses `scikit-learn` and includes imputation, feature scaling where applicable, and classification.

### 4. Model Evaluation

Models are evaluated using stratified 5-fold cross-validation.

Evaluation metrics include:

- Accuracy
- Balanced Accuracy
- Macro F1-score
- Weighted Precision, Recall, and F1-score
- Confusion Matrix
- ROC Curve and AUC, where applicable

Feature importance and prediction probability distributions are also examined to support model interpretation.

```

*The structure above is a suggested organization. Adjust it to match the files actually committed to the repository.*
