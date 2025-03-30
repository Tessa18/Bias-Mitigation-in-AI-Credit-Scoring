# Bias-Mitigation-in-AI-Credit-Scoring
This repository contains a machine learning project for predicting credit scores using Logistic Regression. It also explores privacy-preserving techniques using the Opacus library.

## Introduction
Credit scoring is a crucial process in financial services, helping lenders assess the risk of potential borrowers. This project implements a credit scoring model using Logistic Regression and explores the use of differential privacy with Opacus.

## Features
- Synthetic data generation for credit scoring.
- Data preprocessing including feature scaling.
- Model training using Logistic Regression.
- Evaluation metrics such as accuracy and confusion matrix.
- Implementation of differential privacy using Opacus.
- Fairness analysis of the model using fairness metrics.

## Installation
To run the project, install the required dependencies:
*     pip install --upgrade numpy pandas scikit-learn opacus torch

## Dataset
This project uses two datasets:

**1. Synthetic Credit Dataset:**

- Features: Age, Income, Credit History, Loan Amount.

- Target variable: 0 (High Risk) or 1 (Low Risk).

**2. German Credit Dataset:**

- Description: Contains information about applicants for credit, including their personal and financial details. It is commonly used to study fairness in credit scoring models.

- Fairness Concerns: Gender and age biases are often explored, where certain groups might be unfairly denied credit.

- Source: German Credit Dataset from the UCI Machine Learning Repository.

- Fairness Metrics:

  - Disparate Impact

  - Statistical Parity Difference

## Model Training
The model pipeline includes:
1. **Data Spliting**: 70% training, 30% testing.
2. **Feature Scaling** : Standardization using StandardScaler.
3. **Model Selection**: Logistic Regression.
4. **Training**: The model is trained on the synthetic dataset and German Credit datasets.

## Privacy-Preserving Machine Learning
To introduce differential privacy:
- We use the **Opacus library**, which provides privacy guarantees by adding noise to gradients during training.
- The model is trained with **differential privacy constraints**, ensuring sensitive data is protected.

## Fairness Analysis

To assess the fairness of the credit scoring model:

- We evaluate **disparate impact** and **statistical parity difference** to identify potential biases related to gender and age.

- The fairness metrics help in understanding if certain demographic groups face discrimination in credit approvals.

## Evaluation
The model performance is evaluated using:
- Accuracy Score
- Confusion Matrix
- Fairness Metrics (Disparate Impact, Statistical Parity Difference)

## Usage
1. **Run the colab Notebook** to execute the steps in order.
2. **Synthetic Data Generation**: Creates a dataset with key financial features.
3. **Model Training**: Trains a logistic regression, Random Forest and Gradient Boosting model.
4. **Privacy-Preserving ML**: Applies Opacus for differential privacy.
5. **Fairness Analysis**: Evaluates biases in the credit scoring model.
6. **Evaluation**: Assesses model performance.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Torch
- Opacus

## Results
The Project demonstrates that:
- Logistic Regression provides a baseline for credit scoring.
- Privacy-preserving training with Opacus slightly impacts performance but enhances security.
- Fairness metrics reveal potential biases in the German Credit Dataset, highlighting the importance of ethical AI in financial decision-making.
