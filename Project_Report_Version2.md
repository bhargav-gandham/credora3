# Data Science Project Report

## Project Title
Decision Tree Classifier for Customer Purchase Prediction

## Overview

This project aims to build a decision tree classifier to predict whether a customer will purchase a product or service, using demographic and behavioral data. The process follows standard data science practices including data preprocessing, feature engineering, model training, evaluation, and visualization.

## Dataset

- **Filename:** bank-full.csv
- **Description:** The dataset contains demographic and behavioral information about customers. It is based on the UCI Bank Marketing dataset.
- **Source:** [UCI ML Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Objectives

- Build a decision tree classifier.
- Predict if a customer will purchase a product/service.
- Use demographic and behavioral data.
- Evaluate model performance and visualize results.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Methodology

### 1. Data Exploration
- Loaded and inspected the dataset.
- Checked for missing values and data types.
- Explored class distribution and key features.

### 2. Data Preprocessing
- Handled missing values.
- Encoded categorical features using label encoding.
- Split data into features (X) and target (y).

### 3. Feature Engineering
- Selected relevant features based on exploratory analysis.

### 4. Model Training
- Split data into training and testing sets (80/20 split).
- Trained a Decision Tree Classifier.

### 5. Model Evaluation
- Evaluated accuracy, confusion matrix, and classification report.
- Visualized the decision tree structure.

## Results

- **Accuracy:** _[Insert Accuracy]_
- **Confusion Matrix:**  
  _[Insert matrix as image or table]_
- **Classification Report:**  
  _[Insert precision, recall, f1-score, support]_
- **Decision Tree Visualization:**  
  _[Insert decision tree plot/image]_

## Conclusion

- The decision tree classifier provides insights into which features are most influential in predicting customer purchases.
- The model can be further improved by hyperparameter tuning or using more advanced algorithms.

## References

- UCI ML Repository
- Scikit-learn Documentation

---

*Prepared by: [Your Name]*