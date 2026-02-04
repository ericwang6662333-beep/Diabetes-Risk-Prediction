# Diabetes-Risk-Prediction

## Project Overview
This project applies Unsupervised (K-Means) and Supervised Learning (Random Forest with SMOTE) to predict diabetes risk using the CDC Health Indicators dataset. The goal is to identify high-risk individuals to facilitate early medical intervention.

## Key Features
Data Preprocessing: Cleaning and scaling of 250k+ records.
Clustering: Used Elbow Method and PCA to visualize patient segmentation.
Classification: Implemented Random Forest.
Optimization: Addressed class imbalance using SMOTE and Threshold Tuning, improving Recall from 0.18 to 0.69.

## Requirements
To run this code, please install the dependencies:
```bash
pip install -r requirements.txt
