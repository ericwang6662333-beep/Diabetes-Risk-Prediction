import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. Data Preprocessing
# ---------------------------------------------------------
# Load Dataset
file_path = 'CDC Diabetes Dataset.csv'
df = pd.read_csv(file_path)

if 'Diabetes_012' in df.columns:
    print("Preprocessing: Converting 3-class target to Binary...")
    df = df[df['Diabetes_012'] != 1]
    df['Diabetes_012'] = df['Diabetes_012'].replace({2: 1})
    df.rename(columns={'Diabetes_012': 'Diabetes_binary'}, inplace=True)

X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 2. Unsupervised Learning: Clustering
# ---------------------------------------------------------
print("\n--- Phase 1: Clustering Analysis ---")
# (Elbow method code omitted for brevity in this commit, assuming K=3 is chosen)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------
# 3. Supervised Learning: Baseline Classification
# ---------------------------------------------------------
print("\n--- Phase 2: Classification (Baseline) ---")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original Training Count: {np.bincount(y_train)}")

# Train Random Forest on IMBALANCED data (Baseline)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation: Standard Threshold (0.5)
print("\n[Result: Baseline Random Forest]")
print(classification_report(y_test, rf_model.predict(X_test)))