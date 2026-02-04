import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Try to import SMOTE (Install via: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: Library 'imbalanced-learn' not found. Please run: pip install imbalanced-learn")
    exit()

# ---------------------------------------------------------
# 1. Data Preprocessing
# ---------------------------------------------------------
# Load Dataset
file_path = 'CDC Diabetes Dataset.csv'
df = pd.read_csv(file_path)

# Data Cleaning: Remove Prediabetes (1) and merge Diabetes (2) into Binary (1)
if 'Diabetes_012' in df.columns:
    print("Preprocessing: Converting 3-class target to Binary...")
    df = df[df['Diabetes_012'] != 1]
    df['Diabetes_012'] = df['Diabetes_012'].replace({2: 1})
    df.rename(columns={'Diabetes_012': 'Diabetes_binary'}, inplace=True)

# Feature Scaling
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 2. Unsupervised Learning: Clustering
# ---------------------------------------------------------
print("\n--- Phase 1: Clustering Analysis ---")

# Elbow Method to find optimal K (Figure 1)
wcss = []
K_range = range(1, 11)
for i in K_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Figure 1: Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with K=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize Clusters using PCA with Jitter (Figure 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Add slight noise (jitter) to visualize overlapping points
X_pca_jitter = X_pca + np.random.normal(0, 0.5, size=X_pca.shape)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_jitter[:, 0], y=X_pca_jitter[:, 1], hue=clusters, palette='viridis', alpha=0.1, s=10)
plt.title('Figure 2: Cluster Visualization (PCA)')
plt.show()

# ---------------------------------------------------------
# 3. Supervised Learning: Classification (SMOTE + Threshold)
# ---------------------------------------------------------
print("\n--- Phase 2: Classification with SMOTE ---")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training set
print(f"Original Training Count: {np.bincount(y_train)}")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE Training Count:    {np.bincount(y_train_smote)}")

# Train Random Forest on balanced data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluation 1: Standard Threshold (0.5)
print("\n[Result A: Standard Threshold (0.5)]")
print(classification_report(y_test, rf_model.predict(X_test)))

# Evaluation 2: Adjusted Threshold (0.25) to improve Recall
custom_threshold = 0.25
y_pred_adjusted = (y_prob >= custom_threshold).astype(int)

print(f"\n[Result B: Adjusted Threshold ({custom_threshold}) - Optimized Recall]")
print(classification_report(y_test, y_pred_adjusted))

# ROC Curve (Figure 3)
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Figure 3: ROC Curve (SMOTE Model)')
plt.legend()
plt.show()

# Feature Importance (Figure 4)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]], align="center") # Top 10 features
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.title("Figure 4: Top 10 Feature Importance")
plt.tight_layout()
plt.show()
