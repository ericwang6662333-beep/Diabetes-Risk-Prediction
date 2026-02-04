import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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