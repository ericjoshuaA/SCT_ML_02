import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Feature_1', y='Feature_2', hue='Cluster', data=df,
                palette='viridis', s=100, alpha=0.8, legend='auto')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
plt.title(f'K-means Clustering (K={optimal_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

print(df.head())
print(df['Cluster'].value_counts().sort_index())
print(df.groupby('Cluster')[['Feature_1', 'Feature_2']].mean())