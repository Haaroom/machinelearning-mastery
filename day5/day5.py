import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN  # Fixed: Capitalized 'M' in KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day5(new)\Mall_Customers.csv')

# 1. Separate features FIRST (Fixed: avoids missing column error after dummies)
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X = X.dropna()

# 2. Apply Encoding
X = pd.get_dummies(X, drop_first=True)

# 3. Scaling 
sc = StandardScaler()
scaled_X = sc.fit_transform(X)

# Elbow method & Silhouette loop
array = []
sil_score = []

# Fixed: Elbow tracks WCSS from 1-10
for i in range(1, 11):
    model = KMeans(n_clusters=i, random_state=42, init="k-means++", n_init='auto') # Fixed: corrected init string
    clusters = model.fit_predict(scaled_X)
    array.append(model.inertia_)
    
    # Fixed: Silhouette score only runs if clusters >= 2
    if i > 1:
        score = silhouette_score(scaled_X, clusters)
        sil_score.append(score)
        print(f"For K = {i}, Silhouette Score = {score:.4f}")

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), array, marker="o", linestyle="--", color="b")
plt.title("The Elbow Method For Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Plot Silhouette Scores (Starts from K=2 to 10)
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_score, marker="o", color="g")
plt.title("Silhouette Method For Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Models 
model1 = KMeans(n_clusters=3, random_state=42, init="k-means++", n_init='auto') # Fixed: Capital M
model2 = DBSCAN(eps=0.3, min_samples=5)
model3 = PCA(n_components=2)

# Fit original data
clusters = model1.fit_predict(scaled_X)
db_labels = model2.fit_predict(scaled_X)
pca = model3.fit_transform(scaled_X) # Fixed: Typo in fit_transform

print("\n--- Scores on Original Scaled Data ---")
print("silhouette score of k means  :", silhouette_score(scaled_X, clusters))
# DBSCAN might produce noise (-1). If it only finds 1 cluster + noise, silhouette will fail.
if len(set(db_labels)) > 1:
    print("silhouette score of DBSCAN   :", silhouette_score(scaled_X, db_labels))
else:
    print("DBSCAN failed to find distinct clusters with current eps/min_samples.")

# Models after using PCA 
pca_clusters = model1.fit_predict(pca)
pca_db_labels = model2.fit_predict(pca)

print("\n--- Scores after PCA Reduction ---")
print("silhouette score of k means  :", silhouette_score(pca, pca_clusters))
if len(set(pca_db_labels)) > 1:
    print("silhouette score of DBSCAN   :", silhouette_score(pca, pca_db_labels))
else:
    print("DBSCAN on PCA failed to find distinct clusters. Try increasing eps.")
