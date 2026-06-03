from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import umap
# Load the digits dataset
digits = load_digits()
# Extract the features and target variable
X = digits.data
y = digits.target
#perfoming EDA 
print(X.shape)
print(y.shape)
# Visualize the data
plt.imshow(X[0].reshape(8,8))
plt.show()
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio (PCA):", pca.explained_variance_ratio_)
# Visualize the PCA result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Digits Dataset')
plt.show()
# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
# Visualize the t-SNE result
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Digits Dataset')
plt.show()
# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
# Visualize the UMAP result
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP of Digits Dataset')
plt.show()
# Train a logistic regression model on the original data
model1=LogisticRegression(max_iter=1000)
model2=LogisticRegression(max_iter=1000)
model3=LogisticRegression(max_iter=1000)
model4=LogisticRegression(max_iter=1000)
model1.fit(X_scaled,y)
model2.fit(X_pca,y)
model3.fit(X_tsne,y)
model4.fit(X_umap,y)
# Evaluate the model
accuracy1 = model1.score(X_scaled, y)
accuracy2 = model2.score(X_pca, y)
accuracy3 = model3.score(X_tsne, y)
accuracy4 = model4.score(X_umap, y)
print(f'Accuracy on original data: {accuracy1:.2f}')
print(f'Accuracy on PCA data: {accuracy2:.2f}')
print(f'Accuracy on t-SNE data: {accuracy3:.2f}')
print(f'Accuracy on UMAP data: {accuracy4:.2f}')
