import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



#load csv 
df = pd.read_csv(r'H:\PROGRAMMING\HAAROON\MACHINELEARNING\day8(new)\Mall_Customers.csv')
#basic eda 
#print(df.head())
#print(df.info())
#print(df.describe())
print(df.isnull().sum())
#encoding gender 
df['Gender']=pd.get_dummies(df['Gender'],drop_first=True)
#scaling data
sc=StandardScaler()
scaled_data=sc.fit_transform(df.drop('CustomerID',axis=1))
#finding optimal number of clusters using elbow method
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#finding optimal number of clusters using silhouette score
silhouette_scores=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    cluster_labels=kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data,cluster_labels))
plt.plot(range(2,11),silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
#applying kmeans with optimal number of clusters
kmeans=KMeans(n_clusters=5,random_state=42)
df['Cluster']=kmeans.fit_predict(scaled_data)
#pca for visualization
pca=PCA(n_components=2)
pca_data=pca.fit_transform(scaled_data)
#transforming pca data to dataframe for visualization
pca_df=pd.DataFrame(data=pca_data,columns=['PC1','PC2'])
pca_df['Cluster']=df['Cluster']
#visualizing clusters
plt.figure(figsize=(10,6))
for cluster in pca_df['Cluster'].unique():
    plt.scatter(pca_df[pca_df['Cluster']==cluster]['PC1'],pca_df[pca_df['Cluster']==cluster]['PC2'],label=f'Cluster {cluster}')
plt.title('Customer Segments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
# train a model using pca data 
model2=KMeans(n_clusters=5,random_state=42)
model2.fit(pca_data)


