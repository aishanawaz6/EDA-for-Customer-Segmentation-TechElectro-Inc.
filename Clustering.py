# 5) Customer Segmentation: Apply clustering algorithms (e.g., K-means) to segment customers based on their buying patterns,
# demographics, and preferences.
#Tools: Python, Jupyter Notebook, Pandas, Matplotlib, Seaborn, Scikit-learn

import pandas as pd
from sklearn.cluster import KMeans
import warnings

customerS=pd.read_csv('TechElectro_Customer_Data_Preprocessed.csv')   #Reading saved file
customerS.head()

# To Suppress some warnings i was getting related to KMeans memory leak issue
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Selecting relevant features for clustering
features = ['Age', 'AnnualIncome (USD)', 'TotalPurchases','PreferredCategory_Electronics','PreferredCategory_Appliances']
customerFeatures=customerS[[feature for feature in features]]

# Determining the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(customerFeatures)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method to Find Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-cluster Sum of Squares (Inertia)')
plt.xticks(range(1, 11))
plt.show()


# Elbow Method above shows the optimal k value to be equal to 5
k = 5

# Applying K-means clustering 
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
customerS['Cluster'] = kmeans.fit_predict(customerFeatures)

# Visualizing the clusters using a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=customerS, x='Age', y='TotalPurchases', hue='Cluster', palette='dark', s=80)
plt.title('Customer Segmentation - K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Total Purchases')
plt.legend(title='Cluster', loc='upper right')
plt.show()
