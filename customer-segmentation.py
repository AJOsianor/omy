import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load your dataset
# Replace 'your_dataset.csv' with your actual dataset file
data = pd.read_csv("./customer_segmentation_data.csv")

# Data preprocessing
# Assuming the dataset has columns 'Age', 'Income', 'SpendingS_core'
features = data[['Age', 'Income', 'Spending_Score']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)

# Save the model and scaler for deployment
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predict the clusters
data['Cluster'] = kmeans.predict(scaled_features)

# Save the clustered data
data.to_csv('customer_segmentation_datapip.csv')

print("Model training and deployment files created successfully.")