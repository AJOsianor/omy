import pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
from flask_cors import CORS

# Load the dataset
data = pd.read_csv('customer_segmentation_data.csv')

# Select features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Create a KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the model to the data
kmeans.fit(features)

# Predict the clusters
data['Cluster'] = kmeans.predict(features)

# Plot the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()