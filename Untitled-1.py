# %%
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("customer_segmentation_data.csv")

# Display the first few rows of the dataset
df.head()

# %%
# Display basic information about the dataset
df.info()

# %%
# Display descriptive statistics
df.describe()

# %%
# Check for missing values
df.isnull().sum()

# %%
# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# %%
# Income vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='spending_score', data=df, hue='gender')
plt.title('Income vs Spending Score')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

# %%
# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# %%
# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the data for modeling
X = df[['age', 'income', 'membership_years', 'purchase_frequency', 'last_purchase_amount']]
y = df['spending_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# %%
sns.pairplot(df)

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(x='preferred_category', y='income', data=df)
plt.title('Income Distribution by Preferred Category')
plt.xlabel('Preferred Category')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='gender', y='spending_score', data=data)
plt.title('Spending Score Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()

# %%
print("\nCinsiyet ve Tercih Edilen Kategoriye Göre Gelir Ortalamaları:")
pivot_table = df.pivot_table(values='income', index='gender', columns='preferred_category', aggfunc='mean')
print(pivot_table)

# %%
df.info()

# %%
df_clean = (df)

# %%
df_clean['membership_years'].value_counts()

# %%
df_clean['purchase_frequency'].value_counts()

# %%
df_clean['preferred_category'].value_counts()

# %%
df_clean['last_purchase_amount'].value_counts()

# %%
df_clean['spending_score'].value_counts()

# %%
df_clean['income'].value_counts()

# %%
df_clean['gender'].value_counts()

# %%
df_clean['age'].value_counts()

# %%
df_clean['id'].value_counts()

# %%
from sklearn.ensemble import RandomForestClassifier

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# %%
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# %%
model.fit(X_train, y_train)

# %%
print(y_train.value_counts())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of the target variable (y_train)
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# %%
#scaler = MinMaxScaler((0, 1))
#df_rfm = scaler.fit_transform(rfm.loc[:,['log_Recency','log_Frequency','log_Monetary']])
rfm = (df)
df_rfm = (df)
df_rfm=np.array(rfm.loc[:,['spending_score','income','age']])

# %%
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))

visualizer.fit(df_rfm) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure
plt.show()

# %%
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['age', 'income']])

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='income',
hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()

# %%
km_mdl = KMeans(n_clusters=4, random_state=32)

km = km_mdl.fit_predict(df_rfm)

# %%
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df[['age', 'income']])
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# %%
plt.figure(figsize=(12, 8))
sns.lineplot(x='membership_years', y='spending_score', data=df, ci=None)
plt.title('Spending Score Over Membership Years')
plt.xlabel('Membership Years')
plt.ylabel('Spending Score')
plt.show()

# %%
np.unique(km)

# %%
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_rfm[km == 0,0],df_rfm[km == 0,1],df_rfm[km == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(df_rfm[km == 1,0],df_rfm[km == 1,1],df_rfm[km == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(df_rfm[km == 2,0],df_rfm[km == 2,1],df_rfm[km == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(df_rfm[km == 3,0],df_rfm[km == 3,1],df_rfm[km == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
ax.set_xlabel('age')
ax.set_ylabel('income')
ax.set_zlabel('spending_score')
ax.legend()
plt.show()

# %%
colors = ['blue', 'orange', 'green', '#D12B60']

# Create a 3D scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=df_rfm[km == 0,0], y=df_rfm[km == 0,1], z=df_rfm[km == 0,2], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=df_rfm[km == 1,0], y=df_rfm[km == 1,1], z=df_rfm[km == 1,2], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=df_rfm[km == 2,0], y=df_rfm[km == 2,1], z=df_rfm[km == 2,2],
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))
fig.add_trace(go.Scatter3d(x=df_rfm[km == 3,0], y=df_rfm[km == 3,1], z=df_rfm[km == 3,2],
                           mode='markers', marker=dict(color=colors[3], size=5, opacity=0.4), name='Cluster 3'))

fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='age'),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='income'),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='spending_score'),
    ),
    width=900,
    height=800
)
fig.show()

# %%
r_quarters = rfm['age'].quantile(q=[0.0, 0.25,0.5,0.75, 1]).to_list()
f_quarters = rfm['income'].quantile(q=[0.0, 0.25,0.5,0.75, 1]).to_list()
m_quarters = rfm['spending_score'].quantile(q=[0.0, 0.25,0.5,0.75, 1]).to_list()
quartile_spread = pd.DataFrame(list(zip(r_quarters, f_quarters, m_quarters)), 
                      columns=['Q_age','Q_income', 'Q_spending_score'],
                     index = ['min', 'first_part','second_part','third_part', 'max'])
quartile_spread

# %%
rfm['r_score'] = pd.cut(rfm['age'], bins=r_quarters, labels=[4,3,2,1],include_lowest=True)
rfm['f_score'] = pd.cut(rfm['income'], bins=f_quarters, labels=[1,2,3,4],include_lowest=True)
rfm['m_score'] = pd.cut(rfm['spending_score'], bins=m_quarters, labels=[1,2,3,4],include_lowest=True)
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

# %%
ax = rfm['rfm_score'].value_counts().plot(kind='bar', figsize=(15, 5), fontsize=12)
ax.set_xlabel("RFM Score based on Quartile partitions", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.show()

# %%
rfm['KMean_Segment'] = km

# %%
rfm[["KMean_Segment","age","income","spending_score"]].groupby(["KMean_Segment"]).agg(["mean","count"])

# %%
rfm['KMean_Segment'] = rfm['KMean_Segment'].astype(str)
km_map = {
    r'0': 'Low priority',
    r'1': 'Loyal',
    r'3': 'New',
    r'2': 'Core',
}

rfm['KMeans_seg_trans'] = rfm['KMean_Segment'].replace(km_map, regex=True)

# %%
rfm.head()

# %%
cluster_mean = df.groupby('gender')[['last_purchase_amount', 'purchase_frequency', 'income']].mean().reset_index()

fig, axes = plt.subplots(nrows=3, figsize=(10, 8))

sns.barplot(cluster_mean, x='gender', y='last_purchase_amount', ax=axes[0])
sns.barplot(cluster_mean, x='gender', y='purchase_frequency', ax=axes[1])
sns.barplot(cluster_mean, x='gender', y='income', ax=axes[2])

plt.tight_layout()

# %%
import pandas as pd
import plotly.express as px
from summarytools import dfSummary

# %%
 dfSummary(df)

# %%


# %%
SSE=[]
for cluster in range(1,8):
    kmeans=KMeans(n_clusters=cluster,init='k-means++')
    kmeans.fit(df_rfm)
    SSE.append(kmeans.inertia_)
frame=pd.DataFrame({'cluster':range(1,8),'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['cluster'],frame['SSE'],marker='o')
plt.xlabel("num of cluster")
plt.ylabel("inertia")

# %%
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Cluster', hue=col, data=df, palette='viridis')
    plt.title(f'{col} Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# %%

plt.figure(figsize=(12, 8))
sns.scatterplot(x='age', y='income', hue='Cluster', data=df, palette='viridis')
plt.title('Age vs Income by Cluster')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Harcama skoru ve son satın alma miktarı arasındaki ilişkiyi inceliyorum
plt.figure(figsize=(12, 8))
sns.scatterplot(x='last_purchase_amount', y='spending_score', hue='Cluster', data=df, palette='viridis')
plt.title('Last Purchase Amount vs Spending Score by Cluster')
plt.xlabel('Last Purchase Amount')
plt.ylabel('Spending Score')
plt.show()

# %%
kmeans=KMeans(n_clusters=10,init='k-means++')
kmeans.fit(df_rfm)

# %%
'''
app = Flask(__myfinalproject__)

# Load your pre-trained clustering model
with open('Customer Segmention.pkl', 'rb') as file:
    clustering_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Get input values from the form
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary_value = float(request.form['monetary_value'])

        # Make predictions using the loaded clustering model
        cluster_value = clustering_model.predict([[recency, frequency, monetary_value]])[0]

        # Render the result in the HTML template
        return render_template('index.html', cluster=cluster_value)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
'''

# %%



