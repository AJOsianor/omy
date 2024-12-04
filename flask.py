from flask import Flask, jsonify, request
import pandas as pd

# Initialize the Flask application
app = Flask('_kmeans_model_')

# Load the dataset
data_path = "customer_segmentation_data.csv"
df = pd.read_csv(data_path)

@app.route('/')
def home():
    return "Welcome to the Customer Segmentation API!"

@app.route('/dataset/info', methods=['GET'])
def dataset_info():
    """Return basic information about the dataset."""
    info = {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return jsonify(info)

@app.route('/dataset/head', methods=['GET'])
def dataset_head():
    """Return the first few rows of the dataset."""
    num_rows = request.args.get('rows', default=5, type=int)
    return df.head(num_rows).to_json(orient='records')

@app.route('/dataset/stats', methods=['GET'])
def dataset_stats():
    """Return basic statistics of numerical columns."""
    stats = df.describe().to_dict()
    return jsonify(stats)

# Run the app
if __name__ == '_kmeans_model_':
    app.run(debug=True)





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

if __name__ == '__my_final_project__':
    app.run(debug=True)
'''