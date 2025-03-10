from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the model
MODEL_PATH = './model.pkl'

# Define the preprocessing function here to avoid the attribute error
def preprocess_data(data, is_training=False):
    """
    Preprocess data for the model by creating necessary derived features
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Make a copy to avoid modifying the original
    processed = data.copy()
    
    # Calculate derived features
    current_year = datetime.now().year
    
    # Basic features
    processed['Outlet_Age'] = current_year - processed['Outlet_Establishment_Year'].astype(float)
    processed['Outlet_Age_Squared'] = processed['Outlet_Age'] ** 2
    processed['Item_Visibility_Sqrt'] = np.sqrt(processed['Item_Visibility'].astype(float))
    processed['Item_MRP_Squared'] = processed['Item_MRP'].astype(float) ** 2
    
    # Create MRP buckets
    # Converting to categorical might not work well with some models, so use numeric bins
    mrp = processed['Item_MRP'].astype(float)
    bins = [0, 50, 100, 150, 200, float('inf')]
    labels = [0, 1, 2, 3, 4]  # Numeric labels
    processed['Item_MRP_Bucket'] = pd.cut(mrp, bins=bins, labels=labels, right=False).astype(float)
    
    # Item type coding - simple mapping
    item_types = {
        'Dairy': 0, 'Soft Drinks': 1, 'Meat': 2, 'Fruits and Vegetables': 3,
        'Household': 4, 'Baking Goods': 5, 'Snack Foods': 6, 'Frozen Foods': 7,
        'Breakfast': 8, 'Health and Hygiene': 9, 'Hard Drinks': 10, 'Canned': 11,
        'Breads': 12, 'Starchy Foods': 13, 'Others': 14, 'Seafood': 15
    }
    processed['Item_Type_Code'] = processed['Item_Type'].map(item_types).fillna(0).astype(float)
    
    # Extract item category from identifier
    processed['Item_Category'] = processed['Item_Identifier'].str[:2]
    
    # Price per weight with fallback for missing weights
    weight = processed['Item_Weight'].astype(float)
    mean_weight = weight.mean()
    if pd.isna(mean_weight):  # If all weights are NA
        mean_weight = 10.0  # Default fallback
    processed['Price_per_Weight'] = processed['Item_MRP'].astype(float) / weight.fillna(mean_weight)
    
    # Visibility ratio (simplified version that works for single predictions)
    processed['Visibility_Ratio'] = processed['Item_Visibility'].astype(float) / 0.05  # 0.05 as average visibility
    
    # Outlet type age interaction
    outlet_type_map = {
        'Grocery Store': 0, 
        'Supermarket Type1': 1, 
        'Supermarket Type2': 2, 
        'Supermarket Type3': 3
    }
    outlet_numeric = processed['Outlet_Type'].map(outlet_type_map).fillna(0).astype(float)
    processed['Outlet_Type_Age'] = outlet_numeric * processed['Outlet_Age']
    
    # Ensure all columns are properly formatted for the model
    for col in processed.columns:
        if processed[col].dtype == 'object' and col not in ['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 
                                                         'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
                                                         'Item_Fat_Content', 'Item_Category']:
            try:
                processed[col] = processed[col].astype(float)
            except:
                pass  # Keep as object if conversion fails
    
    return processed

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model_info = pickle.load(file)
        return model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_info = load_model()

# Function to make predictions
def predict_sales(item_data):
    """
    Make predictions with the loaded model
    
    Parameters:
    -----------
    item_data : dict or DataFrame
        Data containing the features needed for prediction
        
    Returns:
    --------
    float
        Predicted sales value
    """
    if model_info is None:
        return {"error": "Model not loaded"}
    
    model = model_info['model']
    preprocess_func = model_info.get('preprocessing_function', preprocess_data)
    is_log_transformed = model_info.get('is_log_transformed', True)
    
    # If input is a dictionary, convert to DataFrame
    if isinstance(item_data, dict):
        item_data = pd.DataFrame([item_data])
    
    # Preprocess the data
    if preprocess_func:
        item_data = preprocess_func(item_data, is_training=False)
    
    # Make prediction
    try:
        prediction = model.predict(item_data)
        
        # If model was trained on log-transformed target, convert back
        if is_log_transformed:
            prediction = np.expm1(prediction)
        
        return {"prediction": float(prediction[0]) if len(prediction) == 1 else prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Log predictions
def log_prediction(input_data, prediction_result, success=True):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "prediction": prediction_result,
        "success": success
    }
    
    log_file = "prediction_log.json"
    
    # Create or append to log file
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error logging prediction: {e}")

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict-page')
def predict_form():
    """Render the prediction form page"""
    # Get model features to display on the form
    features = model_info.get('feature_columns', []) if model_info else []
    return render_template('predict.html', features=features)

@app.route('/model-info-page')
def model_info_page():
    """Render the model info page"""
    if model_info is None:
        return render_template('model-info.html', model_loaded=False)
    
    # Extract relevant information from model_info
    info = {
        "model_name": model_info.get('model_name', 'Unknown'),
        "performance": model_info.get('performance', {}),
        "feature_columns": model_info.get('feature_columns', []),
        "is_log_transformed": model_info.get('is_log_transformed', True)
    }
    
    return render_template('model-info.html', model_loaded=True, model_info=info)

@app.route('/model-info')
def model_info_route():
    """Model information API route"""
    if model_info is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Extract relevant information from model_info
    info = {
        "model_name": model_info.get('model_name', 'Unknown'),
        "performance": model_info.get('performance', {}),
        "feature_columns": model_info.get('feature_columns', []),
        "is_log_transformed": model_info.get('is_log_transformed', True)
    }
    
    # Format for API response
    return jsonify({
        "status": "success",
        "model_info": info
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Process both form and JSON submissions
        if request.is_json:
            data = request.get_json()
        else:
            # Convert form data to dictionary
            data = {}
            for field in request.form:
                try:
                    value = request.form[field]
                    if field in ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']:
                        value = float(value)
                    data[field] = value
                except:
                    data[field] = request.form[field]
        
        # Make prediction
        result = predict_sales(data)
        
        # Check for errors
        if "error" in result:
            print(f"Prediction error: {result['error']}")  # Log error
            log_prediction(data, result, success=False)
            return jsonify({"status": "error", "error": result["error"]}), 400
        
        # Log successful prediction
        log_prediction(data, result, success=True)
        
        # Return prediction as JSON
        return jsonify({
            "status": "success",
            "item_data": data,
            "prediction": result["prediction"],
            "prediction_formatted": f"${result['prediction']:.2f}"
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Exception in predict route: {error_msg}")
        print(f"Stack trace: {stack_trace}")
        return jsonify({"status": "error", "error": f"Server error: {error_msg}"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction API endpoint"""
    # Check if request has JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get data from request
    data = request.get_json()
    
    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of item data"}), 400
    
    # Process each item
    results = []
    for item in data:
        prediction = predict_sales(item)
        if "error" in prediction:
            results.append({
                "item_data": item,
                "error": prediction["error"]
            })
        else:
            results.append({
                "item_data": item,
                "prediction": prediction["prediction"],
                "prediction_formatted": f"${prediction['prediction']:.2f}"
            })
    
    # Return batch results
    return jsonify({
        "status": "success",
        "predictions": results
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Check if model is loaded
    if model_info is None:
        print(f"Warning: Could not load model from {MODEL_PATH}")
        print("The application will run, but predictions will not work.")
    else:
        print(f"Model loaded: {model_info.get('model_name', 'Unknown')}")
        print(f"Model performance: R² = {model_info.get('performance', {}).get('R2', 'Unknown')}")
    
    # Use a different port to avoid the conflict
    port = 5001  # Changed from 5000 to 5001
    
    print(f"Starting server on port {port}")
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)