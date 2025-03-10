import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
train_data = pd.read_csv('Train.csv')

# Display basic information about the dataset
print("Dataset Shape:", train_data.shape)
print("\nDataset Information:")
train_data.info()

# Check for missing values
print("\nMissing Values:")
print(train_data.isnull().sum())

# Exploratory Data Analysis
# Creating descriptive stats
print("\nDescriptive Statistics:")
print(train_data.describe())

# Data Preprocessing
def preprocess_data(df, is_training=True):
    """
    Preprocess the data with advanced techniques for more accurate predictions
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Target Variable Analysis (for training only)
    if is_training and 'Item_Outlet_Sales' in data.columns:
        print("\nTarget Variable Analysis:")
        print(f"Min Sales: {data['Item_Outlet_Sales'].min()}")
        print(f"Max Sales: {data['Item_Outlet_Sales'].max()}")
        print(f"Mean Sales: {data['Item_Outlet_Sales'].mean():.2f}")
        print(f"Median Sales: {data['Item_Outlet_Sales'].median():.2f}")
        print(f"Sales Skewness: {data['Item_Outlet_Sales'].skew():.2f}")
        
        # Log transform to handle skewness
        data['Item_Outlet_Sales_Log'] = np.log1p(data['Item_Outlet_Sales'])
        print(f"Log-transformed Sales Skewness: {data['Item_Outlet_Sales_Log'].skew():.2f}")
    
    # Advanced Missing Value Handling
    # Item_Weight
    missing_weight = data['Item_Weight'].isnull().sum()
    if missing_weight > 0:
        # Group by Item_Identifier and fill with mean
        item_avg_weight = data.groupby('Item_Identifier')['Item_Weight'].transform('mean')
        data['Item_Weight'].fillna(item_avg_weight, inplace=True)
        
        # If still missing, use global mean
        still_missing = data['Item_Weight'].isnull().sum()
        if still_missing > 0:
            data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
        
        print(f"Filled {missing_weight} missing Item_Weight values")
    
    # Outlet_Size - using a more sophisticated approach based on Outlet_Type
    missing_size = data['Outlet_Size'].isnull().sum()
    if missing_size > 0:
        # Create a mapping of common Outlet_Type to Outlet_Size
        size_mode_by_type = data.groupby('Outlet_Type')['Outlet_Size'].apply(
            lambda x: x.mode()[0] if not x.mode().empty else "Medium")
        
        # Fill missing with the mode based on Outlet_Type
        for outlet_type, size_mode in size_mode_by_type.items():
            type_indices = (data['Outlet_Type'] == outlet_type) & (data['Outlet_Size'].isnull())
            data.loc[type_indices, 'Outlet_Size'] = size_mode
        
        # If still missing, use global mode
        still_missing = data['Outlet_Size'].isnull().sum()
        if still_missing > 0:
            global_mode = data['Outlet_Size'].mode()[0]
            data['Outlet_Size'].fillna(global_mode, inplace=True)
        
        print(f"Filled {missing_size} missing Outlet_Size values")
    
    # Data Cleaning and Feature Standardization
    # Standardize Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
    
    # Handle Item_Visibility = 0 (likely missing data)
    zero_visibility = (data['Item_Visibility'] == 0).sum()
    if zero_visibility > 0:
        # Calculate mean visibility by Item_Type
        visibility_by_type = data.groupby('Item_Type')['Item_Visibility'].transform(
            lambda x: x.replace(0, np.nan).mean())
        
        # Replace 0s with calculated means
        data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = \
            visibility_by_type[data['Item_Visibility'] == 0]
        
        # Check if any NaNs introduced and fix them
        if data['Item_Visibility'].isnull().sum() > 0:
            data['Item_Visibility'].fillna(data['Item_Visibility'].mean(), inplace=True)
        
        print(f"Replaced {zero_visibility} zero visibility values")
    
    # Feature Engineering
    # 1. Create outlet age feature
    data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
    
    # 2. Create Item_Type Categories (more granular than original)
    food_categories = {
        'Breads': 'Starchy Foods',
        'Breakfast': 'Starchy Foods',
        'Baking Goods': 'Starchy Foods',
        'Fruits and Vegetables': 'Fresh Foods',
        'Meat': 'Fresh Foods',
        'Seafood': 'Fresh Foods',
        'Dairy': 'Fresh Foods',
        'Soft Drinks': 'Beverages',
        'Hard Drinks': 'Beverages',
        'Household': 'Non-Consumable',
        'Health and Hygiene': 'Non-Consumable',
        'Canned': 'Processed Foods',
        'Frozen Foods': 'Processed Foods',
        'Snack Foods': 'Processed Foods',
        'Others': 'Others'
    }
    data['Item_Category'] = data['Item_Type'].map(food_categories)
    
    # 3. Item identifier first 2 characters indicate item category
    data['Item_Type_Code'] = data['Item_Identifier'].apply(lambda x: x[:2])
    
    # 4. MRP Buckets for non-linear relationships
    data['Item_MRP_Bucket'] = pd.qcut(data['Item_MRP'], 4, labels=['Budget', 'Economy', 'Premium', 'Luxury'])
    
    # 5. Interaction Features
    data['Price_per_Weight'] = data['Item_MRP'] / data['Item_Weight']
    data['Visibility_Ratio'] = data['Item_Visibility'] / data.groupby('Item_Type')['Item_Visibility'].transform('mean')
    data['Outlet_Type_Age'] = data['Outlet_Type'] + "_" + data['Outlet_Age'].astype(str)
    
    # 6. Non-linear transformations
    data['Item_MRP_Squared'] = data['Item_MRP'] ** 2
    data['Outlet_Age_Squared'] = data['Outlet_Age'] ** 2
    data['Item_Visibility_Sqrt'] = np.sqrt(data['Item_Visibility'])
    
    # Drop ID columns
    drop_cols = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year']
    if not is_training:
        # If this is test data, we don't have the target
        drop_cols = drop_cols
    else:
        # If training data, keep original sales (we'll use log transformed for training)
        drop_cols = drop_cols
    
    return data

# Process the data
processed_data = preprocess_data(train_data)

# Display the processed data
print("\nProcessed Data Sample:")
print(processed_data.head())
print("\nProcessed Data Shape:", processed_data.shape)

# Data Visualization
print("\nCreating visualizations...")

# Let's create some visualizations to understand the data better
plt.figure(figsize=(12, 8))

# 1. Sales Distribution
plt.subplot(2, 2, 1)
sns.histplot(processed_data['Item_Outlet_Sales'], kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')

# 2. Log-transformed Sales Distribution
plt.subplot(2, 2, 2)
sns.histplot(processed_data['Item_Outlet_Sales_Log'], kde=True)
plt.title('Distribution of Log-transformed Sales')
plt.xlabel('Log(Sales)')
plt.ylabel('Frequency')

# 3. Sales by Outlet Type
plt.subplot(2, 2, 3)
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=processed_data)
plt.title('Sales by Outlet Type')
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Sales by Item Category
plt.subplot(2, 2, 4)
sns.boxplot(x='Item_Category', y='Item_Outlet_Sales', data=processed_data)
plt.title('Sales by Item Category')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('sales_analysis.png')
plt.close()

# Correlation Analysis
plt.figure(figsize=(12, 10))
numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
correlation = processed_data[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Data Split - Using log-transformed target for better model performance
X = processed_data.drop(['Item_Outlet_Sales', 'Item_Outlet_Sales_Log'], axis=1)
y_log = processed_data['Item_Outlet_Sales_Log']
y_original = processed_data['Item_Outlet_Sales']

print("\nFeatures:", X.columns.tolist())

# Split data with stratification
X_train, X_test, y_log_train, y_log_test, y_train, y_test = train_test_split(
    X, y_log, y_original, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumerical features:", numerical_cols)
print("Categorical features:", categorical_cols)

# Create enhanced preprocessor with advanced transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])

# Function to evaluate models with detailed metrics
def evaluate_model(model, X_train, X_test, y_train, y_test, y_orig_test, model_name, is_log_transformed=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # If we used log transformation, convert predictions back to original scale
    if is_log_transformed:
        y_pred_orig = np.expm1(y_pred)
    else:
        y_pred_orig = y_pred
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Original scale metrics
    orig_mse = mean_squared_error(y_orig_test, y_pred_orig)
    orig_rmse = np.sqrt(orig_mse)
    orig_r2 = r2_score(y_orig_test, y_pred_orig)
    orig_mae = mean_absolute_error(y_orig_test, y_pred_orig)
    
    # Cross-validation score for log-transformed target
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    
    print(f"\n{model_name} Results:")
    print(f"Log Scale - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, CV RMSE: {cv_rmse:.4f}")
    print(f"Original Scale - RMSE: {orig_rmse:.2f}, R²: {orig_r2:.4f}, MAE: {orig_mae:.2f}")
    
    return {
        'model': model,
        'log_rmse': rmse,
        'log_r2': r2,
        'log_mae': mae,
        'log_cv_rmse': cv_rmse,
        'orig_rmse': orig_rmse,
        'orig_r2': orig_r2,
        'orig_mae': orig_mae,
    }

# Create full pipeline with preprocessing
def create_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Define models with optimized parameters
models = {
    'Huber Regression': HuberRegressor(epsilon=1.35, alpha=0.0015, max_iter=2000),
    'Ridge Regression': Ridge(alpha=0.8, solver='saga', max_iter=2000),
    'XGBoost': XGBRegressor(
        learning_rate=0.05, 
        n_estimators=300, 
        max_depth=5, 
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3
    ),
    'LightGBM': LGBMRegressor(
        learning_rate=0.03,
        n_estimators=500,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20
    ),
    'CatBoost': CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        verbose=0
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=250,
        max_depth=5,
        subsample=0.8,
        max_features=0.8
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        early_stopping=True
    )
}

# Train and evaluate models
results = {}
print("\n\n=========== MODEL EVALUATION ===========")

for name, model in models.items():
    pipeline = create_pipeline(model)
    result = evaluate_model(pipeline, X_train, X_test, y_log_train, y_log_test, y_test, name)
    results[name] = result

# Create stacked model from the best performers
print("\n\n=========== CREATING ENSEMBLE MODELS ===========")

# Sort models by performance (original scale R²)
sorted_models = sorted(results.items(), key=lambda x: x[1]['orig_r2'], reverse=True)
# Extract base model from pipeline
def extract_base_model(pipeline):
    """Extract the final estimator from a pipeline"""
    return pipeline.named_steps['model']
top_models = [model for name, model in sorted_models[:4]]
top_model_names = [name for name, _ in sorted_models[:4]]

print(f"Top performing models: {', '.join(top_model_names)}")

# Create base estimators list for ensembles
base_estimators = [(name, extract_base_model(results[name]['model'])) for name in top_model_names]

# Create a voting regressor with proper pipeline
voting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('voting', VotingRegressor(estimators=base_estimators))
])

# Evaluate voting regressor
voting_result = evaluate_model(
    voting_pipeline, X_train, X_test, y_log_train, y_log_test, y_test, 
    "Voting Ensemble"
)
results["Voting Ensemble"] = voting_result

# Create a stacking regressor with proper pipeline
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.1)
    ))
])

# Evaluate stacking regressor
stacking_result = evaluate_model(
    stacking_pipeline, X_train, X_test, y_log_train, y_log_test, y_test, 
    "Stacking Ensemble"
)
results["Stacking Ensemble"] = stacking_result

# Find the best model
best_model_name = max(results, key=lambda k: results[k]['orig_r2'])
best_model_info = results[best_model_name]
print(f"\nBest model: {best_model_name}")
print(f"R² on original scale: {best_model_info['orig_r2']:.4f}")
print(f"RMSE on original scale: {best_model_info['orig_rmse']:.2f}")

# Calculate 'accuracy' as 1 - normalized RMSE (as a percentage)
# Normalizing by the range of the target variable
y_range = y_test.max() - y_test.min()
normalized_rmse = best_model_info['orig_rmse'] / y_range
accuracy = (1 - normalized_rmse) * 100

print(f"\nApproximate model accuracy: {accuracy:.2f}%")

# Create visualization of model performance
plt.figure(figsize=(15, 10))

# Get metrics for plotting
model_names = list(results.keys())
orig_r2_values = [results[name]['orig_r2'] for name in model_names]
orig_rmse_values = [results[name]['orig_rmse'] for name in model_names]

# Sort by R² for better visualization
sorted_indices = np.argsort(orig_r2_values)[::-1]  # Sort in descending order
sorted_names = [model_names[i] for i in sorted_indices]
sorted_r2 = [orig_r2_values[i] for i in sorted_indices]
sorted_rmse = [orig_rmse_values[i] for i in sorted_indices]

# Plot R²
plt.subplot(2, 1, 1)
bars = plt.barh(sorted_names, sorted_r2, color='skyblue')
plt.xlabel('R² (higher is better)')
plt.title('Model Comparison - R²')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim(0, 1)

# Highlight best model
best_index = sorted_names.index(best_model_name)
bars[best_index].set_color('green')

# Plot RMSE
plt.subplot(2, 1, 2)
bars = plt.barh(sorted_names, sorted_rmse, color='salmon')
plt.xlabel('RMSE (lower is better)')
plt.title('Model Comparison - RMSE')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Highlight best model
bars[best_index].set_color('green')

plt.tight_layout()
plt.savefig('model_comparison_enhanced.png')

# Feature importance analysis for the best model
if best_model_name != "Voting Ensemble" and best_model_name != "Stacking Ensemble":
    try:
        best_pipeline = results[best_model_name]['model']
        
        # For tree-based models
        if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
            # Get feature importances
            feature_importances = best_pipeline.named_steps['model'].feature_importances_
            
            # Get feature names
            try:
                feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f'Feature {i}' for i in range(len(feature_importances))]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
            
            # Plot importance
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importances - {best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            print("\nTop 5 Important Features:")
            print(importance_df.head(5))
        
        # For linear models
        elif hasattr(best_pipeline.named_steps['model'], 'coef_'):
            # Get coefficients
            coefficients = best_pipeline.named_steps['model'].coef_
            
            # Get feature names
            try:
                feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f'Feature {i}' for i in range(len(coefficients))]
            
            # Create coefficient dataframe
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(15)
            
            # Plot coefficients
            plt.figure(figsize=(12, 8))
            plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='lightcoral')
            plt.xlabel('Coefficient Value')
            plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_coefficients.png')
            
            print("\nTop 5 Features by Coefficient Magnitude:")
            print(coef_df[['Feature', 'Coefficient']].head(5))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

# Predictions vs Actual for the best model
best_model = results[best_model_name]['model']
if best_model_name in ["Voting Ensemble", "Stacking Ensemble"]:
    y_pred = best_model.predict(X_test)
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
    y_pred_orig = np.expm1(y_pred)
else:
    y_pred = best_model.predict(X_test)
    y_pred_orig = np.expm1(y_pred)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_orig, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'Actual vs Predicted Sales - {best_model_name}')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png')

# Residual Analysis
residuals = y_test - y_pred_orig
plt.figure(figsize=(10, 8))
plt.scatter(y_pred_orig, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred_orig.min(), xmax=y_pred_orig.max(), colors='r', linestyles='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.tight_layout()
plt.savefig('residual_analysis.png')

# Save the final model
print(f"\nSaving the best model: {best_model_name}")
final_model = results[best_model_name]['model']

# Create a dictionary with all relevant information
model_info = {
    'model': final_model,
    'model_name': best_model_name,
    'performance': {
        'R2': results[best_model_name]['orig_r2'],
        'RMSE': results[best_model_name]['orig_rmse'],
        'MAE': results[best_model_name]['orig_mae'],
        'Approximate_Accuracy': accuracy
    },
    'is_log_transformed': True,
    'all_results': results,
    'feature_columns': X.columns.tolist(),
    'preprocessing_function': preprocess_data,
}

# Save the model to disk
with open('enhanced_sales_model.pkl', 'wb') as file:
    pickle.dump(model_info, file)

print("\nModel saved as 'enhanced_sales_model.pkl'")
print("\n=========== MODEL TRAINING COMPLETED ===========")

# Function to make predictions with the saved model
def predict_sales(item_data, model_path='enhanced_sales_model.pkl'):
    """
    Make predictions with the saved model
    
    Parameters:
    -----------
    item_data : dict or DataFrame
        Data containing the features needed for prediction
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    float
        Predicted sales value
    """
    # Load the model
    with open(model_path, 'rb') as file:
        model_info = pickle.load(file)
    
    model = model_info['model']
    preprocess_func = model_info.get('preprocessing_function', None)
    is_log_transformed = model_info.get('is_log_transformed', True)
    
    # If input is a dictionary, convert to DataFrame
    if isinstance(item_data, dict):
        item_data = pd.DataFrame([item_data])
    
    # Preprocess the data
    if preprocess_func:
        item_data = preprocess_func(item_data, is_training=False)
    
    # Make prediction
    prediction = model.predict(item_data)
    
    # If model was trained on log-transformed target, convert back
    if is_log_transformed:
        prediction = np.expm1(prediction)
    
    return prediction[0] if len(prediction) == 1 else prediction

# Example usage
print("\n=========== PREDICTION EXAMPLE ===========")
print("To use the model for predictions:")
print("""
# Example:
sample_item = {
    'Item_Identifier': 'FDA15',
    'Item_Weight': 9.3,
    'Item_Fat_Content': 'Low Fat',
    'Item_Visibility': 0.016,
    'Item_Type': 'Dairy',
    'Item_MRP': 249.81,
    'Outlet_Identifier': 'OUT049',
    'Outlet_Establishment_Year': 1999,
    'Outlet_Size': 'Medium',
    'Outlet_Location_Type': 'Tier 1',
    'Outlet_Type': 'Supermarket Type1'
}

# Predict sales
predicted_sales = predict_sales(sample_item)
print(f"Predicted Sales: ${predicted_sales:.2f}")
""")