import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle


print("Loading data...")
# Load and prepare data
data = pd.read_csv('Datasets\coin_Bitcoin.csv')
data.index = pd.to_datetime(data['Date'])
new_data = data.drop(['SNo', 'Symbol', 'Name', 'Date'], axis=1)

X = new_data[['High', 'Low', 'Open', 'Volume', 'Marketcap']]
y = new_data['Close']

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Quick feature importance analysis (using smaller sample for speed)
print("Analyzing feature importance...")
sample_size = min(1000, len(X))  # Use max 1000 samples for feature importance
X_sample = X.head(sample_size)
y_sample = y.head(sample_size)

model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced trees
model.fit(X_sample, y_sample)

feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

all_features = ['High', 'Low', 'Open', 'Volume', 'Marketcap']
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {all_features[indices[f]]}: {feature_importances[indices[f]]:.4f}")

# Train-test split and scaling
print("Preparing data for training...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models (only fast models)
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'SVR': SVR(kernel='linear') 
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print(f"Training {type(model).__name__}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# Train and evaluate models
print("Training models...")
results = {}
trained_models = {}
predictions = {}

for model_name, model in models.items():
    mse, r2, y_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    results[model_name] = {'MSE': mse, 'R^2': r2}
    trained_models[model_name] = model
    predictions[model_name] = y_pred

# Print results
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R^2 Score: {metrics['R^2']:.4f}")
    print("-" * 30)

# Save models and scaler
print("Saving models...")
joblib.dump(trained_models['Linear Regression'], 'linear_regression_model.pkl')
joblib.dump(trained_models['Decision Tree'], 'tree_regression_model.pkl')
joblib.dump(trained_models['SVR'], 'svr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for later use
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(all_features, f)

# Save sample data for reference
sample_data = {
    'feature_stats': X.describe().to_dict(),
    'target_stats': y.describe().to_dict()
}
with open('data_stats.pkl', 'wb') as f:
    pickle.dump(sample_data, f)

print("\n" + "="*50)
print("SUCCESS! Models saved successfully!")

