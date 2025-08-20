#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Path to the data directory
data_path = 'data/'

# List all CSV files from April to September 2014
files = [
    'uber-raw-data-apr14.csv',
    'uber-raw-data-may14.csv',
    'uber-raw-data-jun14.csv',
    'uber-raw-data-jul14.csv',
    'uber-raw-data-aug14.csv',
    'uber-raw-data-sep14.csv'
]

# Read and concatenate all files into one DataFrame
dataframes = [pd.read_csv(os.path.join(data_path, file)) for file in files]
uber_2014 = pd.concat(dataframes, ignore_index=True)

print("Data loaded and combined successfully.")
uber_2014.head()
#%%
# Convert 'Date/Time' column to datetime objects
uber_2014['Date/Time'] = pd.to_datetime(uber_2014['Date/Time'], format='%m/%d/%Y %H:%M:%S')

# Set 'Date/Time' as the index
uber_2014.set_index('Date/Time', inplace=True)

# Resample the data by hour and count the number of trips (by counting any column, e.g., 'Base')
hourly_counts = uber_2014['Base'].resample('h').count()

# Convert the resulting Series to a DataFrame for easier use
uber_hourly = hourly_counts.to_frame(name='Count')

print("Data resampled to hourly trip counts.")
uber_hourly.head()
#%%
# Plot the hourly trip counts
uber_hourly.plot(figsize=(15, 7), title='Hourly Uber Trips in NYC (Apr-Sep 2014)', linewidth=1)
plt.ylabel('Number of Trips')
plt.xlabel('Date')
plt.show()
#%%
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
# The period is 24*7 for weekly seasonality within daily data
decomposition = seasonal_decompose(uber_hourly['Count'], model='additive', period=24*7)

# Plot the decomposition
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()

# Extract the trend component for the next step
trend = decomposition.trend.dropna()
#%%
# Define the cutoff date based on the trend analysis
cutoff_date = '2014-09-15'

# Split the data into training and testing sets
train_data = uber_hourly.loc[uber_hourly.index < cutoff_date]
test_data = uber_hourly.loc[uber_hourly.index >= cutoff_date]

# Plot the split
fig, ax = plt.subplots(figsize=(15, 7))
train_data.plot(ax=ax, label='Training Set', title='Train/Test Split')
test_data.plot(ax=ax, label='Test Set')
ax.axvline(pd.to_datetime(cutoff_date), color='black', linestyle='--')
plt.legend()
plt.show()
#%%
def create_lagged_features(data, window_size):
    """
    Creates lagged features for a time series dataset.
    """
    data = data.values
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Define the window size
window_size = 24

# Create features for the training set
X_train, y_train = create_lagged_features(train_data['Count'], window_size)

# To create features for the test set, we need the last `window_size` days from the training data
combined_for_test = pd.concat([train_data.tail(window_size), test_data])
X_test, y_test = create_lagged_features(combined_for_test['Count'], window_size)

print(f"Training data shape: X_train -> {X_train.shape}, y_train -> {y_train.shape}")
print(f"Test data shape: X_test -> {X_test.shape}, y_test -> {y_test.shape}")
#%%
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error

# Initialize TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 1. XGBoost Model
print("--- Training XGBoost Model ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 1.0]
}

# Setup and run GridSearchCV
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
xgb_grid_search.fit(X_train, y_train)

print(f"Best XGBoost parameters: {xgb_grid_search.best_params_}")

# Make predictions and evaluate
xgb_predictions = xgb_grid_search.best_estimator_.predict(X_test)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_predictions)
print(f"XGBoost MAPE: {xgb_mape:.2%}") # [cite: 516-517]
#%%
from sklearn.ensemble import RandomForestRegressor

# 2. Random Forest Model
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestRegressor(random_state=42)

# Parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup and run GridSearchCV
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")

# Make predictions and evaluate
rf_predictions = rf_grid_search.best_estimator_.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_predictions)
print(f"Random Forest MAPE: {rf_mape:.2%}") # [cite: 575]
#%%
from sklearn.ensemble import GradientBoostingRegressor

# 3. Gradient Boosting Model
print("\n--- Training Gradient Boosting Model ---")
gbr_model = GradientBoostingRegressor(random_state=42)

# Parameter grid for GBTR
gbr_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup and run GridSearchCV
gbr_grid_search = GridSearchCV(estimator=gbr_model, param_grid=gbr_param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
gbr_grid_search.fit(X_train, y_train)

print(f"Best GBTR parameters: {gbr_grid_search.best_params_}")

# Make predictions and evaluate
gbr_predictions = gbr_grid_search.best_estimator_.predict(X_test)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_predictions)
print(f"GBTR MAPE: {gbr_mape:.2%}") # [cite: 628]
#%%
# The document uses a simple reciprocal of MAPE for weights.
# Recalculating weights based on our results
mape_scores = np.array([xgb_mape, rf_mape, gbr_mape])
weights = 1 / mape_scores
weights /= np.sum(weights) # Normalize to sum to 1

print(f"Model Weights (XGB, RF, GBR): {np.round(weights, 3)}")

# Create ensemble predictions
ensemble_predictions = (weights[0] * xgb_predictions +
                        weights[1] * rf_predictions +
                        weights[2] * gbr_predictions)

# Evaluate ensemble
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_predictions)
print(f"\nEnsemble MAPE: {ensemble_mape:.2%}")


# Final Visualization
plt.figure(figsize=(18, 8))
plt.plot(test_data.index, y_test, label='Actual Trips (Test Set)', color='black', linewidth=2)
plt.plot(test_data.index, xgb_predictions, label=f'XGBoost Predictions (MAPE: {xgb_mape:.2%})', linestyle='--')
plt.plot(test_data.index, rf_predictions, label=f'Random Forest Predictions (MAPE: {rf_mape:.2%})', linestyle='--')
plt.plot(test_data.index, ensemble_predictions, label=f'Ensemble Predictions (MAPE: {ensemble_mape:.2%})', color='purple', linestyle='-')

plt.title('Uber Trip Forecasting: Model Predictions vs. Actual')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.legend()
plt.show()

# %%
