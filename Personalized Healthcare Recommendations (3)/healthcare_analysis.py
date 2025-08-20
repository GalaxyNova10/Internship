#%%
import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('heart_disease_ci.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'heart_disease_ci.csv' not found. Please make sure the file is in the correct project folder.")

# Display the first 5 rows of the dataframe
print("\nFirst 5 rows of the data:")
print(df.head())

# Get a quick summary of the dataset (info about columns, data types, null values)
print("\nDataset Information:")
df.info()

# Get a statistical summary of the numerical columns
print("\nStatistical Summary:")
print(df.describe())
#%%
# Check for missing values in each column
print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with any missing values
# inplace=True modifies the DataFrame directly
df.dropna(inplace=True)

# Verify that missing values have been removed
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Display the new shape of the dataframe to see how many rows we have left
print(f"\nShape of the dataframe after dropping missing values: {df.shape}")
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Visualization (Corrected) ---

# Set up the plotting style
sns.set_style('whitegrid')

# Create a figure for our plots
plt.figure(figsize=(18, 12))

# --- Plot 1: Heart Disease Frequency ---
# This plot works fine with the original data
plt.subplot(2, 2, 1)
ax1 = sns.countplot(x='num', data=df, palette='pastel')
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.xlabel('Heart Disease')
plt.ylabel('Patient Count')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# --- Plot 2: Age Distribution by Heart Disease ---
# This plot also works fine with the original data
plt.subplot(2, 2, 2)
sns.kdeplot(df[df['num'] == 0]['age'], label='No Heart Disease', fill=True, color='blue')
sns.kdeplot(df[df['num'] == 1]['age'], label='Heart Disease', fill=True, color='red')
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()

# --- Plot 3: Cholesterol Distribution by Heart Disease ---
# This plot also works fine with the original data
plt.subplot(2, 2, 3)
sns.kdeplot(df[df['num'] == 0]['chol'], label='No Heart Disease', fill=True, color='blue')
sns.kdeplot(df[df['num'] == 1]['chol'], label='Heart Disease', fill=True, color='red')
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.xlabel('Cholesterol')
plt.ylabel('Density')
plt.legend()

# --- Plot 4: Correlation Heatmap ---
# NOW, we create a version of the data with all text columns converted to numbers
df_encoded = pd.get_dummies(df, drop_first=True)

# Create the heatmap using the newly encoded dataframe
plt.subplot(2, 2, 4)
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Features')


# Show all the plots
plt.tight_layout()
plt.show()
#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Feature Engineering and Preprocessing ---

# 1. Simplify the target variable 'num' to be binary (0 = No Disease, 1 = Disease)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# 2. Drop the original 'num' column and any other non-feature columns like 'id'
df_model = df.drop(['num', 'id', 'dataset'], axis=1)

# 3. Use get_dummies to encode all categorical features
df_model = pd.get_dummies(df_model, drop_first=True)

# 4. Define our features (X) and target (y)
X = df_model.drop('target', axis=1)
y = df_model['target']

# 5. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data successfully preprocessed and split.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")


# --- Model Building and Training ---

# 6. Initialize the Random Forest Classifier
# random_state=42 ensures we get the same results every time we run the code
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 7. Train the model on the training data
print("\nTraining the Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# 8. Make predictions on the test data
y_pred = rf_model.predict(X_test)

# 9. Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
#%%
from sklearn.ensemble import GradientBoostingClassifier

# --- Try a Different Model ---

# Initialize and train the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

print("\nTraining a Gradient Boosting model...")
gb_model.fit(X_train, y_train)
print("Training complete.")

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Calculate accuracy
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Model Accuracy: {gb_accuracy * 100:.2f}%")
#%%
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Technique 1: Train an XGBoost Model ---

# Initialize the XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

print("Training an XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Training complete.")

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")
#%%
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Tune the XGBoost Model ---

# Define a more detailed grid of parameters for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Initialize GridSearchCV
grid_search_xgb = GridSearchCV(
    estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid=xgb_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# Fit the grid search to the data
print("Starting hyperparameter tuning for XGBoost... (This may take a few minutes)")
grid_search_xgb.fit(X_train, y_train)

# Get the best model from the grid search
best_xgb_model = grid_search_xgb.best_estimator_

# Make predictions with the new, tuned model
y_pred_xgb_tuned = best_xgb_model.predict(X_test)

# Calculate the new accuracy
new_xgb_accuracy = accuracy_score(y_test, y_pred_xgb_tuned)

print(f"\nBest Parameters Found: {grid_search_xgb.best_params_}")
print(f"Original XGBoost Accuracy: 83.33%")
print(f"Tuned XGBoost Accuracy: {new_xgb_accuracy * 100:.2f}%")

#%%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Model Evaluation ---

# 1. Generate the Classification Report
print("--- Classification Report ---")
report = classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease'])
print(report)


# 2. Generate and Plot the Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Disease', 'Heart Disease'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %%
