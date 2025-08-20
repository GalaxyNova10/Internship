#%%
import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('Life Expectancy Data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Life Expectancy Data.csv' not found. Please make sure the file is in the correct project folder.")

# The column names have leading/trailing spaces, let's clean them
df.columns = df.columns.str.strip()

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
# --- Data Cleaning: Handle Missing Values (Corrected) ---

# First, let's see which columns have missing data and how much
print("Missing values before cleaning:")
print(df.isnull().sum())

# Impute missing values with the mean of their respective columns
# We will iterate through each column and fill NaNs if any exist
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]): # Only fill numeric columns
            # This is the corrected syntax
            df[col] = df[col].fillna(df[col].mean())

# Verify that all missing values have been handled
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData cleaning complete. All missing numerical values have been imputed.")

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# --- Exploratory Data Analysis (EDA) ---

# Set the plotting style
sns.set_style('whitegrid')

# Create a figure for our plots
plt.figure(figsize=(18, 14))
plt.suptitle('Exploratory Data Analysis of Life Expectancy Factors', fontsize=20)


# --- Plot 1: Distribution of Life Expectancy ---
plt.subplot(2, 2, 1)
sns.histplot(df['Life expectancy'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Life Expectancy', fontsize=14)
plt.xlabel('Life Expectancy (Age)')
plt.ylabel('Frequency')


# --- Plot 2: Correlation Heatmap ---
# We compute the correlation matrix on numerical columns only
plt.subplot(2, 2, 2)
# Exclude non-numeric 'Country' and 'Status' for correlation calculation
numeric_df = df.drop(['Country', 'Status'], axis=1)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Features', fontsize=14)


# --- Plot 3: Schooling vs. Life Expectancy ---
plt.subplot(2, 2, 3)
sns.scatterplot(x='Schooling', y='Life expectancy', data=df, alpha=0.6)
plt.title('Schooling vs. Life Expectancy', fontsize=14)
plt.xlabel('Years of Schooling')
plt.ylabel('Life Expectancy (Age)')


# --- Plot 4: Income Composition vs. Life Expectancy ---
plt.subplot(2, 2, 4)
sns.scatterplot(x='Income composition of resources', y='Life expectancy', data=df, alpha=0.6)
plt.title('Income Composition vs. Life Expectancy', fontsize=14)
plt.xlabel('Income Composition of Resources (Index)')
plt.ylabel('Life Expectancy (Age)')


# Adjust layout and show the plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Preprocessing & Model Training ---

# 1. Handle categorical 'Status' column and drop non-predictive 'Country' column
df_model = pd.get_dummies(df, columns=['Status'], drop_first=True)
df_model = df_model.drop('Country', axis=1)


# 2. Define our features (X) and target (y)
X = df_model.drop('Life expectancy', axis=1)
y = df_model['Life expectancy']


# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data successfully preprocessed and split.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")


# 4. Initialize and Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nTraining the Random Forest Regressor model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")


# 5. Make predictions on the test set
y_pred = rf_model.predict(X_test)


# 6. Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} years")
print(f"R-squared (R2) Score: {r2:.2f}")

# %%
