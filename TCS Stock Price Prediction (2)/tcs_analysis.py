#%%
import yfinance as yf
import pandas as pd

# Define the ticker symbol for TCS on the National Stock Exchange of India
tcs_ticker = 'TCS.NS'

# Download historical data for TCS from 2010 to today
tcs_data = yf.download(tcs_ticker, start='2010-01-01')

# Display the first few rows of the dataset
print("TCS Stock Data downloaded successfully:")
print(tcs_data.head())

# Display the last few rows to see the most recent data
print("\nMost recent data:")
print(tcs_data.tail())
#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Data Preprocessing (Corrected) ---

# Create a new dataframe with only the 'Close' column using double brackets
close_data = tcs_data[['Close']] # <-- This is the corrected line
# Convert the dataframe to a numpy array
dataset = close_data.values
# Get the number of rows to train the model on (80% of the data)
training_data_len = int(np.ceil(len(dataset) * .8))

# Scale the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

# Create the sequences of 60 days
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays to train the LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to be 3-dimensional for the LSTM model
# LSTM expects [samples, timesteps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print("Data preprocessing complete.")
print(f"Shape of the training data (X_train): {x_train.shape}")
print(f"Shape of the training labels (y_train): {y_train.shape}")
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- Build the LSTM Model ---

# Initialize the model
model = Sequential()

# Add the first LSTM layer with Dropout
# 50 units is a good starting point for the dimensionality of the output space
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Add a second LSTM layer with Dropout
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add the final Dense layer that will predict the single price value
model.add(Dense(units=1))

# --- Compile and Train the Model ---

# Compile the model
# Adam is a standard, effective optimizer
# Mean squared error is a common loss function for regression problems
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# An epoch is one full cycle through the training data.
# We'll start with just 1 epoch to make sure it runs correctly.
print("Training the model... This will take a few minutes.")
model.fit(x_train, y_train, batch_size=32, epochs=1)

print("Model training complete.")
#%%

# --- Step 5: Evaluate the Model ---

# Create the testing data set
# Create a new array containing scaled values from index 3011 to 3764
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] # The actual values
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
print("Getting model predictions...")
predictions = model.predict(x_test)
# Un-scale the predictions back to original price scale
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")

# --- Visualize the Prediction ---
print("Plotting the results...")
# Prepare data for plotting
train = close_data[:training_data_len]
valid = close_data[training_data_len:]
valid['Predictions'] = predictions

# Plot the data
plt.figure(figsize=(16, 8))
plt.title('TCS Stock Price Prediction Model')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (INR)', fontsize=12)
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Actual Price (Test Data)', color='blue')
plt.plot(valid['Predictions'], label='Predicted Price', color='orange')
plt.legend(loc='lower right')
plt.show()
# %%
