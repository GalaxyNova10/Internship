#%%
import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('world_population.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'world_population.csv' not found. Please make sure the file is in the correct project folder.")

# --- Data Preparation (Corrected) ---

# Select only the columns that contain population data
population_cols = ['1970 Population', '1980 Population', '1990 Population', '2000 Population',
                   '2010 Population', '2015 Population', '2020 Population', '2022 Population']
world_pop_sum = df[population_cols].sum()

# Convert the resulting Series into a DataFrame
world_pop_ts = world_pop_sum.to_frame(name='Population').reset_index()

# Rename the 'index' column to 'Year'
world_pop_ts.rename(columns={'index': 'Year'}, inplace=True)

# Clean up the 'Year' column to be just the number
world_pop_ts['Year'] = world_pop_ts['Year'].str.extract('(\d+)').astype(int)


# --- Prepare data for Prophet model ---
# Prophet requires two columns: 'ds' (datestamp) and 'y' (value)
prophet_df = world_pop_ts[['Year', 'Population']].copy()
prophet_df.rename(columns={'Year': 'ds', 'Population': 'y'}, inplace=True)

# Convert 'ds' to datetime objects, setting it to the start of each year
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')


print("\nData prepared for forecasting:")
print(prophet_df)
#%%
from prophet import Prophet
import matplotlib.pyplot as plt  # <-- This is the line that was missing

# --- Model Building and Forecasting ---

# 1. Initialize the Prophet model
m = Prophet()

# 2. Fit the model to our historical data
m.fit(prophet_df)

# 3. Create a dataframe for future dates
# We will forecast 28 years into the future to reach 2050
future = m.make_future_dataframe(periods=28, freq='Y')

# 4. Make a forecast
forecast = m.predict(future)

# --- Display the Forecast Results ---

# Print the last few rows of the forecast
print("Forecast results (including future predictions):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# --- Visualize the Forecast ---

# Plot the forecast
print("\nGenerating forecast plot...")
fig1 = m.plot(forecast)
plt.title('World Population Forecast until 2050')
plt.xlabel('Year')
plt.ylabel('Population (in Billions)') # Updated label for clarity
plt.show()

# Plot the forecast components
print("\nGenerating forecast components plot...")
fig2 = m.plot_components(forecast)
plt.show()
# %%
