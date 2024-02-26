import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_validate
import time
from glob import glob
from joblib import Parallel, delayed
from typing import List

tic = time.time()

# Function to process a file
def process_file(file: str) -> pd.DataFrame:
    df = pd.read_csv(file, skiprows=5)
    df = df.apply(pd.to_numeric, errors='coerce')  # Try to convert to numbers, replace failures with NaN
    df = df.dropna()  # Remove any rows with NaN
    df.columns = df.columns.str.strip()  # Remove leading and trailing spaces from column names

    # Rename columns to remove special characters
    df = df.rename(columns={
        'X [ m ]': 'X_m',
        'Y [ m ]': 'Y_m',
        'Velocity [ m s^-1 ]': 'Velocity_ms',
    })

    # Add inlet variables
    basename = os.path.basename(file)
    basename = basename.replace(".csv", "")

    # Use regex to extract the inlet values
    match = re.search(r'Tr\d+\(inlet_1=(\d+),inlet_2=(\d+)\)', basename)
    if match is None:
        print(f"Pattern not found in {basename}. Skipping this file.")
        return None
    inlet_1 = float(match.group(1))
    inlet_2 = float(match.group(2))

    df['inlet_1'] = inlet_1
    df['inlet_2'] = inlet_2

    return df

# Get a list of all training files
train_files = glob('./train/*.csv')

# Process each training file in parallel
with Parallel(n_jobs=-1, verbose=1) as parallel:
    train_data: List[pd.DataFrame] = parallel(delayed(process_file)(file) for file in train_files)
    train_data = [df for df in train_data if df is not None]  # Remove None values

# Concatenate all training data into a single dataframe
train = pd.concat(train_data)

# Define features and targets
features = ['X_m', 'Y_m', 'inlet_1', 'inlet_2']
target = 'Velocity_ms'

# Get X (features) and y (target) from the training data
X = train[features]
y = train[target]

lgb_reg = lgb.LGBMRegressor(random_state=20220,force_col_wise=True, 
                            n_estimators=2000,
                            gpu_device_id = 1,
                            )
lgb_reg.fit(X, y) # train model with all data

# Load the test x_m and y_m data
df_test = pd.read_csv('./test_xy_coordinate.csv', skiprows=4)
df_test = df_test.apply(pd.to_numeric, errors='coerce')  # Try to convert to numbers, replace failures with NaN
df_test = df_test.dropna()  # Remove any rows with NaN
df_test.columns = df_test.columns.str.strip()  # Remove leading and trailing spaces from column names
df_test = df_test.rename(columns={
    'X [ m ]': 'X_m',
    'Y [ m ]': 'Y_m',
})
df_test = df_test[['X_m', 'Y_m']]

# Test Step
test_inputs = [[1.5,1],[1,1.5],[1.5,1.5],[1.2,1.2], [1.8,1.8],[2.5,1],[1,2.5],[2.5,2.5]]
df_results = []
for i, test_input in enumerate(test_inputs):
    # Create a copy of the test dataframe
    X_test = df_test.copy()
    X_test['inlet_1'] = test_input[0]
    X_test['inlet_2'] = test_input[1]

    # Use the trained model to predict the velocities
    test_output = lgb_reg.predict(X_test)

    # Create a new DataFrame with the output and the X_m and Y_m columns
    df_result = X_test[['X_m', 'Y_m']].copy()
    df_result['Te' + str(i+1)] = test_output

    df_results.append(df_result)

# Concatenate the results DataFrames
df_final = pd.concat(df_results, axis=1)

# Remove duplicate columns
df_final = df_final.loc[:,~df_final.columns.duplicated()]

# Save the final DataFrame to a CSV file
df_final.to_csv('test_XGB.csv', index=False)

toc = time.time()

print(f"Total time: {toc - tic} seconds")
