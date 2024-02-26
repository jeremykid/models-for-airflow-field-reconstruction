import os
import re
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error
import time
from glob import glob
from sklearn.svm import SVR
tic = time.time()

# Get a list of all training files
train_files = glob('D:/XGB and DNN Comparison/Case 1 - real case/Two heat wall case/Data/Train data/*.csv')

train_data = []

# Process each training file
for file in train_files:
    df = pd.read_csv(file, skiprows=5)
    df = df.apply(pd.to_numeric, errors='coerce')  # Try to convert to numbers, replace failures with NaN
    df = df.dropna()  # Remove any rows with NaN
    df = df.rename(columns=lambda x: x.strip())  # Remove leading and trailing spaces from column names

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
        continue
    inlet_1 = float(match.group(1))
    inlet_2 = float(match.group(2))

    df['inlet_1'] = inlet_1
    df['inlet_2'] = inlet_2

    # Append the dataframe to the list
    train_data.append(df)

# Concatenate all training data into a single dataframe
train = pd.concat(train_data)

# Export the combined dataframe to a .csv file
train.to_csv('D:/XGB and DNN Comparison/Case 1 - real case/Two heat wall case/Data/Train data/combined_train_data.csv', index=False)

# Define features and target
features = ['X_m', 'Y_m', 'inlet_1', 'inlet_2']
target = 'Velocity_ms'

# Get X (features) and y (target) from the training data
X = train[features]
y = train[target]

# Find rows where y is NaN or infinite, and remove them from both X and y
mask = y.isna() | np.isinf(y)
X = X.loc[~mask]
y = y.loc[~mask]

# Use KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2022)
svr = SVR(C=10, epsilon=0.1, gamma=1, kernel='rbf')
result_svr = cross_validate(svr, X, y, cv=kf, scoring='neg_root_mean_squared_error', return_train_score = True, verbose=True, n_jobs=-1)

def RMSE(result, name):
    return abs(result[name].mean())

train_score = RMSE(result_svr, 'train_score')
print("Train Score: ", train_score)
test_score = RMSE(result_svr, 'test_score')
print("Test Score: ", test_score)

svr.fit(X, y) # train model with all data

# Load the test x_m and y_m data
df_test = pd.read_csv('D:/XGB and DNN Comparison/Case 1 - real case/Two heat wall case/Data/Train data/test_xy_coordinate.csv', skiprows=4)
df_test = df_test.apply(pd.to_numeric, errors='coerce')  # Try to convert to numbers, replace failures with NaN
df_test = df_test.dropna()  # Remove any rows with NaN
df_test = df_test.rename(columns=lambda x: x.strip())  # Remove leading and trailing spaces from column names
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
    test_output = svr.predict(X_test)

    # Create a new DataFrame with the output and the X_m and Y_m columns
    df_result = X_test[['X_m', 'Y_m']].copy()
    df_result['Te' + str(i+1)] = test_output

    df_results.append(df_result)

# Concatenate the results DataFrames
df_final = pd.concat(df_results, axis=1)

# Remove duplicate columns
df_final = df_final.loc[:,~df_final.columns.duplicated()]
df_final = df_final.abs()
# Save the final DataFrame to a CSV file
df_final.to_csv('test.csv', index=False)

toc = time.time()

print(f"Total time: {toc - tic} seconds")
