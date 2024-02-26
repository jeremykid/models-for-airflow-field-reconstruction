import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
import time
from glob import glob
from joblib import Parallel, delayed
from typing import List
from torch.utils.data import TensorDataset, DataLoader

import pickle
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from typing import Type, Any, Callable, Union, List, Optional
from tqdm import tqdm
GPU_NUM = '7' #GPU device number, update
device = torch.device("cuda:"+GPU_NUM if torch.cuda.is_available() else "cpu")

class DNN_model(nn.Module):

    def __init__(
        self,
        input_size: int = 4,
        output_size: int = 1,
        hidden_layer: int = 64,
        n_hidden_layer: int = 40
    ):
        super(DNN_model, self).__init__()
        self.hidden_layer_list = nn.ModuleList()
        self.n_hidden_layer = n_hidden_layer
        self.hidden_layer_start = nn.Linear(input_size, hidden_layer)
        for i in range(self.n_hidden_layer-1):
            hidden_layer_temp = nn.Linear(hidden_layer, hidden_layer)
            self.hidden_layer_list.append(hidden_layer_temp)
        self.relu = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(hidden_layer, output_size)
        
    def forward(self, x):
        x = self.hidden_layer_start(x)
        x = self.relu(x)
        for i in range(self.n_hidden_layer-1):
            x = self.hidden_layer_list[i](x)
            x = self.relu(x)
        result = self.output_layer(x)
        return result

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
train_files = glob('D:/XGB and DNN Comparison/Case 1 - real case/Two heat wall case/Data/Train data/*.csv')

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
Xi = Tensor(X.values).to(device)
Yi = Tensor(y.tolist()).to(device)


# Create a dataset from the tensors
dataset = TensorDataset(Xi, Yi)

# Create a DataLoader with batch size 16
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Need Update

model = DNN_model(input_size = 4, output_size=1, hidden_layer= 64, n_hidden_layer = 40) # Need Update
model = model.to(device)

lossfun = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
#make_optimizer(Adam, model, lr=0.0001, weight_decay=0)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, min_lr=0.000001)
num_epochs = 5000 # Need Update
plot_loss = []
with tqdm(total=num_epochs) as pbar:
    for i in range(num_epochs):
        model.train()
        for batch in dataloader:
            X_b, Y_b = batch
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            train_loss = 0
            y_pred = model(X_b)
            optimizer.zero_grad()
            loss = lossfun(y_pred, Y_b)
            loss.backward()
        train_loss += loss.item()
        optimizer.step()
        plot_loss.append(train_loss/5)
        pbar.set_description(f"[epoch {i+1}/{num_epochs} ]")
        pbar.set_postfix_str(f"loss = {train_loss/5:.4f}")
        pbar.update(1)

# Load the test x_m and y_m data
df_test = pd.read_csv('D:/XGB and DNN Comparison/Case 1 - real case/Two heat wall case/Data/Train data/test_xy_coordinate.csv', skiprows=4)
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
    # test_output = rf_reg.predict(X_test)
    with torch.no_grad():
        test_input = Tensor(X_test.values).to(device)
        test_output = model(test_input)
        # data['te'+str(i)] = test_output.tolist()
    
        # Create a new DataFrame with the output and the X_m and Y_m columns
        df_result = X_test[['X_m', 'Y_m']].copy()
        df_result['Te' + str(i+1)] = [i[0] for i in test_output.tolist()]

        df_results.append(df_result)

# Concatenate the results DataFrames
df_final = pd.concat(df_results, axis=1)

# Remove duplicate columns
df_final = df_final.loc[:,~df_final.columns.duplicated()]

# Save the final DataFrame to a CSV file
df_final.to_csv('test_DNN.csv', index=False)

toc = time.time()

print(f"Total time: {toc - tic} seconds")
