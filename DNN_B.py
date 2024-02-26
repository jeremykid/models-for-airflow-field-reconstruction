import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import RMSprop
import time
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN_model(nn.Module):
    def __init__(self, input_size: int = 4, output_size: int = 1, hidden_layer: int = 16, num_layers: int = 10):
        super(DNN_model, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_layer))
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer, hidden_layer))
        self.relu = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
        result = self.output_layer(x)
        return result

# Prepare data
def prepare_data(file_path, coordinates_df, inlets):
    data = pd.read_csv(file_path, header=0)
    data.columns = data.columns.str.strip()  # Remove leading or trailing spaces from column names
    inputs = pd.concat([coordinates_df]*len(inlets), ignore_index=True)
    #print(data.columns)
    inputs['inlet_1'] = [inlet[0] for inlet in inlets]*len(coordinates_df)
    inputs['inlet_2'] = [inlet[1] for inlet in inlets]*len(coordinates_df)
    outputs = data['Velocity [ m s^-1 ]']
    return inputs.values, outputs.values

model = DNN_model()
model = model.to(device)

lossfun = nn.MSELoss()
optimizer = RMSprop(model.parameters(), lr=0.001)


num_epochs = 500
plot_loss = []
base_dir = "D:\XGB and DNN Comparison\Case 1 - real case\Two heat wall case\Data\Example\\train\\DNNB"
train_inputs = [[1, 1], [1, 0], [1, 2], [0, 1], [2, 1]]  # TODO: replace with your actual inputs
coordinates_df = pd.read_csv('D:\XGB and DNN Comparison\Case 1 - real case\Two heat wall case\Data\Example/test_xy_coordinate.csv')


start_time = time.time()  # Define start_time

for i in range(num_epochs):
    model.train()
    train_loss = 0
    for train_input in train_inputs:
        file_path = os.path.join(base_dir, 'Tr(Inlet_1=' + str(train_input[0]) + ',Inlet_2=' + str(train_input[1]) + ').csv')
        inputs, outputs = prepare_data(file_path, coordinates_df, [train_input])
        inputs = Tensor(inputs).to(device)
        outputs = Tensor(outputs).unsqueeze(1).to(device)  # adds an extra dimension at the 1st position (index 1)
        y_pred = model(inputs)
        optimizer.zero_grad()
        loss = lossfun(y_pred, outputs)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    plot_loss.append(train_loss/len(train_inputs))
    print(f"[epoch {i+1}/{num_epochs} ] loss = {train_loss/len(train_inputs):.4f}")



end_time = time.time()
train_time = end_time - start_time
print("Training time: ", train_time, "seconds")

data = {}
test_inputs = [[1, 1], [1, 0], [1, 2], [0, 1], [2, 1]] # Inlet 1, Inlet 2 values for testing
i = 0
coordinate_values = coordinates_df.values  # Get the values from the DataFrame
with torch.no_grad():
    for test_input in test_inputs:
        file_path = os.path.join(base_dir, 'Tr(Inlet_1=' + str(test_input[0]) + ',Inlet_2=' + str(test_input[1]) + ').csv')
        inlet_values = np.full((coordinate_values.shape[0], 2), test_input)
        inputs = np.hstack((coordinate_values, inlet_values))
        inputs = Tensor(inputs).to(device)
        test_output = model(inputs)
        i += 1
        data['Tr'+str(i)] = test_output.squeeze().tolist()  # remove extra dimension


pd.DataFrame.from_dict(data).to_csv('train_predication_DNNB.csv')

end_time = time.time()
train_time = end_time - start_time
print("Total time: ", train_time, "seconds")

# Draw Training loss
import matplotlib.pyplot as plt
plt.plot([i for i in range(num_epochs)], plot_loss, 'bo')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.show()
plt.savefig('MSE_loss_plot.png', dpi=600, bbox_inches='tight')
