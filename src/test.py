# This is my PNN testing script
# Data loading parts need adaptation since the original data isn't included

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import src.config as config
from src.Column_Genearator import *
from src.ProgNet import ProgNet


# DataController and Reporter need to be implemented by anyone using this code
from data import DataController
from reporter import Reporter


# Check if GPU is available
is_cuda = torch.cuda.is_available()

# Set device to GPU if available, otherwise CPU
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def evaluate_presicion(model, X, y, class_type='binary', interval_length=100):
    # Function to evaluate precision of the PNN on given data
    tp = 0
    fp = 0
    fn = 0
    corrects = 0
    start_index = 0  # starting index of the interval of X to predict
    length = len(X)
    
    # Predict in intervals to avoid memory issues
    while start_index + interval_length < length:
        predicted_output = model(model.numCols - 1, X[start_index:start_index + interval_length])[:, -1]
        predicted_labels = torch.argmax(predicted_output, dim=1)
        if class_type == 'binary':
            for i in range(start_index, start_index + interval_length):
                if y[i] == predicted_labels[i - start_index]:
                    corrects += 1
        start_index += interval_length
    precision = corrects / len(y)
    return precision


# Initialize data controller and column generator
controller = DataController(batch_size=config.column_batch_size, mode='binary', flatten=False, data_list=config.all_labels)
column_generator = Column_generator_LSTM(
    input_size=config.pkt_size,
    hidden_size=128,
    num_dens_Layer=2,
    num_LSTM_layer=1,
    num_of_classes=2
)

# Create the PNN and move it to the device
binary_pnn = ProgNet(column_generator)
binary_pnn.to(device)

# Prepare validation set (2500 flows)
X_val = []
y_val = []
while True:
    data = controller.generate('validation')
    if not data:
        break
    X_val += data['x'].tolist()
    y_val += data['y'].tolist()
print("Number of validation flows:", len(X_val))

X_val = torch.tensor(np.array(X_val)).float().to(device)
y_val = np.array(y_val)

# Define loss function
loss_criteria = nn.BCELoss()
precisions = []
number_of_columns = 0

# Train progressive columns
while True:
    start = time.time()
    data = controller.generate('train', output_model='prob')
    if not data:
        break

    # Add a new column to PNN
    idx = binary_pnn.addColumn(device=device)
    flows = data['x']
    labels = data['y']
    optimizer = optim.Adam(binary_pnn.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        current_flow_index = 0
        while current_flow_index < len(flows):
            # Training on batch
            optimizer.zero_grad()
            X_train = Variable(torch.tensor(flows[current_flow_index:current_flow_index + config.learning_batch_size]).float().to(device))
            y_train = Variable(torch.tensor(labels[current_flow_index:current_flow_index + config.learning_batch_size]).float().to(device))
            output = binary_pnn(idx, X_train)[:, -1]
            loss = loss_criteria(output.float(), y_train)
            loss.backward()
            optimizer.step()
            current_flow_index += config.learning_batch_size

    # Freeze all columns after training current column
    binary_pnn.freezeAllColumns()

    # Evaluate precision on validation set
    if number_of_columns % 1 == 0:
        precision = evaluate_presicion(binary_pnn, X_val, y_val)
        precisions.append(precision)
        print("Columns created:", number_of_columns + 1)
        print("Precision:", precision)

    number_of_columns += 1
    end = time.time()
    print(str(end - start), "seconds")

# Save trained PNN weights
torch.save(binary_pnn.state_dict(), 'binary-PNN-weights.pth')

# Plot validation precision over columns
plt.plot([i * 10 for i in range(number_of_columns)], precisions, color='b')
plt.suptitle('Precisions')
plt.show()

# Prepare test set (1000 flows)
X_test = []
y_test = []
while True:
    data = controller.generate('test')
    if not data:
        break
    X_test += data['x'].tolist()
    y_test += data['y'].tolist()

X_test = torch.tensor(np.array(X_test)).float().to(device)
y_test = np.array(y_test)

# Evaluate final precision on test set
final_precision = evaluate_presicion(binary_pnn, X_test, y_test)
print("Precision on test:", final_precision)
