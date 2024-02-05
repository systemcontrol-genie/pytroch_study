import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.datasets import load_boston

dataset = load_boston()
dataframe = pd.DataFrame(dataset["data"])
dataframe.columns = dataset["feature_names"]
dataframe["target"] = dataset["target"]

print(dataframe.head())

# Model definition
model = nn.Sequential(
    nn.Linear(13, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

# Input data and target
x = dataframe.iloc[:, :13].values
y = dataframe["target"].values

# Convert to PyTorch tensors
x = torch.FloatTensor(x)
y = torch.FloatTensor(y).view(-1, 1)  # Reshape y to have shape (batch_size, 1)

batch_size = 100
learning_rate = 0.001

optim = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(200):
    for i in range(0, len(x), batch_size):
        start = i
        end = start + batch_size
        x_batch = x[start:end]
        y_batch = y[start:end]

        optim.zero_grad()
        preds = model(x_batch)
        loss = nn.MSELoss()(preds, y_batch)
        loss.backward()
        optim.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Making a prediction
sample_input = torch.FloatTensor(x[0, :])  # Use a new variable for prediction
prediction = model(sample_input.view(1, -1))  # Reshape to match model input size
real = y[0].item()

print(f"Prediction: {prediction.item()}, Real: {real}")