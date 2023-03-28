import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as torch_DataLoader


loss_function = torch.nn.MSELoss()  # Always use this loss function


def make_network(N_hid_layers, hid_layer_size, input_size, dropout=0.95, activation=torch.nn.LeakyReLU(), bias=True):
    """Make densne neural network structure."""
    dropout = torch.nn.Dropout(p=1 - (dropout))
    layer0 = torch.nn.Linear(torch.tensor(input_size),
                             hid_layer_size, bias=bias)
    layers = [layer0, activation, dropout]
    for i in range(N_hid_layers):
        lin_layer = torch.nn.Linear(hid_layer_size, hid_layer_size)
        layers.append(lin_layer)
        layers.append(activation)
        layers.append(dropout)
    layers.append(torch.nn.Linear(hid_layer_size, 1))  # Output layer
    return torch.nn.Sequential(*layers)


def convert_to_tensors(X, y, batch=10):
    """Convert pandas data frames to pytorch tensors."""
    X = torch.t(torch.stack(
        [torch.tensor(X[i].values, dtype=float) for i in X]))
    #y = torch.t(torch.stack([torch.tensor(y[i].values) for i in y]))
    y = torch.from_numpy(np.array(y)).reshape(-1, 1)

    data = TensorDataset(X, y)
    data_loader = torch_DataLoader(data, shuffle=True, batch_size=batch)
    return data_loader


# NN TRAINING PROCEDURE

def train_(loader, model, optimizer, scheduler):
    model.train()
    loss_tot = 0
    for data in loader:
        y = data[1].float()
        data = data[0].float()
        optimizer.zero_grad()
        out = model(data)
        loss_mse = loss_function(out, y)
        loss = torch.log(loss_mse)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_tot += loss.item()
    return loss_tot/len(loader)


def test_(loader, model):
    model.eval()
    loss_tot = 0
    for data in loader:
        y = data[1].float()
        data = data[0].float()
        out = model(data)
        loss_mse = loss_function(out, y)
        loss = torch.log(loss_mse)
        loss_tot += loss.item()
    return loss_tot/len(loader)


def validation(loader, model):
    model.eval()
    pred_act = list()
    for data in loader:  # Iterate in batches over the dataset.
        with torch.no_grad():
            y = data[1].float()
            out = model(data[0].float())
            pred_act.append((out, y))
    return pred_act


def main_training(model, train_loader, test_loader, LR, WD, n_epochs):
    """Main training procedure for the DNN - we use ADAM optimizer but have 
    explicitly tried that the use of other optimizers does not improve the results"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=LR, max_lr=1.e-3, cycle_momentum=False)
    train_losses, valid_losses = [], []
    for epoch in range(1, n_epochs+1):
        train_loss = train_(train_loader, model, optimizer, scheduler)
        test_loss = test_(test_loader, model)
        train_losses.append(train_loss)
        valid_losses.append(test_loss)
        # print(epoch)
    return train_losses, valid_losses
