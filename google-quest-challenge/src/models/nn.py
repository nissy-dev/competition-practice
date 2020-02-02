import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# for tabular data loader
def create_data_loader(x_trn, y_trn, x_val, y_val, x_test, batch_size=128):
    scaler = StandardScaler()
    x_trn = scaler.fit_transform(x_trn)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    # convert Tensor value
    train_ds = TensorDataset(torch.FloatTensor(x_trn), torch.FloatTensor(y_trn))
    val_ds = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    dummy_val = np.zeros(x_test.shape[0], y_val.shape[1])
    test_ds = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(dummy_val))
    # create dataloader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    train_dls = {}
    train_dls['train'] = train_dl
    train_dls['valid'] = val_dl
    return train_dls, test_dl


def swish(x):
    return x * F.sigmoid(x)


class SimpleNN(nn.Module):
    def __init__(self, input_dim=1000, out_dim=30):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = swish(self.bn1(self.fc1(x)))
        x = swish(self.bn2(self.fc2(x)))
        x = swish(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.fc4(x))
        return F.log_softmax(x, dim=1)
