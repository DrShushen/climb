import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

torch.manual_seed(42)


# Define a baseline neural network
class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        return output


# Pytorch dataset class
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(
                self.data[index]
                .astype(
                    np.uint8,
                )
                .transpose(1, 2, 0),
            )
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


"""Enables MC Dropout - i.e dropout at test time"""


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


class benchmark_trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _train(self, train_loader, optimizer, epoch):
        """
        > This function define the actual training loop

        Args:
          train_loader: the training data
          optimizer: The optimizer to use.
          epoch: The number of times the model will go through the entire dataset.
        """
        self.model.train()
        loss_val = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_val.append(loss.item())

        # logging.info(f"Epoch {epoch} - Loss = {np.mean(loss_val)}")
        print(f"Epoch {epoch} - Loss = {np.mean(loss_val)}")

    def fit(self, train_loader, optimizer, epochs):
        """
        For each epoch, we iterate through the training data & train the model with the train loop

        Args:
          train_loader: the training data loader
          optimizer: The optimizer to use for training.
          epochs: number of epochs to train for
        """
        for epoch in range(1, epochs + 1):
            self._train(train_loader, optimizer, epoch)

    def predict(self, test_loader, mc_samples=None):
        """
        > This function defines a generic prediction function whether we use
        MC sampling or not

        Args:
          test_loader: the test data loader
          mc_samples: number of Monte Carlo samples to take

        Returns:
          The means and standard deviations of the predictions.
        """
        self.model.eval()

        with torch.no_grad():
            means = []
            stds = []
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                if mc_samples is not None:
                    enable_dropout(self.model)
                    mc_samps = np.array(
                        [self.model(data).data.cpu().numpy() for _ in range(mc_samples)],
                    ).squeeze()

                    means.extend(list(np.mean(mc_samps, axis=0)))
                    stds.extend(list(np.std(mc_samps, axis=0)))
                else:
                    output = self.model(data)
                    means.extend(output.data.cpu().numpy())
                    [0] * len(output)

        return means, stds
