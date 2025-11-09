import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from climb.tool.impl.data_suite.models.base_model import MyDataset, Net, benchmark_trainer

torch.manual_seed(42)


def all_equal2(iterator):
    return len(set(iterator)) <= 1


class mc_dropout:
    def __init__(self, epochs=10, lr=0.01, batch_size=5, device="cpu"):
        self.epochs = epochs

        self.lr = lr

        self.batch_size = batch_size

        self.device = device

        self.tr = None

    def fit(self, x_train, y_train):
        """
        > The function instantiates a model, trains it, and then checks if the predictions are all
        equal. If they are, it instantiates a new model and repeats the process

        Args:
          x_train: the training data
          y_train: the training labels
        """

        dataset_train = MyDataset(
            data=x_train,
            targets=y_train,
            transform=None,
        )
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size)

        dim = x_train.shape[1]

        all_equal = True

        runs = 0

        while all_equal:
            # Instantiate the model + optimizer - can be anything
            logging.info("Instantiating model - MCD")
            model = Net(dim=dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            # generic training & test loop
            self.tr = benchmark_trainer(model, self.device)
            logging.info("Training model (MCD)...")
            self.tr.fit(train_loader, optimizer, epochs=self.epochs)

            preds, uncertainty = self.tr.predict(train_loader, mc_samples=3)

            runs += 1

            if runs > 3:
                break

            if all_equal2(preds) is False:
                break

    def predict(self, x_test, y_test, mc_samples=3):
        """
        > The function takes in the test data and test labels, and returns the predictions and the
        uncertainty of the predictions

        Args:
          x_test: the test data
          y_test: the actual labels of the test set
          mc_samples: number of Monte Carlo samples to use for prediction. Defaults to 3

        Returns:
          The predictions and the uncertainty of the predictions.
        """
        logging.info("Testing model...")

        dataset_test = MyDataset(data=x_test, targets=y_test, transform=None)
        test_loader = DataLoader(dataset_test, batch_size=self.batch_size)
        preds, uncertainty = self.tr.predict(
            test_loader,
            mc_samples=mc_samples,
        )
        return preds, uncertainty
