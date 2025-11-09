import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from climb.tool.impl.data_suite.models.base_model import MyDataset, Net, benchmark_trainer

torch.manual_seed(42)


class ensemble:
    def __init__(self, epochs=10, lr=0.01, batch_size=5, n_models=5, device="cpu"):
        self.epochs = epochs

        self.lr = lr

        self.batch_size = batch_size

        self.device = device

        self.n_models = n_models

        self.tr = None

        self.ensemble = {}

    def fit(self, x_train, y_train):
        """
        > This function fits an ensemble of n_models to the data

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

        for i in range(self.n_models):
            # Instantiate the model + optimizer - can be anything
            all_equal = True

            def all_equal2(iterator):
                return len(set(iterator)) <= 1

            runs = 0
            while all_equal:
                # Instantiate the model + optimizer - can be anything
                logging.info(f"Instantiating model {i} - Ensemble")
                model = Net(dim=dim).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)

                # generic training & test loop
                self.tr = benchmark_trainer(model, self.device)
                logging.info(f"Training model {i} - Ensemble...")
                self.tr.fit(train_loader, optimizer, epochs=self.epochs)

                preds, uncertainty = self.tr.predict(
                    train_loader,
                    mc_samples=3,
                )

                runs += 1

                if runs > 2:
                    break

                if all_equal2(preds) is False:
                    self.ensemble[i] = self.tr
                    break

    def predict(self, x_test, y_test, mc_samples=3):
        """
        > For each model we get the predictions and the uncertainty.
        Args:
          x_test: the test data
          y_test: the true labels of the test set
          mc_samples: number of Monte Carlo samples to use for prediction

        Returns:
          The mean of the predictions and the standard deviation of the predictions.

        """
        dataset_test = MyDataset(data=x_test, targets=y_test, transform=None)
        test_loader = DataLoader(dataset_test, batch_size=self.batch_size)

        preds_overall = []

        for i in range(self.n_models):
            logging.info(f"Testing model {i}...")
            preds, _ = self.tr.predict(test_loader, mc_samples=mc_samples)
            preds_overall.append(preds)

        if self.n_models > 1:
            preds = np.mean(np.array(preds_overall), axis=0)
            uncertainty = np.std(np.array(preds_overall), axis=0)

        return preds, uncertainty
