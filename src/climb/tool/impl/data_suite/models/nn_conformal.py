import copy
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from nonconformist.base import RegressorAdapter
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class MSENet_RegressorAdapter(RegressorAdapter):
    def __init__(
        self,
        model,
        fit_params=None,
        in_shape=1,
        hidden_size=1,
        learn_func=torch.optim.Adam,
        epochs=1000,
        batch_size=10,
        dropout=0.1,
        lr=0.01,
        wd=1e-6,
        test_ratio=0.2,
        random_state=0,
    ):
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = mse_model(
            in_shape=in_shape,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.loss_func = torch.nn.MSELoss()
        self.learner = LearnerOptimized(
            self.model,
            partial(learn_func, lr=lr, weight_decay=wd),
            self.loss_func,
            device=device,
            test_ratio=self.test_ratio,
            random_state=self.random_state,
        )

    def fit(self, x, y):
        """
        > The function takes in a set of inputs and outputs, and uses them to train the model

        Args:
          x: The input data
          y: The target values.
        """

        self.learner.fit(x, y, self.epochs, batch_size=self.batch_size)

    def predict(self, x):
        """
        > The predict function takes in a single data point and returns the prediction

        Args:
          x: the input data

        Returns:
          The predicted value of the input x
        """

        return self.learner.predict(x)


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

###############################################################################
# Helper functions
###############################################################################


def train_loop(
    model,
    loss_func,
    x_train,
    y_train,
    batch_size,
    optimizer,
    cnt=0,
    best_cnt=np.inf,
):
    """
    > The function defines a training loop for the model

    Args:
      model: the model we're training
      loss_func: the loss function we want to use
      x_train: the training data
      y_train: the training labels
      batch_size: The number of samples to use for each gradient update.
      optimizer: the optimizer to use.
      cnt: the number of batches we've trained on. Defaults to 0
      best_cnt: the number of epochs to train for.

    Returns:
      The epoch loss and the count
    """

    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()
        batch_x = x_train[idx : min(idx + batch_size, x_train.shape[0]), :]
        batch_y = y_train[idx : min(idx + batch_size, y_train.shape[0])]
        preds = model(batch_x)

        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt


###############################################################################
# Deep conditional mean regression
# Minimizing MSE loss
###############################################################################


# Define the network
class mse_model(nn.Module):
    """Conditional mean estimator, formulated as neural net"""

    def __init__(self, in_shape=1, hidden_size=64, dropout=0.5):
        """Initialization

        Parameters
        ----------

        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """Construct the network"""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self):
        """Initialize the network parameters"""
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass"""
        return torch.squeeze(self.base_model(x))


# Define the training procedure
class LearnerOptimized:
    """Fit a neural network (conditional mean) to training data"""

    def __init__(
        self,
        model,
        optimizer_class,
        loss_func,
        device="cpu",
        test_ratio=0.2,
        random_state=0,
    ):
        """Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : int, seed to be used in CV when splitting to train-test

        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs, batch_size, verbose=False):
        """Fit the model to data

        Parameters
        ----------

        x : numpy array, containing the training features (nXp)
        y : numpy array, containing the training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD

        """

        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(
            x,
            y,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )

        x_train = (
            torch.from_numpy(x_train)
            .float()
            .to(
                self.device,
            )
            .requires_grad_(False)
        )
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = (
            torch.from_numpy(y_train)
            .float()
            .to(
                self.device,
            )
            .requires_grad_(False)
        )
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_cnt = 1e10
        best_test_epoch_loss = 1e10

        cnt = 0
        for e in range(epochs):
            epoch_loss, cnt = train_loop(
                model,
                self.loss_func,
                x_train,
                y_train,
                batch_size,
                optimizer,
                cnt,
            )
            self.loss_history.append(epoch_loss)

            #             print(epoch_loss)

            # test
            model.eval()
            preds = model(xx)
            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()

            self.test_loss_history.append(test_epoch_loss)

            if test_epoch_loss <= best_test_epoch_loss:
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                best_cnt = cnt

            if (e + 1) % 10 == 0 and verbose:
                print(
                    "CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(
                        e + 1,
                        epoch_loss,
                        test_epoch_loss,
                        best_epoch,
                        best_test_epoch_loss,
                    ),
                )
                sys.stdout.flush()

        # use all the data to train the model, for best_cnt steps
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch + 1):
            if cnt > best_cnt:
                break

            epoch_loss, cnt = train_loop(
                self.model,
                self.loss_func,
                x,
                y,
                batch_size,
                self.optimizer,
                cnt,
                best_cnt,
            )
            self.full_loss_history.append(epoch_loss)

            if (e + 1) % 100 == 0 and verbose:
                print(
                    "Full: Epoch {}: {}, cnt {}".format(
                        e + 1,
                        epoch_loss,
                        cnt,
                    ),
                )
                sys.stdout.flush()

    def predict(self, x):
        """Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        self.model.eval()
        ret_val = (
            self.model(
                torch.from_numpy(x)
                .to(
                    self.device,
                )
                .requires_grad_(False),
            )
            .cpu()
            .detach()
            .numpy()
        )
        return ret_val
