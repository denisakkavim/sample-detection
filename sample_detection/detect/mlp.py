import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader


class MLPClassifier:
    def __init__(
        self,
        normalisation_layer=None,
        activation_layer=torch.nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        hidden_layer_sizes: Tuple[int, int] = (256, 64),
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:

        super().__init__()

        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.normalisation_layer = normalisation_layer
        self.activation_layer = activation_layer
        self.dropout = dropout
        self.bias = bias
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

        self.pytorch_model = None

    def _init_pytorch_model(
        self,
        in_dim: int,
        hidden_layer_sizes: Tuple[int],
        normalisation_layer=None,
        activation_layer=torch.nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
    ) -> torch.nn.Sequential:

        dims = (in_dim,) + hidden_layer_sizes + (1,)
        layers = []

        for i in range(len(dims) - 1):
            layer_in_dim = dims[i]
            layer_out_dim = dims[i + 1]
            layers.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
            if normalisation_layer is not None:
                layers.append(normalisation_layer(layer_out_dim))
            layers.append(activation_layer)
            layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        if len(X.shape) != 2:
            raise ValueError("X must be a 2d np array")

        if len(y.shape) != 1:
            raise ValueError("y must be a 1d np array")

        n_rows, n_features = X.shape
        y = y.reshape((n_rows, 1))

        self.pytorch_model = self._init_pytorch_model(
            in_dim=n_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            normalisation_layer=None,
            activation_layer=torch.nn.ReLU(),
            dropout=self.dropout,
            bias=self.bias,
        )

        self.pytorch_model.train()

        X = torch.Tensor(X).float()  # transform to torch tensor
        y = torch.Tensor(y).float()

        dataloader = DataLoader(
            TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.pytorch_model.parameters(), lr=self.learning_rate
        )

        prev_loss = np.float32("inf")

        for i in range(self.max_iter):

            for (X, y) in dataloader:
                pred = self.pytorch_model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_loss = loss.item()
            if np.abs(prev_loss - current_loss) < self.tol:
                break

            prev_loss = loss.item()

    def predict_proba(self, X) -> np.ndarray:

        self.pytorch_model.eval()

        X = torch.tensor(X).float()
        with torch.no_grad():
            pred = self.pytorch_model(X)

        return np.hstack([1 - pred, pred])
