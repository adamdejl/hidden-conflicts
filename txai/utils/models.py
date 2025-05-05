import torch
import torch.nn as nn

from collections import namedtuple
from torcheval.metrics.functional import (
    mean_squared_error,
    multiclass_f1_score,
    multiclass_accuracy,
    binary_f1_score,
    binary_accuracy,
    auc,
)
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, identity, block):
        super().__init__()
        self.identity = identity
        self.block = block

    def forward(self, x):
        return self.identity(x) + self.block(x)


class Slice(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.slice = slice

    def forward(self, x):
        return x[self.slice]


def construct_feedforward_nn(nn_dims, activation_fun):
    layers = []
    for i in range(1, len(nn_dims)):
        in_dim, out_dim = nn_dims[i - 1], nn_dims[i]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fun())
    # Remove the last activation layer
    layers = layers[:-1]

    return nn.Sequential(*layers).to(DEVICE)


def train_nn(
    model,
    train_dl,
    num_epochs,
    val_dl=None,
    loss=nn.CrossEntropyLoss,
    optimizer_constructor=None,
    report_progress=False,
):
    if optimizer_constructor is None:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        opt = optimizer_constructor(model.parameters())
    loss_fun = loss()

    losses = []
    for epoch in (training_progress := tqdm(range(num_epochs), leave=False)):
        model.train()
        training_progress.set_description(f"Training epoch {epoch}")
        total_loss = 0
        i = 0
        for x, y in (batch_progress := tqdm(train_dl, leave=False)):
            i += 1
            batch_progress.set_description(f"Batch {i}")
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            if out.ndim == 2 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            if isinstance(loss_fun, nn.BCEWithLogitsLoss) or isinstance(
                loss_fun, nn.MSELoss
            ):
                y = y.float()
            if isinstance(loss_fun, nn.CrossEntropyLoss):
                y = y.long()
            loss = loss_fun(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        losses.append(total_loss)
        if report_progress:
            val_progress = ""
            if val_dl is not None:
                with torch.no_grad():
                    if isinstance(loss_fun, nn.MSELoss):
                        val_results = eval_continuous_nn(model, val_dl)
                        val_progress = f" | Validation loss: {val_results.loss} (RMSE: {val_results.rmse})"
                    else:
                        val_results = eval_nn(model, val_dl)
                        val_progress = f" | Validation loss: {val_results.loss} (F1: {val_results.f1}, Acc: {val_results.accuracy})"

            print(f"Epoch {epoch} training loss: {total_loss}{val_progress}")


def eval_nn(model, test_dl, loss=nn.CrossEntropyLoss):
    model.eval()
    model.to(DEVICE)
    loss_fun = loss()

    predictions = []
    labels = []
    num_classes = None
    i = 0
    total_loss = 0
    for x, y in (batch_progress := tqdm(test_dl, leave=False)):
        i += 1
        batch_progress.set_description(f"Batch {i}")
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        if num_classes is None:
            num_classes = max(out.shape[-1], 2)
        if num_classes == 2:
            if out.ndim == 2 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            predictions.append(out)
        else:
            # Only use argmax predictions to avoid storing large tensors
            predictions.append(torch.argmax(out, dim=-1))
        if isinstance(loss_fun, nn.CrossEntropyLoss):
            y = y.long()
        labels.append(y)
        if isinstance(loss_fun, nn.BCEWithLogitsLoss) or isinstance(
            loss_fun, nn.MSELoss
        ):
            y = y.float()
        total_loss += loss_fun(out, y).item()

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    num_classes = len(labels.unique())
    if num_classes == 2 and predictions.ndim == 1:
        f1 = binary_f1_score(predictions, labels).item()
        accuracy = binary_accuracy(predictions, labels).item()
        auc_score = auc(predictions, labels).item()

        Metric = namedtuple("Metric", ["loss", "f1", "accuracy", "auc"])
        return Metric(total_loss, f1, accuracy, auc_score)
    else:
        f1 = multiclass_f1_score(
            predictions, labels, num_classes=len(labels.unique()), average="macro"
        ).item()
        accuracy = multiclass_accuracy(predictions, labels).item()

        Metric = namedtuple("Metric", ["loss", "f1", "accuracy"])
        return Metric(total_loss, f1, accuracy)


def eval_continuous_nn(model, test_dl, loss=nn.MSELoss):
    model.eval()
    model.to(DEVICE)
    loss_fun = loss()

    predictions = []
    labels = []
    i = 0
    total_loss = 0
    for x, y in (batch_progress := tqdm(test_dl, leave=False)):
        i += 1
        batch_progress.set_description(f"Batch {i}")
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x).squeeze(-1)
        predictions.append(out)
        labels.append(y)
        total_loss += loss_fun(out, y).item()

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    rmse = torch.sqrt(
        mean_squared_error(predictions, labels, multioutput="uniform_average")
    ).item()
    Metric = namedtuple("Metric", ["loss", "rmse"])
    return Metric(total_loss, rmse)
