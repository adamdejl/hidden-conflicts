import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

from txai.datasets import TitanicDataset
from txai.experiments import (
    MlpConflictPrevalenceMetric,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100

train_dataset, test_dataset = TitanicDataset.create_datasets(
    test_size=0.2,
    split_seed=43,
)
train_dl = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=False,
)
test_dl = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
)

categorical_variables_spans = [(5, 7)]
DATA_DIM = 7
NETWORK_SEEDS = list(range(42, 42 + 5))
NUM_EPOCHS = 100
Experiment = namedtuple("Experiment", ["name", "nn_dims", "activation_fun"])
EXPERIMENTS = [
    Experiment(
        name="titanic-relu-256-hidden", nn_dims=[7, 256, 256, 1], activation_fun=nn.ReLU
    ),
]

from txai.experiments import (
    FeedforwardExperimenter,
    TabularInfidelityMetric,
    SensitivityMetric,
    QuantusMetric,
)

experimenter = FeedforwardExperimenter(
    experiment_name="Titanic",
    seeds=list(range(42, 42 + 5)),
    num_epochs=100,
    checkpoint_path_base="models/titanic-relu-256-hidden-nn",
    loss=nn.BCEWithLogitsLoss,
    data_dim=7,
    num_classes=1,
    num_layers=2,
    hidden_size=256,
    activation_fun=nn.ReLU,
    train_dl=train_dl,
)

feature_mask = torch.tensor([[0, 1, 2, 3, 4, 5, 5]]).to(DEVICE)
exp_methods = experimenter.get_explanation_methods(
    use_perturb_methods=True,
    feature_mask=feature_mask,
    method_filter=["LRP", "KernelSHAP", "LIME"],
)

metrics = experimenter.get_summary_metrics() + [
    MlpConflictPrevalenceMetric(),
]

experimenter.run_experiment(exp_methods=exp_methods, metrics=metrics, test_dl=test_dl)

experimenter.report_experiment_results(run_stat_tests=False)
