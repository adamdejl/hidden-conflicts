import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from txai.datasets import TitanicDataset
from txai.experiments import (
    GenericExperimenter,
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

from txai.models import construct_ft_transformer

categorical_variables_spans = [(5, 7)]


def model_constructor():
    feature_mask = torch.tensor([[0, 1, 2, 3, 4, 5, 5]])
    ft_transformer = construct_ft_transformer(
        feature_cols_ids=feature_mask, d_out=1, n_layers=2
    )
    ft_transformer.to(DEVICE)
    ft_transformer.train()
    return ft_transformer


experimenter = GenericExperimenter(
    experiment_name="FT-Transformer Titanic",
    seeds=list(range(42, 42 + 5)),
    num_epochs=100,
    checkpoint_path_base="models/ft-transformer-titanic",
    loss=nn.BCEWithLogitsLoss,
    optimizer_constructor=lambda mp: torch.optim.AdamW(mp),
    model_constructor=model_constructor,
    train_dl=train_dl,
)

exp_methods = experimenter.get_explanation_methods(
    use_perturb_methods=True,
    feature_mask=torch.tensor([[0, 1, 2, 3, 4, 5, 5]]).to(DEVICE),
    method_filter=["LRP", "KernelSHAP", "LIME"],
)

metrics = experimenter.get_summary_metrics() + [
    MlpConflictPrevalenceMetric(),
]

experimenter.run_experiment(exp_methods=exp_methods, metrics=metrics, test_dl=test_dl)

experimenter.report_experiment_results(run_stat_tests=False)
print("Done")
