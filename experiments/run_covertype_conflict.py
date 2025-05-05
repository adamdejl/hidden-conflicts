import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from txai.datasets import CovertypeDataset
from txai.experiments import (
    GenericExperimenter,
    MlpConflictPrevalenceMetric,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

categorical_variables_spans = [(10, 14), (14, 54)]

train_dataset, test_dataset = CovertypeDataset.create_datasets(
    test_size=0.02, split_seed=43
)
train_dl = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=False,
)
test_dl = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
)

from txai.models import construct_ft_transformer


def model_constructor():
    feature_mask = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 4 + [11] * 40])
    ft_transformer = construct_ft_transformer(
        feature_cols_ids=feature_mask, d_out=7, n_layers=2
    )
    ft_transformer.to(DEVICE)
    ft_transformer.train()
    return ft_transformer


experimenter = GenericExperimenter(
    experiment_name="FT-Transformer Covertype",
    seeds=list(range(42, 42 + 5)),
    num_epochs=100,
    checkpoint_path_base="models/ft-transformer-covertype",
    loss=nn.CrossEntropyLoss,
    optimizer_constructor=lambda mp: torch.optim.AdamW(mp),
    model_constructor=model_constructor,
    train_dl=train_dl,
)

exp_methods = experimenter.get_explanation_methods(
    use_perturb_methods=True,
    feature_mask=torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 4 + [11] * 40]
    ).to(DEVICE),
    method_filter=["LRP", "KernelSHAP", "LIME"],
)

metrics = experimenter.get_summary_metrics() + [
    MlpConflictPrevalenceMetric(),
]

experimenter.run_experiment(exp_methods=exp_methods, metrics=metrics, test_dl=test_dl)

experimenter.report_experiment_results(run_stat_tests=False)
print("Done")
