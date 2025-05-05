import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantus.metrics.complexity.complexity import Complexity
from torch.utils.data import Dataset, DataLoader

from txai.datasets import CovertypeDataset
from txai.experiments import (
    TabularInfidelityMetric,
    InputStructuralInfidelityMetric,
    SensitivityMetric,
    QuantusMetric,
    ft_transformer_select_neurons_in,
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

from txai.experiments import (
    FeedforwardExperimenter,
    TabularInfidelityMetric,
    SensitivityMetric,
    QuantusMetric,
)

experimenter = FeedforwardExperimenter(
    experiment_name="Covertype",
    seeds=list(range(42, 42 + 5)),
    num_epochs=100,
    checkpoint_path_base="models/covertype-gelu-4-256-hidden-nn",
    loss=nn.CrossEntropyLoss,
    data_dim=54,
    num_classes=7,
    num_layers=4,
    hidden_size=256,
    activation_fun=nn.GELU,
    train_dl=train_dl,
)

feature_mask = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 4 + [11] * 40]).to(
    DEVICE
)
exp_methods = experimenter.get_explanation_methods(
    use_perturb_methods=True,
    feature_mask=feature_mask,
    method_filter=["LRP", "KernelSHAP", "LIME"],
)

metrics = experimenter.get_summary_metrics() + [
    TabularInfidelityMetric(
        perturbation_scale=0.5,
        cat_resample_proba=0.1,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
        normalize=True,
    ),
    TabularInfidelityMetric(
        perturbation_scale=0.75,
        cat_resample_proba=0.2,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
    ),
    # FT-Transformer neuron selection is also compatible with FFNNs
    InputStructuralInfidelityMetric(
        neuron_selection_fun=ft_transformer_select_neurons_in,
        n_neuron_samples=50,
        perturbation_scale=0.5,
        cat_resample_proba=0.1,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
        neuron_aggregate="mean",
    ),
    InputStructuralInfidelityMetric(
        neuron_selection_fun=ft_transformer_select_neurons_in,
        n_neuron_samples=50,
        perturbation_scale=0.75,
        cat_resample_proba=0.2,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
        neuron_aggregate="mean",
    ),
    SensitivityMetric(
        n_perturb_samples=2,
    ),
    QuantusMetric(
        name="Complexity",
        metric_evaluator=Complexity(),
        higher_is_better=False,
    ),
]

experimenter.run_experiment(exp_methods=exp_methods, metrics=metrics, test_dl=test_dl)

experimenter.report_experiment_results(run_stat_tests=True)
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()

basic_experimenter = FeedforwardExperimenter(
    experiment_name="Covertype (Basic Metrics)",
    seeds=list(range(42, 42 + 5)),
    num_epochs=100,
    checkpoint_path_base="models/covertype-gelu-4-256-hidden-nn",
    loss=nn.CrossEntropyLoss,
    data_dim=54,
    num_classes=7,
    num_layers=4,
    hidden_size=256,
    activation_fun=nn.GELU,
    train_dl=train_dl,
)

feature_mask = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [10] * 4 + [11] * 40]).to(
    DEVICE
)
all_exp_methods = basic_experimenter.get_explanation_methods(
    use_perturb_methods=True,
    feature_mask=feature_mask,
)

limited_metrics = basic_experimenter.get_summary_metrics() + [
    TabularInfidelityMetric(
        perturbation_scale=0.5,
        cat_resample_proba=0.1,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
        normalize=True,
    ),
    TabularInfidelityMetric(
        perturbation_scale=0.75,
        cat_resample_proba=0.2,
        categorical_variables_spans=categorical_variables_spans,
        n_perturb_samples=20,
        test_dl=test_dl,
    ),
    SensitivityMetric(
        n_perturb_samples=2,
    ),
    QuantusMetric(
        name="Complexity",
        metric_evaluator=Complexity(),
        higher_is_better=False,
    ),
]

basic_experimenter.run_experiment(
    exp_methods=all_exp_methods, metrics=limited_metrics, test_dl=test_dl
)

basic_experimenter.report_experiment_results(run_stat_tests=True)

print("Done")
