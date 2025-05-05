import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: For some reason, importing this with the default __init__.py from OpenXAI breaks tqdm.
from openxai.dataloader import ReturnLoaders as return_loaders


from txai.experiments import (
    GenericExperimenter,
    MlpConflictPrevalenceMetric,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["adult", "compas", "german", "heloc"]

for dataset in datasets:
    loader_train, loader_test = return_loaders(
        data_name=dataset, download=True, batch_size=32
    )
    train_data_all = torch.FloatTensor(loader_train.dataset.data)
    train_labels_all = torch.LongTensor(loader_train.dataset.targets.values)
    test_data_all = torch.FloatTensor(loader_test.dataset.data)
    test_labels_all = torch.LongTensor(loader_test.dataset.targets.values)

    print(f"———————————————[ Information for dataset {dataset} ]———————————————")
    print(f"Training samples: {len(train_data_all)}")
    print(f"Test samples: {len(test_data_all)}")
    print(f"Total samples: {len(train_data_all) + len(test_data_all)}")
    print(f"# of features: {len(train_data_all[0])}")
    print()

from txai.models import construct_ft_transformer


def run_openai_experiment(dataset):
    print(f"Running experiment for dataset {dataset}")

    loader_train, loader_test = return_loaders(
        data_name=dataset, download=True, batch_size=256
    )

    feature_mask = None
    categorical_variables_spans = []
    if dataset == "compas":
        feature_types = ["c", "d", "c", "c", "d", "d", "d"]
        feature_n_cols = [1] * 7
    elif dataset == "adult":
        feature_types = ["c"] * 6 + ["d"] * 7
        feature_n_cols = [1] * 13
    elif dataset == "synthetic":
        feature_types = ["c"] * 20
        feature_n_cols = [1] * 20
    elif dataset == "heloc":
        feature_types = ["c"] * 23
        feature_n_cols = [1] * 23
    elif dataset == "german":
        feature_types = ["c"] * 8 + ["d"] * 12
        feature_n_cols = [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2]
    else:
        raise ValueError("Currently unsupported dataset")
    if categorical_variables_spans == []:
        current_index = 0
        feature_mask = []
        for i, (t, n) in enumerate(zip(feature_types, feature_n_cols)):
            feature_mask.extend([i] * n)
            if t == "d":
                categorical_variables_spans.append((current_index, current_index + n))
            current_index += n
    else:
        raise ValueError("Unexpected value for categorical_variables_spans")
    if feature_mask is not None:
        feature_mask = torch.tensor([feature_mask])
    else:
        raise ValueError("Unexpected feature_mask")

    def model_constructor():
        ft_transformer = construct_ft_transformer(
            feature_cols_ids=feature_mask.squeeze(), d_out=2, n_layers=2
        )
        ft_transformer = ft_transformer.double()
        ft_transformer.to(DEVICE)
        ft_transformer.train()
        return ft_transformer

    experimenter = GenericExperimenter(
        experiment_name=f"FT-Transformer {dataset.capitalize()}",
        seeds=list(range(42, 42 + 5)),
        num_epochs=100,
        checkpoint_path_base=f"models/ft-transformer-openxai-{dataset}",
        loss=nn.CrossEntropyLoss,
        optimizer_constructor=lambda mp: torch.optim.AdamW(mp),
        model_constructor=model_constructor,
        train_dl=loader_train,
    )

    exp_methods = experimenter.get_explanation_methods(
        use_perturb_methods=True,
        feature_mask=feature_mask.to(DEVICE),
        method_filter=["LRP", "KernelSHAP", "LIME"],
    )

    metrics = experimenter.get_summary_metrics() + [
        MlpConflictPrevalenceMetric(),
    ]

    experimenter.run_experiment(
        exp_methods=exp_methods, metrics=metrics, test_dl=loader_test
    )

    return experimenter


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

experimenter = run_openai_experiment(args.dataset)
experimenter.report_experiment_results(run_stat_tests=False)
print("Done")
