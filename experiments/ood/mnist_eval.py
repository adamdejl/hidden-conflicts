from captum.attr import Saliency
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm
from types import MethodType
from PIL import Image

from txai.explainability import CafeNgExplainer

import captum
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import scipy

src_folder = 'runs_mnist'

class StatisticsCollector:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metrics = {
            'predicted': [],
            'labels': [],
            'entropies': [],
            'cl1_activations': [],
            'cl2_activations': [],
            'attr_s_plus': [],
            'attr_s_minus': [],
            'attr_s_star': [],
            'attr_s_conf': [],
            'ig_attrs': [],
            'maxlogit': []
        }

    def update(self, *, predicted, labels, outputs, model, s_plus, s_minus, attr_ig, maxlogit):
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

        self.metrics['predicted'].append(predicted)
        self.metrics['labels'].append(labels)        
        entropy = scipy.stats.entropy(F.softmax(outputs, dim=1).cpu().numpy(), axis=1)
        self.metrics['entropies'].append(torch.tensor(entropy, device=labels.device).cpu())
        self.metrics['cl1_activations'].append(model[-4].activations.mean(dim=-1).cpu())
        self.metrics['cl2_activations'].append(model[-2].activations.mean(dim=-1).cpu())
        self.metrics['attr_s_plus'].append(s_plus.sum(dim=-1).cpu())
        self.metrics['attr_s_minus'].append(s_minus.sum(dim=-1).cpu())
        self.metrics['attr_s_star'].append((s_plus - s_minus).abs().sum(dim=-1))
        self.metrics['attr_s_conf'].append(torch.min(s_plus.sum(dim=-1), s_minus.sum(dim=-1)))
        self.metrics['ig_attrs'].append(attr_ig.abs().sum(dim=-1).cpu())
        self.metrics['maxlogit'].append(maxlogit.cpu())

    def finalize(self):
        for key, value in self.metrics.items():
            self.metrics[key] = torch.cat(value)

    def compute_accuracy(self):
        return self.correct / self.total

    def compute_quantiles(self, quantiles):
        quantile_results = {}
        for key, value in self.metrics.items():
            if key in ['predicted', 'labels']:
                continue
            quantile_results[key] = {q: torch.quantile(value, q) for q in quantiles}
        return quantile_results

    def get_metric_mean(self, metric):
        return self.metrics[metric].mean()


def run_experiment(model, data, data_dist_stats=None):
    dataset, loader = data

    print(f"Running experiment for dataset: {dataset.name}")
    print()

    print("Evaluating model...")

    def cache_forward(self, x):
        activations = self.original_forward(x)
        self.activations = activations
        return activations

    stats = StatisticsCollector()

    with torch.no_grad(), override_forward(model[-2], cache_forward), override_forward(model[-4], cache_forward):
        model.eval()
        cls_model = deepcopy(model[-3:])
        cls_model.eval()

        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            maxlogit, _ = outputs.topk(1, dim=1)
            maxlogit = maxlogit.squeeze().detach().cpu()
            predicted = torch.argmax(outputs.data, 1)

            with torch.enable_grad():
                explainer = CafeNgExplainer(cls_model, c=1., report_discrepancy=True)
                xs = model[-4].activations.detach()
                (s_plus, s_minus), _ = explainer.attribute(xs, target=labels)

            ig = captum.attr.IntegratedGradients(cls_model)
            attr_ig, _ = ig.attribute(xs, target=labels, return_convergence_delta=True)

            stats.update(predicted=predicted, labels=labels, outputs=outputs, model=model, s_plus=s_plus, s_minus=s_minus, attr_ig=attr_ig, maxlogit=maxlogit)

    stats.finalize()

    # Use the current stats as data_dist_stats if it's not provided
    if data_dist_stats is None:
        data_dist_stats = stats

    accuracy = stats.compute_accuracy()
    print(f"Accuracy on dataset: {dataset.name} - {accuracy * 100:.2f}%")

    # Compute quantiles for comparison
    quantiles = [0.1, 0.9]
    current_quantiles = stats.compute_quantiles(quantiles)
    reference_quantiles = data_dist_stats.compute_quantiles(quantiles)

    print(f"Average entropy: {stats.get_metric_mean('entropies'):.4f} ({current_quantiles['entropies'][0.1]:.4f}, {current_quantiles['entropies'][0.9]:.4f})")
    print(f"Average classifier initial hidden layer activation: {stats.get_metric_mean('cl1_activations'):.4f} ({current_quantiles['cl1_activations'][0.1]:.4f}, {current_quantiles['cl1_activations'][0.9]:.4f})")
    print(f"Average classifier final hidden layer activation: {stats.get_metric_mean('cl2_activations'):.4f} ({current_quantiles['cl2_activations'][0.1]:.4f}, {current_quantiles['cl2_activations'][0.9]:.4f})")
    print(f"Average CAFE S+ classifier attribution: {stats.get_metric_mean('attr_s_plus'):.2f} ({current_quantiles['attr_s_plus'][0.1]:.2f}, {current_quantiles['attr_s_plus'][0.9]:.2f})")
    print(f"Average CAFE S- classifier attribution: {stats.get_metric_mean('attr_s_minus'):.2f} ({current_quantiles['attr_s_minus'][0.1]:.2f}, {current_quantiles['attr_s_minus'][0.9]:.2f})")
    print(f"Average CAFE S* classifier attribution: {stats.get_metric_mean('attr_s_star'):.2f} ({current_quantiles['attr_s_star'][0.1]:.2f}, {current_quantiles['attr_s_star'][0.9]:.2f})")
    print(f"Average CAFE classifier conflict: {stats.get_metric_mean('attr_s_conf'):.2f} ({current_quantiles['attr_s_conf'][0.1]:.2f}, {current_quantiles['attr_s_conf'][0.9]:.2f})")
    print(f"Average IG classifier attribution: {stats.get_metric_mean('ig_attrs'):.2f} ({current_quantiles['ig_attrs'][0.1]:.2f}, {current_quantiles['ig_attrs'][0.9]:.2f})")
    print()

    print("Computing outliers...")
    considered_metrics = ['entropies', 'cl1_activations', 'cl2_activations', 'attr_s_plus', 'attr_s_minus', 'attr_s_star', 'attr_s_conf', 'ig_attrs', 'maxlogit']
    for metric in considered_metrics:
        low_q, high_q = reference_quantiles[metric][0.1], reference_quantiles[metric][0.9]
        outliers = torch.nonzero((stats.metrics[metric] < low_q) | (stats.metrics[metric] > high_q)).flatten()
        print(f"Number of outliers for {metric}: {len(outliers)}/{len(stats.metrics[metric])} ({len(outliers) / len(stats.metrics[metric]) * 100:.2f}%)")
    print()

    return stats

def model_constructor(type="CNN"):
    if type == "CNNSmall":
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(28 * 14 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
    if type == "CNNBig":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(28 * 14 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
    if type == "CNNNarrow":
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(28 * 14 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
    if type == "CNNWide":
        return nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(28 * 14 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
    if type == "CNNShort":
        return nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(28 * 14 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform, name="Image Dataset"):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], int(self.labels[idx])
        image = Image.fromarray(image, mode='L')
        image = self.transform(image)
        return image, label
    
def get_transform(data='MNIST'):

    if data == 'MNIST':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.1307,), (0.3081,))
            ]
        )
    return transform

@contextlib.contextmanager
def override_forward(target_layer, new_forward):
    """
    Overrides the forward function of the target layer with the provided forward_attribute function.
    """
    target_layer.original_forward = target_layer.forward
    target_layer.forward = MethodType(new_forward, target_layer)
    try:
        yield
    finally:
        target_layer.forward = target_layer.original_forward
        del target_layer.original_forward


def load_concatmnist(corr_p, split):
    path = f"concatmnist_{corr_p}_{split}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"), name=f"ConcatMNIST c={corr_p}")

def load_fashiondigit5(corr_p, split):
    path = f"digitfashion5_{corr_p}_{split}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"), name=f"FashionDigit c={corr_p}")

keys = (
    'concatmnist_1',
    'concatmnist_0',
    'fashiondigit_0',
)

ds = [
    load_concatmnist(corr_p=1, split='test'),
    load_concatmnist(corr_p=0, split='test'),
    load_fashiondigit5(corr_p=0, split='test')
]

loaders = [
    DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    for dataset in ds
]

datasets = {
    k: (d, l) for k, d, l in zip(keys, ds, loaders)
}

seeds = [42,43,44,45,46]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = []

seeds = [42,43,44,45,46]
# models = ["MLPSmall" ,"MLPBig"]
# models = ["CNNSmall"]
# models = ["CNNBig"]
# models = ["CNNNarrow", "CNNWide"]
models = ["CNNShort"]

for model_type in models:
    for seed in seeds:
        model_path = f'{src_folder}/model_{model_type}_seed{seed}.pt'
        model = model_constructor(model_type)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        stats_base = run_experiment(model, datasets['concatmnist_1'])
        stats_corrshift = run_experiment(model, datasets['concatmnist_0'], data_dist_stats=stats_base)
        stats_semshift = run_experiment(model, datasets['fashiondigit_0'], data_dist_stats=stats_base)

        m = 'maxlogit'
        np.savez(f'{src_folder}/stats/model_{model_type}_{m}_seed{seed}.npy', base=stats_base.metrics[m].cpu(), corr=stats_corrshift.metrics[m].cpu(), sem=stats_semshift.metrics[m].cpu())
        m = 'attr_s_conf'
        np.savez(f'{src_folder}/stats/model_{model_type}_{m}_seed{seed}.npy', base=stats_base.metrics[m].cpu(), corr=stats_corrshift.metrics[m].cpu(), sem=stats_semshift.metrics[m].cpu())

        del model
        del stats_base
        del stats_corrshift
        del stats_semshift
        torch.cuda.empty_cache()