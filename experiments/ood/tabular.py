import captum
import contextlib
import itertools
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math 
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from txai.explainability import CafeNgExplainer, CafeUpgradeExplainer
from txai.models import construct_ft_transformer
from types import MethodType

class TabDataset(Dataset):
    def __init__(self, samples, labels, name="Tabular Dataset"):
        super().__init__()
        self.samples = samples
        self.labels = labels
        self.name = name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    

def generate(corr_p, num_encoded_cats, sem_shift, num_samples):

    continuous_feature = torch.randn(num_samples).unsqueeze(dim=1)

    categories = [0,1,2,3,4]
    feature_A = torch.randint(low=0, high=len(categories), size=(num_samples,))

    feature_B = torch.empty(num_samples, dtype=torch.long)
    
    noise_p = (1 - corr_p) / 4
    for i in range(num_samples):
        if feature_A[i] == 0:
            feature_B[i] = torch.multinomial(torch.tensor([corr_p, noise_p, noise_p, noise_p, noise_p]), 1).item() 
        elif feature_A[i] == 1:
            feature_B[i] = torch.multinomial(torch.tensor([noise_p, corr_p, noise_p, noise_p, noise_p]), 1).item() 
        elif feature_A[i] == 2:
            feature_B[i] = torch.multinomial(torch.tensor([noise_p, noise_p, corr_p, noise_p, noise_p]), 1).item() 
        elif feature_A[i] == 3:
            feature_B[i] = torch.multinomial(torch.tensor([noise_p, noise_p, noise_p, corr_p, noise_p]), 1).item() 
        else:
            feature_B[i] = torch.multinomial(torch.tensor([noise_p, noise_p, noise_p, noise_p, corr_p]), 1).item() 

        # if sem_shift:
        #     if feature_A[i] >= 3:
        #         feature_B[i] = 5


    labels = torch.empty(num_samples, dtype=torch.long)
    main_p = 1
    off_p = (1 - main_p) / 4

    # categorical label
    for i in range(num_samples):

        if feature_A[i] == 0:
            labels[i] = torch.multinomial(torch.tensor([main_p, off_p, off_p, off_p, off_p]), 1).item() 
        elif feature_A[i] == 1:
            labels[i] = torch.multinomial(torch.tensor([off_p, main_p, off_p, off_p, off_p]), 1).item() 
        elif feature_A[i] == 2:
            labels[i] = torch.multinomial(torch.tensor([off_p, off_p, main_p, off_p, off_p]), 1).item() 
        elif feature_A[i] == 3:
            labels[i] = torch.multinomial(torch.tensor([off_p, off_p, off_p, main_p, off_p]), 1).item() 
        else:
            labels[i] = torch.multinomial(torch.tensor([off_p, off_p, off_p, off_p, main_p]), 1).item() 


    feature_A = F.one_hot(feature_A, num_classes=num_encoded_cats['A'])
    feature_B = F.one_hot(feature_B, num_classes=num_encoded_cats['B'])

    if sem_shift:
        num_to_zero = int((1-corr_p) * len(feature_B)) 
        indices_to_zero = torch.randperm(len(feature_B))[:num_to_zero]
        feature_B[indices_to_zero] = 0

    samples = torch.concatenate([continuous_feature, feature_A, feature_B], axis=1).float()
    return samples, labels


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


def construct_nn(nn_dims, activation_fun):    
    layers = []
    for i in range(1, len(nn_dims)):
        in_dim, out_dim = nn_dims[i-1], nn_dims[i]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fun())
    # Remove the last activation layer (regression problem)
    layers = layers[:-1]
    
    return nn.Sequential(*layers)

class StatisticsCollector:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metrics = {
            'predicted': [],
            'labels': [],
            'entropies': [],
            # 'cl1_activations': [],
            # 'cl2_activations': [],
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
        
        # self.metrics['cl1_activations'].append(model[-2].activations.mean(dim=-1)) # was -4
        # self.metrics['cl2_activations'].append(model[-2].activations.mean(dim=-1)) # was -2
        self.metrics['attr_s_plus'].append(s_plus.sum(dim=-1))
        self.metrics['attr_s_minus'].append(s_minus.sum(dim=-1))
        self.metrics['attr_s_star'].append((s_plus - s_minus).abs().sum(dim=-1))
        self.metrics['attr_s_conf'].append(torch.min(s_plus.sum(dim=-1), s_minus.sum(dim=-1)))
        self.metrics['ig_attrs'].append(attr_ig.abs().sum(dim=-1))
        self.metrics['maxlogit'].append(maxlogit)

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

    # print(f"Running experiment for dataset: {dataset.name}")
    print()

    print("Evaluating model...")

    def cache_forward(self, x):
        activations = self.original_forward(x)
        self.activations = activations
        return activations

    stats = StatisticsCollector()

    model.to(device)
    # with torch.no_grad(), override_forward(model[-2], cache_forward):
    with torch.no_grad():

        model.eval()
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            maxlogit, _ = outputs.topk(1, dim=1)
            maxlogit = maxlogit.squeeze().detach().cpu()
            predicted = torch.argmax(outputs.data, 1)

            # cls_model = deepcopy(model[-1:]) # last layer
            cls_model = deepcopy(model) # full model
            cls_model.eval()
            with torch.enable_grad():
                explainer = CafeNgExplainer(cls_model, c=1., report_discrepancy=True)
                # xs = model[-2].activations.detach() # last layer
                xs = images.detach() # full model

                (s_plus, s_minus), _ = explainer.attribute(xs, ref=torch.zeros_like(xs), target=labels)

            # cls_model = deepcopy(model[-1:]) # last layer
            cls_model = deepcopy(model) # full model

            cls_model.eval()
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
    # print(f"Average classifier initial hidden layer activation: {stats.get_metric_mean('cl1_activations'):.4f} ({current_quantiles['cl1_activations'][0.1]:.4f}, {current_quantiles['cl1_activations'][0.9]:.4f})")
    # print(f"Average classifier final hidden layer activation: {stats.get_metric_mean('cl2_activations'):.4f} ({current_quantiles['cl2_activations'][0.1]:.4f}, {current_quantiles['cl2_activations'][0.9]:.4f})")
    print(f"Average CAFE S+ classifier attribution: {stats.get_metric_mean('attr_s_plus'):.2f} ({current_quantiles['attr_s_plus'][0.1]:.2f}, {current_quantiles['attr_s_plus'][0.9]:.2f})")
    print(f"Average CAFE S- classifier attribution: {stats.get_metric_mean('attr_s_minus'):.2f} ({current_quantiles['attr_s_minus'][0.1]:.2f}, {current_quantiles['attr_s_minus'][0.9]:.2f})")
    print(f"Average CAFE S* classifier attribution: {stats.get_metric_mean('attr_s_star'):.2f} ({current_quantiles['attr_s_star'][0.1]:.2f}, {current_quantiles['attr_s_star'][0.9]:.2f})")
    print(f"Average CAFE classifier conflict: {stats.get_metric_mean('attr_s_conf'):.2f} ({current_quantiles['attr_s_conf'][0.1]:.2f}, {current_quantiles['attr_s_conf'][0.9]:.2f})")
    print(f"Average IG classifier attribution: {stats.get_metric_mean('ig_attrs'):.2f} ({current_quantiles['ig_attrs'][0.1]:.2f}, {current_quantiles['ig_attrs'][0.9]:.2f})")
    print()

    print("Computing outliers...")
    # considered_metrics = ['entropies', 'cl1_activations', 'cl2_activations', 'attr_s_plus', 'attr_s_minus', 'attr_s_star', 'attr_s_conf', 'ig_attrs', 'maxlogit']
    considered_metrics = ['entropies', 'attr_s_plus', 'attr_s_minus', 'attr_s_star', 'attr_s_conf', 'ig_attrs', 'maxlogit']

    for metric in considered_metrics:
        low_q, high_q = reference_quantiles[metric][0.1], reference_quantiles[metric][0.9]
        outliers = torch.nonzero((stats.metrics[metric] < low_q) | (stats.metrics[metric] > high_q)).flatten()
        print(f"Number of outliers for {metric}: {len(outliers)}/{len(stats.metrics[metric])} ({len(outliers) / len(stats.metrics[metric]) * 100:.2f}%)")
    print()


    return stats


num_features = 11
seeds = [42,43,44,45,46]
model_type = 'ft'
if model_type == 'mlp':
    model_defs = [[num_features, 16, 16, 5], 
                [num_features, 24, 24, 5],
                [num_features, 32, 32, 5],
                [num_features, 40, 40, 5]]
else:
    model_defs = [[4,32], 
                [4,64],
                [8,32],
                [8,64]]


for exp_num, nn_dims in enumerate(model_defs):
    for seed in seeds:
        torch.manual_seed(seed)


        num_encoded_cats = {'A' : 5, 'B': 5}

        train = generate(corr_p=1, num_encoded_cats=num_encoded_cats, sem_shift=False, num_samples=10000)
        train = TabDataset(train[0], train[1])

        corr = generate(corr_p=0., num_encoded_cats=num_encoded_cats, sem_shift=False, num_samples=10000)
        corr = TabDataset(corr[0], corr[1])

        sem = generate(corr_p=0., num_encoded_cats=num_encoded_cats, sem_shift=True, num_samples=10000)
        sem = TabDataset(sem[0], sem[1])

        keys = (
            'train',
            'corr',
            'sem'
        )

        ds = [
            train,
            corr,
            sem
        ]

        loaders = [
            DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
            for dataset in ds
        ]

        datasets = {
            k: (d, l) for k, d, l in zip(keys, ds, loaders)
        }

        activation_fun = nn.ReLU


        d_numerical = 1
        feature_mask = [[i for i in range(d_numerical)]]
        feature_mask += [[i + d_numerical] * v for i, (k,v) in enumerate(num_encoded_cats.items())]
        feature_mask = torch.Tensor(list(itertools.chain(*feature_mask)))



        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_epochs = 25

        models = []
        dataset, loader = datasets['train']
        if model_type == 'mlp':
            model = construct_nn(nn_dims, activation_fun)
        else:
            model = construct_ft_transformer(d_numerical=d_numerical, feature_cols_ids=feature_mask, d_out=5, n_layers=1, n_heads=nn_dims[0], d_model=nn_dims[1])
        model.train()
        model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        print(f"Training model for dataset: {dataset.name}")

        for epoch in tqdm(range(num_epochs), desc=f"Dataset: {dataset.name}"):
            running_loss = 0.0
            for samples, labels in loader:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch+1}, Loss: {running_loss/len(loader)}")

        model_path = f'model-{model_type}-{exp_num}-{seed}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved at {model_path}")

        stats_base = run_experiment(model, datasets['train'])
        stats_corr = run_experiment(model, datasets['corr'], data_dist_stats=stats_base)
        stats_sem = run_experiment(model, datasets['sem'], data_dist_stats=stats_base)

        m = 'attr_s_conf'
        np.savez(f'{m}_{model_type}_{exp_num}_{seed}.npy', base=stats_base.metrics[m].cpu(), corr=stats_corr.metrics[m].cpu(), sem=stats_sem.metrics[m].cpu())