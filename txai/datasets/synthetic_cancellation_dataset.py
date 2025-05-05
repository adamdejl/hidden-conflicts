import math
import numpy as np
import torch
import torch.nn.functional as F
import warnings

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader


class SyntheticCancellationDataset(Dataset):
    def __init__(
        self,
        samples,
        labels,
        ground_truth_attributions,
        dist,
        weights,
    ):
        self.samples = samples
        self.labels = labels
        self.ground_truth_attributions = ground_truth_attributions
        self.dist = dist
        self.weights = weights

    @classmethod
    def generate(
        cls,
        num_samples,
        seed=42,
        continuous_dim=2,
        cancellation_features=None,
        cancellation_likelihood=0.5,
        standard_dev=1.0,
        weight_range=(-1.0, 1.0),
        one_hot=False,
    ):
        torch.manual_seed(seed)
        samples = []
        labels = []
        ground_truth_attributions = []

        if cancellation_features is None:
            cancellation_features = [[n] for n in range(continuous_dim)]

        # Initialise random distribution for samples
        dist = MultivariateNormal(
            torch.zeros(continuous_dim), standard_dev**2 * torch.eye(continuous_dim)
        )

        # Randomly generate feature weights
        b1, b2 = weight_range
        weights = (b2 - b1) * torch.rand(continuous_dim) + b1

        # Generate cancellation feature masks
        if len(cancellation_features) > continuous_dim:
            warnings.warn(
                "There are more cancellation features than regular features. This can"
                "cause cancellation features confounding and is not recommended."
            )
        cancellation_masks = []
        for feature_idxs in cancellation_features:
            mask = torch.ones(continuous_dim)
            mask[feature_idxs] = 0
            cancellation_masks.append(mask)

        # Generate samples
        for i in range(num_samples):
            # Sample from a normal distribution
            base_sample = dist.sample()
            weighted_sample = base_sample * weights
            masked_sample = base_sample.clone().detach()

            # Determine cancel features
            cancellation_values = []
            cancellation_attributions = []
            for mask in cancellation_masks:
                if cancellation_likelihood > torch.rand(1).item():
                    masked_sample *= mask
                    cancellation_values.append(1.0)
                    cancellation_attributions.append(
                        -(weighted_sample * (1 - mask)).sum().item()
                    )
                    if one_hot:
                        cancellation_values.append(0.0)
                        cancellation_attributions.append(0.0)
                else:
                    cancellation_values.append(0.0)
                    cancellation_attributions.append(0.0)
                    if one_hot:
                        cancellation_values.append(1.0)
                        cancellation_attributions.append(0.0)

            # Construct final sample and compute its label
            samples.append(torch.cat((base_sample, torch.tensor(cancellation_values))))
            labels.append((masked_sample @ weights).item())
            ground_truth_attributions.append(
                torch.cat((weighted_sample, torch.tensor(cancellation_attributions)))
            )

        return cls(
            samples,
            labels,
            ground_truth_attributions,
            dist,
            weights,
        )

    # TODO: Consider refactoring so that default split method works
    def split(self, split_ratio):
        num_samples = int(len(self.samples) * split_ratio)
        d1 = SyntheticCancellationDataset(
            self.samples[:num_samples],
            self.labels[:num_samples],
            self.ground_truth_attributions[:num_samples],
            self.dist,
            self.weights,
        )
        d2 = SyntheticCancellationDataset(
            self.samples[num_samples:],
            self.labels[num_samples:],
            self.ground_truth_attributions[num_samples:],
            self.dist,
            self.weights,
        )
        return d1, d2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
