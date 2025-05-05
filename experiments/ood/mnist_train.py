
from captum.attr import Saliency
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm
from types import MethodType
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F

from txai.explainability import CafeNgExplainer

import captum
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import scipy


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 25

seeds = [42,43,44,45,46]
# models = ["MLPSmall" ,"MLPBig"]
# models = ["CNNSmall"]
# models = ["CNNNarrow", "CNNWide"]
models = ["CNNShort"]
for model_type in models:
    for seed in seeds:

        torch.manual_seed(seed)
        np.random.seed(seed)

        models = []
        dataset, loader = datasets['concatmnist_1']
        model = model_constructor(model_type)
        model.train()
        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=0.001)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        print(f"Training model for dataset: {dataset.name}")

        for epoch in tqdm(range(num_epochs), desc=f"Dataset: {dataset.name}"):
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch+1}, Loss: {running_loss/len(loader)}")

        model_path = f'runs_mnist/model_{model_type}_seed{seed}.pt'
        torch.save(model.state_dict(), model_path)
