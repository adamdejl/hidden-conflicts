from captum.attr import Saliency
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm
from types import MethodType
from PIL import Image
from scipy.spatial.distance import jensenshannon
from txai.explainability import CafeNgExplainer

import captum
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import scipy
from scipy.interpolate import make_interp_spline

fig, ax = plt.subplots(2,4, figsize=(15,6))
lw = 2
fontsize = 12

ms = ['maxlogit', 'attr_s_conf']

src_folder = 'runs_mnist'

allBins1 = [[0 + 2 * i for i in range(30)], [0 + 2 * i for i in range(30)], [0 + 2 * i for i in range(30)], [0 + 2 * i for i in range(30)]]
allBins2 = [[10 + 3 * i for i in range(30)], [16 + 3 * i for i in range(30)], [10 + 2 * i for i in range(30)], [10 + 2.5 * i for i in range(30)]]
allBinsList = [allBins1, allBins2]

titles = ["CNN Shallow", "CNN Deep", "CNN Narrow", "CNN Wide"]
seeds = [42,43,44,45,46]
models = ["CNNShort" ,"CNNWide", "CNNSmall", "CNNBig"]

for j, m in enumerate(ms):
    allBins = allBinsList[j]
    for i, model_type in enumerate(models):
        bins = allBins[i]
        stats_base = []
        stats_corrshift = []
        stats_semshift = []
        for seed in seeds:
            stats = np.load(f'{src_folder}/stats/model_{model_type}_{m}_seed{seed}.npy.npz')
            stats_base.append(stats['base'])
            stats_corrshift.append(stats['corr'])
            stats_semshift.append(stats['sem'])
        
        stats_base = np.concatenate(stats_base)
        stats_corrshift = np.concatenate(stats_corrshift)
        stats_semshift = np.concatenate(stats_semshift)

        hist1 = np.histogram(stats_base, bins=bins)
        hist2 = np.histogram(stats_corrshift, bins=bins)
        hist3 = np.histogram(stats_semshift, bins=bins)
        kl_sem = jensenshannon(hist3[0], hist1[0])
        kl_corr = jensenshannon(hist2[0], hist1[0])

        ax[j][i].hist(bins[:-1], bins, weights=hist1[0], alpha=0.5, label='In-distribution', edgecolor='white', color='black')
        ax[j][i].hist(bins[:-1], bins, weights=hist3[0], alpha=0.5, label='Sem. shift.', edgecolor='white', color='blue')
        ax[j][i].hist(bins[:-1], bins, weights=hist2[0], alpha=0.5, label='Corr. shift', edgecolor='white', color='orange')
        if i == 3 and j == 0:
            ax[j][i].legend(fontsize=10, loc='upper right')
        ax[j][i].set_xticklabels('')
        ax[j][i].set_xticks([])
        if i == 0:
            ax[j][i].set_ylabel('Frequency', fontsize=fontsize)
        ax[j][i].set_yticklabels('')
        ax[j][i].set_yticks([])
        ax[j][i].set_ylim(top=3000)
        if j == 0:
            ax[j][i].set_title(titles[i], fontsize=fontsize)
            ax[j][i].set_xlabel('Max Logit', fontsize=fontsize)
    
        if j == 1:
            ax[j][i].set_xlabel('Conflict Score', fontsize=fontsize)
        ax[j][i].spines[['right', 'top']].set_visible(False)
        ax[j][i].text(0.75, 0.6, f"JSD: {kl_sem:.2f}", color='blue', ha='left', va='center', transform=ax[j][i].transAxes, fontsize=fontsize)
        ax[j][i].text(0.75, 0.5, f"JSD: {kl_corr:.2f}", color='orange', ha='left', va='center', transform=ax[j][i].transAxes, fontsize=fontsize)

    plt.tight_layout()
    fig.savefig(f"{src_folder}/figs/fig.png", dpi=300)
