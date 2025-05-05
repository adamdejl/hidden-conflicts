import random
import torch
import torch.nn as nn

from txai.utils.models import ResidualBlock, Slice


def ft_transformer_select_neurons_in(model, data_x, n_samples=50, seed=42):
    layer_models = []
    for i, layer in enumerate(model):
        if isinstance(layer, ResidualBlock):
            for j, sublayer in enumerate(layer.block):
                if isinstance(sublayer, nn.Linear):
                    block_submodel = nn.Sequential(*layer.block[: j + 1])
                    layer_model = nn.Sequential(*model[:i], block_submodel)
                    layer_models.append(layer_model)
        elif isinstance(layer, nn.Linear):
            layer_models.append(nn.Sequential(*model[: i + 1]))

    random.seed(seed)
    test_x = data_x[0]
    test_x = test_x.to(next(model.parameters()).device).unsqueeze(0)
    neuron_models = []
    for i in range(n_samples):
        layer_model = random.choice(layer_models)

        with torch.no_grad():
            test_out = layer_model(test_x)

        if test_out.ndim == 3:
            b, seq_len, d_out = test_out.shape
            seq_idx = random.randint(0, seq_len - 1)
            d_out_idx = random.randint(0, d_out - 1)
            neuron_model = nn.Sequential(
                layer_model, Slice((slice(None), [seq_idx], d_out_idx))
            )
        elif test_out.ndim == 2:
            b, d_out = test_out.shape
            d_out_idx = random.randint(0, d_out - 1)
            neuron_model = nn.Sequential(layer_model, Slice((slice(None), [d_out_idx])))
        else:
            raise ValueError(f"Unsupported test_out shape: {test_out.shape}")

        neuron_models.append(neuron_model)

    return neuron_models


def ft_transformer_select_layers_out(model, data_x):
    layer_models = []
    for i, layer in enumerate(model):
        if isinstance(layer, ResidualBlock) or isinstance(layer, nn.GELU):
            layer_models.append((nn.Sequential(*model[:i]), nn.Sequential(*model[i:])))

    return layer_models
