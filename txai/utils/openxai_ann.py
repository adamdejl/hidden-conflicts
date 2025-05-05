import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

def load_openxai_ann(dataset, device='cpu'):
    checkpoint_path = f"models/openxai_pretrained/ann_{dataset}.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    d_h, d_in = state_dict['input1.weight'].shape
    d_out, _ = state_dict['input2.weight'].shape
    model = nn.Sequential(
        nn.Linear(d_in, d_h),
        nn.ReLU(),
        nn.Linear(d_h, d_out),
    )
    model[0].weight = nn.Parameter(state_dict['input1.weight'])
    model[0].bias = nn.Parameter(state_dict['input1.bias'])
    model[2].weight = nn.Parameter(state_dict['input2.weight'])
    model[2].bias = nn.Parameter(state_dict['input2.bias'])
    model.eval()
    return model

class OpenXaiAnn(nn.Sequential):

    @classmethod
    def construct_model(cls, data_name, pretrained=True):
        from openxai.model import LoadModel
        model = LoadModel(data_name=data_name, ml_model='ann', pretrained=pretrained)
        return cls(
            *[m if not isinstance(m, nn.ReLU) else nn.ReLU() for m in model.network],
            nn.Softmax(dim=1),
        )

    def predict_with_logits(self, x: torch.FloatTensor):
        """
        Forwardpass through the network
        :param input: tabular data
        :return: prediction
        """
        return self[:3](x)

    def predict_proba(self, data: Union[torch.FloatTensor, np.array]) -> np.array:
        """
        Computes probabilistic output for c classes
        :param data: torch tabular input
        :return: np.array
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = data.float()

        return self(input).detach().numpy()

    def predict(self, data):
        """
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data).float()
        
        return self(input).detach().numpy()