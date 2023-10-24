import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import get_activation_fn, print_params


class GLU(nn.Module):
    def __init__(self, dim, hidden_dim, act_fun, fina_act="None", dropout=0.0, bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.weight_layer = nn.Linear(dim, hidden_dim, bias=bias)
        self.feature_layer = nn.Linear(dim, hidden_dim, bias=bias)
        self.feature_mapping = nn.Linear(hidden_dim, dim, bias=bias)
        self.act_fun = get_activation_fn(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = get_activation_fn(fina_act) ## None for current usage

    def forward(self, x):
        weight = self.weight_layer(x)
        weight = self.act_fun(weight)
        if self.p > 0.0:
            weight = self.dropout(weight)
        feature = self.feature_layer(x)
        output = weight * feature
        output = self.feature_mapping(output)
        output = self.fina_act(output)

        return output
