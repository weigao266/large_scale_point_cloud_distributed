# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .helpers import ActLayer, get_activation_fn, print_params
from .rpe import Rpe

from scipy.fftpack import next_fast_len

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    # fast_len = next_fast_len(N + M - 1)
    fast_len = M

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    # f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)

    # f_x = torch.fft.rfft(x, n = 2*N, dim = dim)
    # f_weight = torch.fft.rfft(weights, n = 2*N, dim = weight_dim).unsqueeze(0)

    f_v_weight = f_x * f_weight

    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out[:,:,:N,:]

    # out = out.roll(-1, dims = (dim,))
    # indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    # out = out.index_select(dim, indices)

    return out

class Tno(nn.Module):
    def __init__(
        self, h, dim, rpe_dim, causal=False, use_decay=False, use_multi_decay=False, residual=False, 
        act="relu", par_type=1, gamma=0.999, bias=True, act_type="none", layers=3, norm_type="simplermsnorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.h = h
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(h, 1, dim) * gamma, requires_grad=False)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(h, 1, dim))

        self.rpe = Rpe(dim=rpe_dim, outdim=h * dim, residual=residual,
            act=act, bias=bias, layers=layers, norm_type=norm_type,
        )
        
        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal
            
        self.act_fun = get_activation_fn(act_type)

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)
        
        return index
        
    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)
            
        return index

    def get_neg(self, n):
        if self.causal:
            index = torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim) * self.zero_value
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index
    
    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, 'n (h d) -> h n d', h=self.h)

        return res
    
    def forward_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos, zero], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)
        # output = x

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        
    # def forward_non_causal(self, x, dim=-2, normalize=False):
    #     # x: b, h, n, d
    #     n = x.shape[dim]
    #     # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
    #     ##### coef
    #     # 1, d, 1 -> h, 1, d
    #     zero = self.rpe_transform(self.get_zero().to(x))
    #     pos = self.rpe_transform(self.get_pos(n - 1).to(x))

    #     neg_index = self.get_neg(n - 1).to(x)
    #     if self.causal:
    #         neg = neg_index
    #     else:
    #         neg = self.rpe_transform(neg_index)

    #     if self.use_decay or self.use_multi_decay:
    #         coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
    #         if self.use_decay:
    #             gamma = self.gamma
    #         else:
    #             gamma = torch.sigmoid(self.gamma)
    #             gamma = self.lambda_ + (1 - self.lambda_) * gamma
    #         gamma = gamma ** coef
    #         pos = gamma * pos
    #         neg = torch.flip(gamma, dims=[1]) * neg

    #     a = torch.cat([zero, pos, zero, neg], dim=1)
    #     a = self.act_fun(a)
    #     # x: b, h, n, d
    #     # a: h, l, d
    #     output = self.compute(x, a, dim, n)
    #     # output = x

    #     if normalize:
    #         size = list(x.shape[:-1]) + [1]
    #         ones = torch.ones(size).to(x)
    #         denorm = self.compute(ones, a, dim, n)
    #         output = output / denorm

    #     return output

    def forward_non_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        # print('==========================')
        # print(x.size())
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)
        
        m = next_power_of_2(n)
        pos = F.pad(pos, (0, 0, 0, m - n, 0, 0))
        neg = F.pad(neg, (0, 0, 0, m - n, 0, 0))
        x = F.pad(x, (0, 0, 0, m - n, 0, 0, 0, 0))
        
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, m).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        a = torch.cat([zero, pos, zero, neg], dim=1)
        a = self.act_fun(a)
        # x: b, h, n, d
        # a: h, l, d
        # print(x.size())
        # print(a.size())
        output = self.compute(x, a, dim, m)[:, :, :n, :]
        # print(output.size())

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, m)
            output = output / denorm

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, 2*n, d
        # n_adj = next_power_of_2(n)
        # print(n_adj)
        # print('---------------')
        # print(x.size())
        # print(a.size())
        # print(x.size())
        # print(n)
        # print('-------------------------------')
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        # print('-----------------')
        # print(y.size())
        # print(v.size())
        u = v * y
        # print(u.size())
        # print('++++++++++++++++++++')
        tmp = torch.fft.irfft(u, 2 * n, dim=dim)
        # print(tmp.size())
        output = tmp[:, :, :n, :]
        # print(output.size())

        # test = conv1d_fft(x, a, dim=dim, weight_dim=dim)
        # print('=====================')
        # print(output.size())
        # print(test.size())
        # print((test-output).sum())

        # torch.backends.cuda.cufft_plan_cache[2].max_size = 32

        return output

    def toeplizt_matrix(self, x, dim):
        assert dim == -2
        # shape of x: b, h, n, d
        n = x.shape[dim]
        # c: first col, r: first row
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)
                
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        zero = self.act_fun(zero)
        pos = self.act_fun(pos)
        if not self.causal:
            neg = self.act_fun(neg)
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(self.h, n, n, -1)

        res = torch.einsum('h n m d, b h m d -> b h n d', T, x)
        return res
