import torch

from scipy.fftpack import next_fast_len

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def compute(x, a, dim, n_out, n_fft, n_ifft=None):
    y = torch.fft.rfft(x, n_fft, dim=dim)
    v = torch.fft.rfft(a, n_fft, dim=dim).unsqueeze(0)
    u = v*y
    output = torch.fft.irfft(u, n_ifft, dim=dim)
    return output[:,:,:n_out,:]

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)
    print(fast_len)
    # fast_len = M

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)

    # f_x = torch.fft.rfft(x, n = 2*N, dim = dim)
    # f_weight = torch.fft.rfft(weights, n = 2*N, dim = weight_dim).unsqueeze(0)
    # f_v_weight = f_x * f_weight

    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    # out = out[:,:,:N,:]

    out = out.roll(-1, dims = (dim,))
    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)

    return out


# x = torch.randn([1,8, 200, 7])
# k1 = torch.randn([8, 400, 7])

# k2 = torch.randn([8, 200, 7])

# res1 = compute(x, k1, dim=-2, n_out=200, n_fft=200)
# res2 = compute(x, k1, dim=-2, n_out=200, n_fft=400)
# print((res1-res2).sum())

# res1 = compute(x, k2, dim=-2, n_out=200, n_fft=600)
# res_same_size = compute(x, k2, dim=-2, n_out=200, n_fft=600)
# print((res1-res_same_size).sum())

print('Ground truth: set N=None in FFT functions, printed value: sum of different between ground truth and different sequence length choice.')

print('Case 1: equivalent sequence length')

x = torch.randn([1,8, 256, 7])
k1 = torch.randn([8, 256, 7])

# k2 = torch.randn([8, 200, 7])

# print(next_fast_len(x.size(-2)+k1.size(-2)-1))
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=256)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        ----------------In FFT, N=sequence length----------------')
print('        In iFFT, N=sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=512)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        In iFFT, N=2*sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=400)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        In iFFT, sequence length < N < 2*sequence length:', str((res1-res2).sum().item()))

print('        ----------------In FFT, sequence length < N < 2*sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=400)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        iFFT length = FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=800)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        iFFT length = 2 * FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        FFT length < iFFT length < 2 * FFT length:', str((res1-res2).sum().item()))


print('        ----------------In FFT, N >= 2 * sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=256)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        N = 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=1024, n_ifft=256)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        N > 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))


res1 = compute(x, k1, dim=-2, n_out=256, n_fft=800, n_ifft=256)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        N = a non power of 2 value (but > 2 * sequence length) in both FFT and iFFT:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=2048, n_ifft=2048)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=1024, n_ifft=1024)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=None, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res3 = conv1d_fft(x, k1, dim = -2, weight_dim = -2)
print(str((res3-res2).sum().item()))

print('===================================================================================')
print('Case 2: kernel length = 2 * sequence length')

x = torch.randn([1,8, 256, 7])
k1 = torch.randn([8, 512, 7])

# k2 = torch.randn([8, 200, 7])

# print(next_fast_len(x.size(-2)+k1.size(-2)-1))
print('        ----------------In FFT, N=sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=256)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        In iFFT, N=sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=512)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        In iFFT, N=2*sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=256, n_ifft=400)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        In iFFT, sequence length < N < 2*sequence length:', str((res1-res2).sum().item()))

print('        ----------------In FFT, sequence length < N < 2*sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=400)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        iFFT length = FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=800)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        iFFT length = 2 * FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=400, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        FFT length < iFFT length < 2 * FFT length:', str((res1-res2).sum().item()))


print('        ----------------In FFT, N >= 2 * sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=512)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        N = 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=1024, n_ifft=512)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        N > 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))


res1 = compute(x, k1, dim=-2, n_out=256, n_fft=800, n_ifft=512)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        N = a non power of 2 value (but > 2 * sequence length) in both FFT and iFFT:', str((res1-res2).sum().item()))


res1 = compute(x, k1, dim=-2, n_out=256, n_fft=2048, n_ifft=2048)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=256, n_fft=1024, n_ifft=1024)
res2 = compute(x, k1, dim=-2, n_out=256, n_fft=512, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res3 = conv1d_fft(x, k1, dim = -2, weight_dim = -2)
print(str((res3-res2).sum().item()))


print('===================================================================================')
print('Case 3: equivalent sequence length, but not power of 2')

x = torch.randn([1,8, 300, 7])
k1 = torch.randn([8, 300, 7])

# k2 = torch.randn([8, 200, 7])

# print(next_fast_len(x.size(-2)+k1.size(-2)-1))
res1 = compute(x, k1, dim=-2, n_out=300, n_fft=300, n_ifft=300)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        ----------------In FFT, N=sequence length----------------')
print('        In iFFT, N=sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=300, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        In iFFT, N=2*sequence length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=300, n_ifft=500)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        In iFFT, sequence length < N < 2*sequence length:', str((res1-res2).sum().item()))

print('        ----------------In FFT, sequence length < N < 2*sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=300, n_fft=400, n_ifft=300)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        iFFT length = FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=400, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        iFFT length = 2 * FFT length:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=400, n_ifft=500)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        FFT length < iFFT length < 2 * FFT length:', str((res1-res2).sum().item()))


print('        ----------------In FFT, N >= 2 * sequence length----------------')
res1 = compute(x, k1, dim=-2, n_out=300, n_fft=600, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        N = 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=1200, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        N = 2 * sequence length in both FFT and iFFT:', str((res1-res2).sum().item()))


res1 = compute(x, k1, dim=-2, n_out=300, n_fft=800, n_ifft=600)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        N = a non power of 2 value (but > 2 * sequence length) in both FFT and iFFT:', str((res1-res2).sum().item()))



res1 = compute(x, k1, dim=-2, n_out=300, n_fft=2048, n_ifft=2048)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res1 = compute(x, k1, dim=-2, n_out=300, n_fft=1024, n_ifft=1024)
res2 = compute(x, k1, dim=-2, n_out=300, n_fft=None, n_ifft=None)
print('        extreme large n:', str((res1-res2).sum().item()))

res3 = conv1d_fft(x, k1, dim = -2, weight_dim = -2)
print(str((res3-res2).sum().item()))