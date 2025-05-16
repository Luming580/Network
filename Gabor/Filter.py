import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

def sobel_x():
    """Sobel x kernel"""
    kernel = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
    return kernel.view(1, 1, 3, 3)


def sobel_y():
    """Sobel y kernel"""
    kernel = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32)
    return kernel.view(1, 1, 3, 3)


def calculate_gradients(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    sobel_kernel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)

    g_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1)
    g_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1)

    M = torch.abs(g_x) + torch.abs(g_y)

    Diff = M / image_tensor

    MN = image_tensor.shape[2] * image_tensor.shape[3]
    Q = torch.sum(torch.sum(Diff)) / (MN + 2)

    return Q


def adaptive_window_width(feature_block):
    device = feature_block.device
    kernel_size = feature_block.shape[1]
    base = feature_block.shape[2]
    sobel_kernel_x = sobel_x().to(device).repeat(kernel_size, 1, 1, 1)
    sobel_kernel_y = sobel_y().to(device).repeat(kernel_size, 1, 1, 1)

    grad_x = F.conv2d(feature_block, sobel_kernel_x, padding=1, groups=kernel_size)
    grad_y = F.conv2d(feature_block, sobel_kernel_y, padding=1, groups=kernel_size)

    grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)

    max_grad = torch.max(grad_magnitude.view(grad_magnitude.shape[0], -1), dim=1)[0].mean()

    hnw = np.sqrt(base) * (1 - torch.exp(-0.5 * max_grad))
    lww = np.sqrt(base) * (1 + torch.exp(-0.5 * max_grad))

    return hnw, lww


class GlobalFilter(nn.Module):
    def __init__(self, dim, size=32, num=4, len=128, number=16, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()

        assert dim % num_blocks == 0, f"hidden_size {dim} should be divisble by num_blocks {num_blocks}"

        self.len = len
        self.number = number
        self.dim = dim
        self.size = size
        self.num = num

        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.dim // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size))
        self.relu = nn.ReLU(inplace=True)

        self.raw_val1 = DynamicRangeModule(1024, min_value=4.6, max_value=5.6)
        self.raw_val2 = DynamicRangeModule(512, min_value=7.0, max_value=8.0)
        self.raw_val3 = DynamicRangeModule(256, min_value=10.3, max_value=11.3)
        self.raw_val4 = DynamicRangeModule(128, min_value=15.0, max_value=16.0)

        self.raw_val21 = DynamicRangeModule2(1024, min_value=5.6, max_value=7)
        self.raw_val22 = DynamicRangeModule2(512, min_value=8.0, max_value=10.0)
        self.raw_val23 = DynamicRangeModule2(256, min_value=11.3, max_value=13.0)
        self.raw_val24 = DynamicRangeModule2(128, min_value=16.0, max_value=16.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bias = x
        B = x.shape[0]
        C = x.shape[1]
        a = x.shape[2]
        b = x.shape[3]
        x = x.reshape(B, a, b, C)
        val1, val2 = adaptive_window_width(x)
        if a == 32:
            rr1, hh1 = analysis_window_r0(val1, len=self.len, number=self.number)
            rr2, hh2 = analysis_window_r0(val2, len=self.len, number=self.number)
        elif a == 64:
            rr1, hh1 = analysis_window_r0(val1, len=self.len, number=self.number)
            rr2, hh2 = analysis_window_r0(val2, len=self.len, number=self.number)
        elif a == 128:
            rr1, hh1 = analysis_window_r0(val1, len=self.len, number=self.number)
            rr2, hh2 = analysis_window_r0(val2, len=self.len, number=self.number)
        else:
            rr1, hh1 = analysis_window_r0(val1, len=self.len, number=self.number)
            rr2, hh2 = analysis_window_r0(val2, len=self.len, number=self.number)

        rr = rr1 + rr2
        hh = hh1 + hh2
        agray = torch.zeros((B, self.num, self.num, self.size, self.size, C)).to(device)  # (B,4,4,8,8,C)
        for m in range(self.num):
            for n in range(self.num):
                m_range = np.mod(np.arange(-m * self.size, a - m * self.size), a)
                n_range = np.mod(np.arange(-n * self.size, b - n * self.size), b)
                R = x * rr[m_range][:, n_range].unsqueeze(-1)

                R1 = torch.zeros(1, self.size, self.size, C).to(device)  # (1,16,16,512)
                for i1 in range(self.num):  # 0->7
                    for i2 in range(self.num):  # 0->7
                        R1 = R1 + R[:, i1 * self.size:(i1 + 1) * self.size,
                                  i2 * self.size:(i2 + 1) * self.size, :]
                        # (B, 16, 16, C)
                FT = torch.fft.fftn(R1, dim=(1, 2))
                agray[:, m, n, :, :, :] = torch.real(FT) - torch.imag(FT)
        agray1 = agray.reshape(B, a, b, self.num_blocks, self.block_size)

        o1_real = F.relu(torch.einsum('...bi,bio->...bo', agray1[:, :, :], self.w1) + self.b1)
        o2_real = (torch.einsum('...bi,bio->...bo', o1_real[:, :, :], self.w2) + self.b2)

        x = F.softshrink(o2_real, lambd=self.sparsity_threshold)
        x = x.reshape(B, self.num, self.num, self.size, self.size, C)

        aa = torch.zeros((B, self.num, self.num, self.size, self.size, C)).to(device)

        for m in range(self.num):  # 0->1
            for n in range(self.num):  # 0->1
                x1 = x[:, m, n, :, :, :].view(B, self.size, self.size, C)
                Rf = torch.fft.fft2(x1, dim=(1, 2))
                aa[:, m, n, :, :, :] = torch.real(Rf) - torch.imag(Rf)

        I = None
        for qx in range(self.num):
            dw = None
            for qy in range(self.num):
                II = torch.zeros(self.size, self.size).unsqueeze(0).unsqueeze(-1).to(device)
                for m in range(self.num):
                    for n in range(self.num):
                        gmn = torch.tensor(hh[torch.arange((qx - m) * self.size, (qx + 1 - m) * self.size) % a,
                                              torch.arange((qy - n) * self.size, (qy + 1 - n) * self.size) % b])

                        aa1 = aa[:, m, n, :, :, :].view(B, self.size, self.size, C)
                        II = II + aa1 * gmn.unsqueeze(0).unsqueeze(-1)

                if qy == 0:
                    dw = II
                else:
                    dw = torch.cat((dw, II), dim=2)

            if qx == 0:
                I = dw
            else:
                I = torch.cat((I, dw), dim=1)

        x = I.float()  # torch.Size([1, 64, 64, 512])
        x = x.reshape(B, x.shape[3], x.shape[1], x.shape[2])

        return x + bias


def analysis_window_r0(q, len, number):
    device = q.device  # Assuming `q` is on the correct device
    # Parameters
    n1 = len
    n2 = number
    L = n1
    N = n2
    N0 = n2
    M = int(L / N0)
    M0 = int(L / N)
    q = q.to(device)  # Ensure q is on the correct device

    # Initialize V vector (ensure it's complex)
    V = torch.zeros((M0 * N0, 1), dtype=torch.complex64, device=device)
    V[0] = L / (M * N)

    # Create matrix H using broadcasting
    k = torch.arange(L, dtype=torch.float32, device=device)
    m = torch.arange(M0, dtype=torch.float32, device=device)
    n = torch.arange(N0, dtype=torch.float32, device=device)

    k1 = (k[None, :] + m[:, None] * N) % L
    exp_term = torch.sqrt(torch.sqrt(torch.tensor(3.0, device=device)) / q) * torch.exp(
        -torch.pi * ((k1 - (len / 2)) / q) ** 2)

    exp_term = exp_term[:, None, :]

    cos_term = torch.cos(2 * torch.pi * n[:, None] * k / N0)
    sin_term = torch.sin(2 * torch.pi * n[:, None] * k / N0)

    cos_term = cos_term[None, :, :]
    sin_term = sin_term[None, :, :]

    H = exp_term * (cos_term + 1j * sin_term)
    H = H.reshape(M0 * N0, L)

    # Compute pseudo-inverse and r0
    H_conj = torch.conj(H)
    G = torch.linalg.pinv(torch.matmul(H, torch.conj(H)))
    G = G.to(torch.complex64)
    r0 = torch.matmul(torch.matmul(H_conj, G), V)

    # Create reference window h
    j = torch.arange(L, dtype=torch.float32, device=device)
    h = torch.sqrt(torch.sqrt(torch.tensor(2.0, device=device)) / q) * torch.exp(
        -torch.pi * ((j - len / 2) / q) ** 2)

    rr = torch.matmul(r0, r0.T)  # Signal power distribution
    hh = torch.outer(h, h)

    return rr, hh


class DynamicRangeModule(nn.Module):
    def __init__(self, input_channels, min_value=0.1, max_value=10.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.range_net = nn.Sequential(
            nn.Linear(input_channels, input_channels // 16),
            nn.ReLU(),
            nn.Linear(input_channels // 16, 1),
            nn.Softplus()
        )

    def forward(self, x):
        range_value = self.range_net(x)
        range_value = range_value.mean()
        range_value = self.min_value + (self.max_value - self.min_value) * torch.sigmoid(range_value)

        return range_value


class DynamicRangeModule2(nn.Module):
    def __init__(self, input_channels, min_value=0.1, max_value=10.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.range_net = nn.Sequential(
            nn.Linear(input_channels, input_channels // 16),
            nn.ReLU(),
            nn.Linear(input_channels // 16, 1),
            nn.Softplus()
        )

    def forward(self, x):
        range_value = self.range_net(x)
        range_value = range_value.mean()
        range_value = self.max_value - (self.max_value - self.min_value) * torch.sigmoid(range_value)
        return range_value


if __name__ == '__main__':
    x = torch.randn(1, 1024, 32, 32).cuda()
    model = GlobalFilter(1024, 16, 2, 32, 4).cuda()
    y = model(x)
    print(y.shape)
