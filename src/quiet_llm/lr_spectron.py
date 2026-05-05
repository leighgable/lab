import torch
import torch.nn as nn
import torch.linalg as linalg

"""
Low-Rattling regularized Spectron-style training/inference. Using
the determinant of $(I_r + B^T A^T A B)$ as a penalty. Since we're
already projecting into rank r, it's r x r. TODO: replace MatFormer
with Adaptive Power Iterated Clustering (APIC) style iteratively
refined low-rank approximation. As surprisal increases, APIC
increases rank. Could be implemented as a form of random-path or
stochastic-depth training.
"""

def factorize_linear(linear_layer, rank):
    """
        Converts a dense nn.linear into Spectron factors A and B.
        Uses Randomized SVD for speed.
    """
    W = linear_layer.weight.data # Shape: (out_features, in_features)
    device = W.device

    # compute randomized SVD
    # returns U (out, r), S (r), V (in, r)
    U, S, V = torch.svd_lowrank(W, q=rank, niter=2)

    # balance singular values
    sqrt_S = torch.diag(torch.sqrt(S))
    A = U @ sqrt_S
    B = V @ sqrt_S
    return A, B, linear_layer.bias.data if linear_layer.bias is not None else None

class SpectronLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int,
                 bias=True,
             ):
        super(SpectronLinear, self).__init__()
        # factors
        self.A = nn.Parameter(torch.randn(out_features, rank)) # up
        self.B = nn.Parameter(torch.randn(in_features, rank))  # down
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # spectron state for power iteration (to track spectral norm)
        self.register_buffer('u_a', torch.randn(rank, 1))
        self.register_buffer('u_b', torch.randn(rank, 1))

    def forward(self, x):
        return (x @ self.B @ self.A.T) + (self.bias if self.bias is not None else 0)

    @torch.no_grad()
    def _spectral_norms(self):
        # estimate spectral norm (sigma)
        def power_iter(W, u, iters=1):
            for _ in range(iters=1):
                v = (W.T @ (W @ u))
                v /= v.norm()
                u = v
            sigma = torch.norm(W @ u)
            return sigma, u

        sigma_a, self.u_a = power_iter(self.A, self.u_a)
        sigma_b, self.u_b = power_iter(self.B, self.u_b)
        return sigma_a, sigma_b


    def low_rattling_loss(self):
        AtA = self.A.T @ self.A
        BtAtAB = self.B.T @ (AtA @ self.B)

        M = torch.eye(AtA.shape[0], device=AtA.device) + BtAtAB
        return torch.linalg.slogdet(M)[1]

# # Assuming 'm' is your pre-trained nano-GPT model
# d = config.n_embd
# rank = d // 8 # Example scaling

# # Replace Up-projection (c_fc: d -> 4d)
# A_up, B_up, bias_up = factorize_linear(m.transformer.h[0].mlp.c_fc, rank)
# new_up = SpectronLinear(d, 4*d, rank)
# new_up.A.data, new_up.B.data = A_up, B_up

# # Replace Down-projection (c_proj: 4d -> d)
# A_down, B_down, bias_down = factorize_linear(m.transformer.h[0].mlp.c_proj, rank)
# new_down = SpectronLinear(4*d, d, rank)
# new_down.A.data, new_down.B.data = A_down, B_down
