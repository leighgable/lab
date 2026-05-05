import torch

def apply_rattle_regularization(A, B, lambda_rattle, k):
    A_k = A[:, :k]
    B_k = B[:, :k]

    R = B_k.T @ A_k
    I = torch.eye(k, device=A.device)

    # gradient of ||R - I||_F^2
    G = 2 * (R - I)

    grad_A = B_k @ G
    grad_B = A_k @ G.T

    return grad_A, grad_B

class RattleController:
    def __init__(
        self,
        r_target=0.1,
        H_target=None,        # set after a warmup estimate
        alpha=0.5, beta=0.25, gamma=0.1, eta=0.05,
        clip_e=0.5,
        I_max=10.0,
        log_lambda_init=-6.9,   # ~1e-3
        log_lambda_min=-12.0,   # ~6e-6
        log_lambda_max=-2.0     # ~0.135
    ):
        self.r_target = r_target
        self.H_target = H_target
        self.alpha, self.beta, self.gamma, self.eta = alpha, beta, gamma, eta
        self.clip_e = clip_e
        self.I = 0.0
        self.I_max = I_max
        self.log_lambda = log_lambda_init
        self.log_lambda_min = log_lambda_min
        self.log_lambda_max = log_lambda_max

    @torch.no_grad()
    def step(self, r, s=None, H=None):
        # primary error
        e_r = (r - self.r_target)
        e_r = torch.clamp(e_r, -self.clip_e, self.clip_e)

        # spectral term
        e_s = 0.0
        if s is not None:
            e_s = torch.clamp(s, -self.clip_e, self.clip_e)

        # uncertainty term
        e_u = 0.0
        if (H is not None) and (self.H_target is not None):
            e_u = (H - self.H_target) / (self.H_target + 1e-6)
            e_u = torch.clamp(e_u, -self.clip_e, self.clip_e)

        # integral on primary signal
        self.I = float(torch.clamp(torch.tensor(self.I + e_r), -self.I_max, self.I_max))

        # update log-lambda
        self.log_lambda += (
            self.alpha * float(e_r)
            + self.beta * float(e_s)
            + self.gamma * float(e_u)
            + self.eta * self.I
        )

        # clamp
        self.log_lambda = float(torch.clamp(
            torch.tensor(self.log_lambda),
            self.log_lambda_min,
            self.log_lambda_max
        ))

        return float(torch.exp(torch.tensor(self.log_lambda)))

def normalized_entropy(logits, eps=1e-8):
    probs = torch.softmax(logits, dim=-1)
    H = -(probs * (probs + eps).log()).sum(dim=-1)

    # normalize by log vocab size
    H_norm = H / torch.log(torch.tensor(probs.shape[-1], device=logits.device))
    return H_norm.mean()

class EMA:
    """
        Smoothing, raw entropy is noisy
    """
    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value
    
def difficulty_to_rank(d, k_min, k_max, sharpness=6.0, center=0.5):
    # sigmoid mapping
    x = torch.sigmoid(sharpness * (d - center))
    k = k_min + (k_max - k_min) * x
    return int(k.item())

def difficulty_to_lambda(d, lambda_min, lambda_max, sharpness=4.0):
    x = torch.sigmoid(sharpness * (d - 0.5))
    return lambda_min + (lambda_max - lambda_min) * x

class RankLambdaAllocator:
    def __init__(
        self,
        k_min,
        k_max,
        lambda_min=1e-5,
        lambda_max=1e-2,
        ema_decay=0.95
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.ema = EMA(ema_decay)

    @torch.no_grad()
    def step(self, logits, controller_lambda):
        # 1. difficulty
        d_raw = normalized_entropy(logits)
        d = self.ema.update(d_raw)

        # 2. rank
        k = difficulty_to_rank(d, self.k_min, self.k_max)

        # 3. lambda from difficulty
        lambda_diff = difficulty_to_lambda(d, self.lambda_min, self.lambda_max)

        # 4. combine with controller
        lambda_final = controller_lambda * lambda_diff

        return k, lambda_final, d

def attention_entropy(attn_probs, eps=1e-8):
    # attn_probs: [B, H, T, T]
    H = -(attn_probs * (attn_probs + eps).log()).sum(dim=-1)
    return H.mean()   # scalar

def normalized_attention_entropy(attn_probs):
    T = attn_probs.shape[-1]
    H = attention_entropy(attn_probs)
    return H / torch.log(torch.tensor(T, device=attn_probs.device))

# smooth per layer
# layer_ema = [EMA(0.9) for _ in range(num_layers)]

# d_l = layer_ema[l].update(d_l_raw)

class LayerNormTracker:
    def __init__(self, decay=0.99):
        self.ema = EMA(decay)

    def normalize(self, x):
        mean = self.ema.update(x)
        return x / (mean + 1e-6)
    
# Head aware refinement:
# H_heads = -(attn_probs * (attn_probs + 1e-8).log()).sum(dim=-1)  # [B,H,T]
# H_heads = H_heads.mean(dim=(0,2))  # per head

# # focus on top-k entropy heads
# d_l = H_heads.topk(k_heads).values.mean()
#
def layer_rank(d_l, k_min, k_max, sharpness=5.0):
    x = torch.sigmoid(sharpness * (d_l - 1.0))  # centered at 1 (relative)
    return int(k_min + (k_max - k_min) * x)

def layer_lambda(d_l, lambda_min, lambda_max):
    x = torch.sigmoid(4.0 * (d_l - 1.0))
    return lambda_min + (lambda_max - lambda_min) * x

class LayerwiseAllocator:
    def __init__(self, num_layers, k_min, k_max):
        self.num_layers = num_layers
        self.k_min = k_min
        self.k_max = k_max

        self.ema = [EMA(0.9) for _ in range(num_layers)]
        self.norm = [LayerNormTracker() for _ in range(num_layers)]

    @torch.no_grad()
    def step(self, attn_probs_list, controller_lambdas):
        ks = []
        lambdas = []

        for l, attn_probs in enumerate(attn_probs_list):
            d_raw = normalized_attention_entropy(attn_probs)
            d_smooth = self.ema[l].update(d_raw)
            d_rel = self.norm[l].normalize(d_smooth)

            k = layer_rank(d_rel, self.k_min, self.k_max)
            lam = controller_lambdas[l] * layer_lambda(d_rel, 1e-5, 1e-2)

            ks.append(k)
            lambdas.append(lam)

        return ks, lambdas

#1

def value_weighted_attention_sensitivity(attn_probs, V, eps=1e-8):
    # attn_probs: [B, H, T, T]
    # V:          [B, H, T, D]

    V_norm = V.norm(dim=-1)  # [B, H, T]

    # broadcast to match attention
    V_norm = V_norm.unsqueeze(-2)  # [B, H, 1, T]

    S = -(attn_probs * (attn_probs + eps).log()) * V_norm
    return S.sum(dim=-1).mean()   # scalar
#2
def attention_variance_sensitivity(attn_probs, V):
    # variance over keys
    var = attn_probs.var(dim=-1)  # [B, H, T]

    V_norm = V.norm(dim=-1)       # [B, H, T]

    S = var * V_norm
    return S.mean()
#3
def head_disagreement(attn_probs):
    # attn_probs: [B, H, T, T]
    mean = attn_probs.mean(dim=1, keepdim=True)
    diff = (attn_probs - mean).pow(2)
    return diff.mean()
#4
def finite_diff_sensitivity(attn_probs, V, epsilon=1e-2):
    noise = torch.randn_like(attn_probs)
    noise = noise / (noise.norm() + 1e-6)

    P_perturbed = attn_probs + epsilon * noise
    P_perturbed = torch.softmax(P_perturbed, dim=-1)

    out = attn_probs @ V
    out_perturbed = P_perturbed @ V

    return (out - out_perturbed).norm() / epsilon

#5
# Weighted mix
def combined_sensitivity(attn_probs, V):
    s1 = value_weighted_attention_sensitivity(attn_probs, V)
    s2 = attention_variance_sensitivity(attn_probs, V)
    s3 = head_disagreement(attn_probs)

    return 0.5 * s1 + 0.3 * s2 + 0.2 * s3

# Training loop example
# active rank k
# # forward pass
# logits = model(x)

# # allocator
# k, lambda_rattle, d = allocator.step(logits, controller_lambda)

# # apply rank mask + rattle gradients
# grad_A_r, grad_B_r = apply_rattle_regularization(A, B, lambda_rattle, k)

# A.grad[:, :k] += grad_A_r
# B.grad[:, :k] += grad_B_r
#
# Inference time
# with torch.no_grad():
    # logits = model(x)

    # k, _, d = allocator.step(logits, controller_lambda=1.0)

    # use k to select rank
