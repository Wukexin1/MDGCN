"""Explicit, matched Euclidean coordinate-only revision architecture.

This implements the inspected model1.py operators, not a claimed reconstruction
of the June submission. All operators have the same five-layer local branch,
four-layer background branch and classifier. Normal/geodesic variants and the
original four-layer MVGCN baseline must be evaluated separately.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def neighbors(x, allowed, k, distances=None):
    """Top-k plus a validity mask; forbidden points never count as neighbors.

    Query points cover the entire cloud. Self is eligible when in the view,
    consistent with the inspected network's Euclidean graph builder.
    """
    if allowed.dtype != torch.bool or allowed.shape != (x.shape[0], x.shape[2]):
        raise ValueError('Expected a Boolean B x N candidate/view mask')
    with torch.no_grad():
        if distances is None:
            sq = (x * x).sum(1)
            score = 2 * x.transpose(1, 2) @ x - sq[:, :, None] - sq[:, None, :]
        else:
            if distances.shape != (x.shape[0], x.shape[2], x.shape[2]):
                raise ValueError('Distance shape mismatch')
            if torch.isnan(distances).any() or (distances < 0).any():
                raise ValueError('Distances must be nonnegative, with inf for unreachable pairs')
            score = -distances
        score = score.masked_fill(~allowed[:, None, :], -torch.inf)
        values, index = score.topk(min(k, x.shape[2]), dim=-1)
        valid = torch.isfinite(values)
        index = index.masked_fill(~valid, 0)
    return index, valid


def gather(x, index):
    b = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x.transpose(1, 2)[b, index]


def edge_features(x, index, valid, operator):
    q = gather(x, index)
    center = x.transpose(1, 2)[:, :, None, :].expand_as(q)
    weights = valid.to(x.dtype)
    if operator == 'standard':
        edge = torch.cat((q - center, center), -1)
    elif operator == 'hard':
        magnitude = torch.linalg.vector_norm(q, dim=-1)
        threshold = (magnitude * weights).sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp_min(1)
        selected = valid & (magnitude >= threshold)
        c = (q * selected[..., None]).sum(2) / selected.sum(2)[..., None].clamp_min(1)
        # Match the inspected implementation: use c_j, not a repeated c_i.
        neighbor_c = gather(c.transpose(1, 2), index)
        edge = torch.cat((neighbor_c - center, center), -1)
    elif operator == 'soft':
        logits = q @ q.transpose(-1, -2) / math.sqrt(x.shape[1])
        logits = logits.masked_fill(~valid[..., None, :], -torch.inf)
        has_neighbor = valid.any(-1, keepdim=True)[..., None]
        logits = torch.where(has_neighbor, logits, torch.zeros_like(logits))
        a = torch.softmax(logits, dim=-1) * valid[..., None, :]
        attended = a @ q
        edge = torch.cat((q - center, attended), -1)
    else:
        raise ValueError(operator)
    return (edge * weights[..., None]).permute(0, 3, 1, 2).contiguous()


class MaskedBatchNorm(nn.Module):
    """BatchNorm2d whose statistics exclude invalid padded neighbor slots."""
    def __init__(self, channels, eps=1e-5, momentum=.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, valid):
        w = valid[:, None, :, :].to(x.dtype)
        count = w.sum()
        if self.training and count > 1:
            mean = (x * w).sum((0, 2, 3)) / count
            centered = x - mean[None, :, None, None]
            var = (centered.square() * w).sum((0, 2, 3)) / count
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(var.detach() * count / (count - 1), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return (y * self.weight[None, :, None, None] + self.bias[None, :, None, None]) * w


class EdgeBlock(nn.Module):
    def __init__(self, cin, cout, activation):
        super().__init__()
        self.conv = nn.Conv2d(cin * 2, cout, 1, bias=False)
        self.bn = MaskedBatchNorm(cout)
        self.activation = activation

    def forward(self, x, allowed, k, operator):
        index, valid = neighbors(x, allowed, k)
        edge = edge_features(x, index, valid, operator)
        y = self.bn(self.conv(edge), valid)
        if self.activation:
            y = F.leaky_relu(y, .2)
        y = y.masked_fill(~valid[:, None, :, :], -torch.inf).max(-1).values
        return torch.where(valid.any(-1)[:, None, :], y, torch.zeros_like(y))


class RevisedGraphClassifier(nn.Module):
    def __init__(self, classes=3, operator='hard', fusion='add', k=15,
                 widths=(64, 64, 128, 256), embedding=1024, dropout=.5):
        super().__init__()
        if operator not in ('standard', 'hard', 'soft') or fusion not in ('add', 'replace'):
            raise ValueError('Unknown operator or fusion')
        self.operator, self.fusion, self.k = operator, fusion, k
        ins = [3, *widths[:-1]]
        self.local = nn.ModuleList([EdgeBlock(i, o, j == 0) for j, (i, o) in enumerate(zip(ins, widths))])
        self.background = nn.ModuleList([EdgeBlock(i, o, j == 3) for j, (i, o) in enumerate(zip(ins, widths))])
        joined = sum(widths)
        self.local_fifth = EdgeBlock(joined, joined, True)
        self.embed = nn.Sequential(nn.Conv1d(joined, embedding, 1, bias=False),
                                   nn.BatchNorm1d(embedding), nn.LeakyReLU(.2))
        self.head = nn.Sequential(nn.Linear(embedding * 2, 512, bias=False), nn.BatchNorm1d(512),
                                  nn.LeakyReLU(.2), nn.Dropout(dropout), nn.Linear(512, 256),
                                  nn.BatchNorm1d(256), nn.LeakyReLU(.2), nn.Dropout(dropout),
                                  nn.Linear(256, classes))

    def forward(self, xyz, candidate_mask):
        if xyz.shape[1] != 3:
            raise ValueError('This revision entry uses xyz only')
        outputs = []
        x = xyz
        for block in self.local:
            x = block(x, candidate_mask, self.k, self.operator)
            outputs.append(x)
        local = self.local_fifth(torch.cat(outputs, 1), candidate_mask, self.k, self.operator)
        outputs = []
        x = xyz
        for block in self.background:
            x = block(x, ~candidate_mask, self.k, 'standard')
            outputs.append(x)
        background = torch.cat(outputs, 1)
        mask = candidate_mask[:, None, :]
        fused = background + local * mask if self.fusion == 'add' else torch.where(mask, local, background)
        features = self.embed(fused)
        return self.head(torch.cat((features.max(-1).values, features.mean(-1)), dim=1))
