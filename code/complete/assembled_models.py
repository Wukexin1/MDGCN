import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device
    if idx is None:
        idx = knn(x, k=k)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def knn_geo(x, pairwise_distance, k):
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature_geo(x, geod_dist, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device
    if idx is None:
        idx = knn_geo(x, geod_dist, k=k)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points

def knn_HGCNN_cov(x, k, local_idx, local=True):
    local_idx = local_idx.detach().to(device=x.device, dtype=torch.bool)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    row_min = pairwise_distance.min(dim=-1, keepdim=True)[0]
    excluded = ~local_idx if local else local_idx
    scores = torch.where(excluded.unsqueeze(1), row_min, pairwise_distance)
    idx = scores.topk(k=k, dim=-1)[1]
    knn_x = index_points(x.permute(0, 2, 1), idx)
    centered = knn_x - knn_x.mean(dim=2, keepdim=True)
    cov = torch.matmul(centered.transpose(2, 3), centered).view(x.size(0), x.size(2), -1).permute(0, 2, 1)
    return (idx, torch.cat([x, cov], dim=1))

def get_graph_feature_HGCNN_cov(x, k, local_idx, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device
    if idx is None:
        idx, x1 = knn_HGCNN_cov(x, k, local_idx, local=local)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    _, num_dims, _ = x1.size()
    x1 = x1.transpose(2, 1).contiguous()
    feature = x1.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x1 = x1.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x1, x1), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def knn_HGCNN(x, k, local_idx, local=True):
    local_idx = local_idx.detach().to(device=x.device, dtype=torch.bool)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    row_min = pairwise_distance.min(dim=-1, keepdim=True)[0]
    excluded = ~local_idx if local else local_idx
    scores = torch.where(excluded.unsqueeze(1), row_min, pairwise_distance)
    idx = scores.topk(k=k, dim=-1)[1]
    return idx

def get_cov_features(x, k):
    """
    无监督地获取局部 k 邻域的协方差矩阵展开特征 (9维)
    x: (B, 3, N)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    knn_x = index_points(x.permute(0, 2, 1), idx)
    mean = torch.mean(knn_x, dim=2, keepdim=True)
    knn_x = knn_x - mean
    covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(x.size(0), x.size(2), -1).permute(0, 2, 1)
    return covariances

def query_geodesic_knn(nsample, geod_dist, new_xyz_idx):
    """
    根据测地线距离矩阵获取每个质心的最近邻索引
    geod_dist: (B, N, N)
    new_xyz_idx: (B, S) - FPS 选出的质心索引
    """
    device = geod_dist.device
    B, S = new_xyz_idx.shape
    new_geod = index_points_pt2(geod_dist, new_xyz_idx)
    group_idx = new_geod.topk(k=nsample, dim=-1, largest=True)[1]
    return group_idx

def get_sub_geod_dist(geod_dist, idx):
    """
    根据 FPS 采样的索引，从完整的测地线矩阵中提取同步降采样的子矩阵
    geod_dist: (B, N, N) - 当前层的距离矩阵
    idx: (B, M) - FPS 采样出的质心索引
    Returns: (B, M, M) - 下一层的局部距离矩阵
    """
    row_subset = index_points_pt2(geod_dist, idx)
    row_subset_T = row_subset.permute(0, 2, 1)
    sub_geod_dist_T = index_points_pt2(row_subset_T, idx)
    sub_geod_dist = sub_geod_dist_T.permute(0, 2, 1)
    return sub_geod_dist

class GraphLayer_dgcnn(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel):
        super(GraphLayer_dgcnn, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x, k, local_idx, local=True, idx=None):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn(x, k=k)
            knn_x = index_points(x.permute(0, 2, 1), idx)
            x1, _ = torch.topk(knn_x, k=math.ceil(k * 2 / 3), dim=2)
            x1 = torch.mean(x1, dim=2).permute(0, 2, 1)
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
        x1 = x1.transpose(2, 1).contiguous()
        x = x.transpose(2, 1).contiguous()
        feature = x1.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl2(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel):
        super(GraphLayer_dgcl2, self).__init__()

    def forward(self, x, k, local_idx, local=True, idx=None):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN(x, k, local_idx, local=local)
        knn_x = index_points(x.permute(0, 2, 1), idx)
        q = knn_x.view(num_points * batch_size, k, num_dims)
        attn_weight = F.softmax(q @ q.transpose(-2, -1) / num_dims ** 0.5, dim=-1)
        attn_output = attn_weight @ q
        attn_feature = attn_output.view(batch_size, num_points, k, num_dims)
        feature = knn_x
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, attn_feature), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl2_norm(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel):
        super(GraphLayer_dgcl2_norm, self).__init__()

    def forward(self, x, k, local_idx, local=True, idx=None):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_norm(x, k, local_idx, local=local)
        knn_x = index_points(x.permute(0, 2, 1), idx)
        q = knn_x.view(num_points * batch_size, k, num_dims)
        attn_weight = F.softmax(q @ q.transpose(-2, -1) / num_dims ** 0.5, dim=-1)
        attn_output = attn_weight @ q
        attn_feature = attn_output.view(batch_size, num_points, k, num_dims)
        feature = knn_x
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, attn_feature), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl2_geo(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel):
        super(GraphLayer_dgcl2_geo, self).__init__()

    def forward(self, x, k, local_idx, geod_dist, local=True, idx=None):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_geo(x, k, local_idx, geod_dist, local=local)
        knn_x = index_points(x.permute(0, 2, 1), idx)
        q = knn_x.view(num_points * batch_size, k, num_dims)
        attn_weight = F.softmax(q @ q.transpose(-2, -1) / num_dims ** 0.5, dim=-1)
        attn_output = attn_weight @ q
        attn_feature = attn_output.view(batch_size, num_points, k, num_dims)
        feature = knn_x
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, attn_feature), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl2_norm_geo(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel):
        super(GraphLayer_dgcl2_norm_geo, self).__init__()

    def forward(self, x, k, local_idx, geod_dist, local=True, idx=None):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_norm_geo(x, k, local_idx, geod_dist, local=local)
        knn_x = index_points(x.permute(0, 2, 1), idx)
        q = knn_x.view(num_points * batch_size, k, num_dims)
        attn_weight = F.softmax(q @ q.transpose(-2, -1) / num_dims ** 0.5, dim=-1)
        attn_output = attn_weight @ q
        attn_feature = attn_output.view(batch_size, num_points, k, num_dims)
        feature = knn_x
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, attn_feature), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl1(nn.Module):
    """
    Graph layer (DGCL1 - Hard Purification).
    """

    def __init__(self, pa):
        super(GraphLayer_dgcl1, self).__init__()
        self.pa = pa

    def forward(self, x, k, local_idx, local=True, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x_view = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN(x_view, k, local_idx, local=local)
        knn_x = index_points(x_view.permute(0, 2, 1), idx)
        magnitudes = torch.norm(knn_x, dim=-1)
        local_thresh = magnitudes.mean(dim=2, keepdim=True)
        mask = (magnitudes >= local_thresh).float()
        valid_count = mask.sum(dim=2).clamp(min=1e-05).unsqueeze(-1)
        mask_expanded = mask.unsqueeze(-1)
        purified_knn_x = knn_x * mask_expanded
        sum_features = purified_knn_x.sum(dim=2)
        x1 = (sum_features / valid_count).permute(0, 2, 1)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_flatten = idx + idx_base
        idx_flatten = idx_flatten.view(-1)
        x1_t = x1.transpose(2, 1).contiguous()
        x_t = x_view.transpose(2, 1).contiguous()
        feature = x1_t.view(batch_size * num_points, -1)[idx_flatten, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x_expand = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x_expand, x_expand), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl1_norm(nn.Module):
    """
    Graph layer (DGCL1 - Hard Purification).
    """

    def __init__(self, pa):
        super(GraphLayer_dgcl1_norm, self).__init__()
        self.pa = pa

    def forward(self, x, k, local_idx, local=True, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x_view = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_norm(x_view, k, local_idx, local=local)
        knn_x = index_points(x_view.permute(0, 2, 1), idx)
        magnitudes = torch.norm(knn_x, dim=-1)
        local_thresh = magnitudes.mean(dim=2, keepdim=True)
        mask = (magnitudes >= local_thresh).float()
        valid_count = mask.sum(dim=2).clamp(min=1e-05).unsqueeze(-1)
        mask_expanded = mask.unsqueeze(-1)
        purified_knn_x = knn_x * mask_expanded
        sum_features = purified_knn_x.sum(dim=2)
        x1 = (sum_features / valid_count).permute(0, 2, 1)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_flatten = idx + idx_base
        idx_flatten = idx_flatten.view(-1)
        x1_t = x1.transpose(2, 1).contiguous()
        x_t = x_view.transpose(2, 1).contiguous()
        feature = x1_t.view(batch_size * num_points, -1)[idx_flatten, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x_expand = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x_expand, x_expand), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl1_geo(nn.Module):
    """
    Graph layer (DGCL1 - Hard Purification).
    """

    def __init__(self, pa):
        super(GraphLayer_dgcl1_geo, self).__init__()
        self.pa = pa

    def forward(self, x, k, local_idx, geo_dist, local=True, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x_view = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_geo(x_view, k, local_idx, geo_dist, local=local)
        knn_x = index_points(x_view.permute(0, 2, 1), idx)
        magnitudes = torch.norm(knn_x, dim=-1)
        local_thresh = magnitudes.mean(dim=2, keepdim=True)
        mask = (magnitudes >= local_thresh).float()
        valid_count = mask.sum(dim=2).clamp(min=1e-05).unsqueeze(-1)
        mask_expanded = mask.unsqueeze(-1)
        purified_knn_x = knn_x * mask_expanded
        sum_features = purified_knn_x.sum(dim=2)
        x1 = (sum_features / valid_count).permute(0, 2, 1)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_flatten = idx + idx_base
        idx_flatten = idx_flatten.view(-1)
        x1_t = x1.transpose(2, 1).contiguous()
        x_t = x_view.transpose(2, 1).contiguous()
        feature = x1_t.view(batch_size * num_points, -1)[idx_flatten, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x_expand = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x_expand, x_expand), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

class GraphLayer_dgcl1_norm_geo(nn.Module):
    """
    Graph layer (DGCL1 - Hard Purification).
    """

    def __init__(self, pa):
        super(GraphLayer_dgcl1_norm_geo, self).__init__()
        self.pa = pa

    def forward(self, x, k, local_idx, geo_dist, local=True, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        num_dims = x.size(1)
        x_view = x.view(batch_size, -1, num_points)
        device = x.device
        if idx is None:
            idx = knn_HGCNN_norm_geo(x_view, k, local_idx, geo_dist, local=local)
        knn_x = index_points(x_view.permute(0, 2, 1), idx)
        magnitudes = torch.norm(knn_x, dim=-1)
        local_thresh = magnitudes.mean(dim=2, keepdim=True)
        mask = (magnitudes >= local_thresh).float()
        valid_count = mask.sum(dim=2).clamp(min=1e-05).unsqueeze(-1)
        mask_expanded = mask.unsqueeze(-1)
        purified_knn_x = knn_x * mask_expanded
        sum_features = purified_knn_x.sum(dim=2)
        x1 = (sum_features / valid_count).permute(0, 2, 1)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_flatten = idx + idx_base
        idx_flatten = idx_flatten.view(-1)
        x1_t = x1.transpose(2, 1).contiguous()
        x_t = x_view.transpose(2, 1).contiguous()
        feature = x1_t.view(batch_size * num_points, -1)[idx_flatten, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x_expand = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x_expand, x_expand), dim=3).permute(0, 3, 1, 2).contiguous()
        return (feature, idx)

def get_graph_feature_HGCNN1(x, k, local_idx, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device
    if idx is None:
        idx = knn_HGCNN(x, k, local_idx, local=local)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x, feature, torch.square(feature - x)), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def get_graph_feature_HGCNN(x, k, local_idx, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device
    if idx is None:
        idx = knn_HGCNN(x, k, local_idx, local=local)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def knn_HGCNN_norm(x, k, local_idx, local=True):
    local_idx = local_idx.detach().to(device=x.device, dtype=torch.bool)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    row_min = pairwise_distance.min(dim=-1, keepdim=True)[0]
    excluded = ~local_idx if local else local_idx
    scores = torch.where(excluded.unsqueeze(1), row_min, pairwise_distance)
    idx = scores.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature_HGCNN_norm(x, k, local_idx, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_HGCNN_norm(x, k, local_idx, local=local)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def knn_HGCNN_geo(x, k, local_idx, pairwise_distance, local=True):
    local_idx = local_idx.detach().to(device=x.device, dtype=torch.bool)
    pass
    row_min = pairwise_distance.min(dim=-1, keepdim=True)[0]
    excluded = ~local_idx if local else local_idx
    scores = torch.where(excluded.unsqueeze(1), row_min, pairwise_distance)
    idx = scores.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature_HGCNN_geo(x, k, local_idx, geod_dist, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_HGCNN_geo(x, k, local_idx, geod_dist, local=local)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

def knn_HGCNN_norm_geo(x, k, local_idx, pairwise_distance, local=True):
    local_idx = local_idx.detach().to(device=x.device, dtype=torch.bool)
    pass
    row_min = pairwise_distance.min(dim=-1, keepdim=True)[0]
    excluded = ~local_idx if local else local_idx
    scores = torch.where(excluded.unsqueeze(1), row_min, pairwise_distance)
    idx = scores.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature_HGCNN_norm_geo(x, k, local_idx, geod_dist, local=True, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_HGCNN_norm_geo(x, k, local_idx, geod_dist, local=local)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature, idx)

class PointNet(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x, res):
        x0 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x0)))
        x = F.relu(self.bn3(self.conv3(x1)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class PointNet_norm(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet_norm, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x, res):
        x0 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x0)))
        if res:
            x1 = x1 - x0
        x = F.relu(self.bn3(self.conv3(x1)))
        if res:
            x = x - x1
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class PointNet_cov(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet_norm, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x1, res):
        _, x = knn_HGCNN_cov(x1, self.kl, idx=None, local=True)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x1)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class DGCNN(nn.Module):

    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, res):
        batch_size = x.size(0)
        x, idx = get_graph_feature(x, k=self.k, idx=None)
        x11 = self.conv1(x)
        x1 = x11.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x1, k=self.k, idx=None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x2, k=self.k, idx=None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x3, k=self.k, idx=None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN_norm(nn.Module):

    def __init__(self, args):
        super(DGCNN_norm, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, res):
        batch_size = x.size(0)
        x, idx = get_graph_feature(x, k=self.k, idx=None)
        x11 = self.conv1(x)
        x1 = x11.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x1, k=self.k, idx=None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x2, k=self.k, idx=None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature(x3, k=self.k, idx=None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN_geo(nn.Module):

    def __init__(self, args):
        super(DGCNN_geo, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, geod_dist, res):
        batch_size = x.size(0)
        x, idx = get_graph_feature_geo(x, geod_dist, k=self.k, idx=None)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x1, geod_dist, k=self.k, idx=None)
        x = self.conv2(x)
        if res:
            x = x - x1
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x2, geod_dist, k=self.k, idx=None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x3, geod_dist, k=self.k, idx=None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN_norm_geo(nn.Module):

    def __init__(self, args):
        super(DGCNN_norm_geo, self).__init__()
        self.args = args
        self.k = args.k
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, geod_dist, res):
        batch_size = x.size(0)
        x, idx = get_graph_feature_geo(x, geod_dist, k=self.k, idx=None)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x1, geod_dist, k=self.k, idx=None)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x2, geod_dist, k=self.k, idx=None)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_geo(x3, geod_dist, k=self.k, idx=None)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Graphcnn(nn.Module):

    def __init__(self, args):
        super(Graphcnn, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.pa = 2 / 3
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn41 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn10 = nn.BatchNorm2d(512)
        if args.self == False:
            self.graph_layer1 = GraphLayer_dgcl1(self.pa)
            self.graph_layer2 = GraphLayer_dgcl1(self.pa)
            self.graph_layer3 = GraphLayer_dgcl1(self.pa)
            self.graph_layer4 = GraphLayer_dgcl1(self.pa)
            self.graph_layer5 = GraphLayer_dgcl1(self.pa)
        else:
            self.graph_layer1 = GraphLayer_dgcl2(0, 0)
            self.graph_layer2 = GraphLayer_dgcl2(0, 0)
            self.graph_layer3 = GraphLayer_dgcl2(0, 0)
            self.graph_layer4 = GraphLayer_dgcl2(0, 0)
            self.graph_layer5 = GraphLayer_dgcl2(0, 0)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2)
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3)
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4)
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn5)
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6)
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7)
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False), self.bn10, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = self.graph_layer1(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer2(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer3(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer4(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x_local, idx = self.graph_layer5(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv10(x_local)
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x, self.k, local_idx, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x + x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Graphcnn_norm(nn.Module):

    def __init__(self, args):
        super(Graphcnn_norm, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.pa = 1 / 2
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.bn10 = nn.BatchNorm2d(512)
        if args.self == False:
            self.graph_layer1 = GraphLayer_dgcl1_norm(self.pa)
            self.graph_layer2 = GraphLayer_dgcl1_norm(self.pa)
            self.graph_layer3 = GraphLayer_dgcl1_norm(self.pa)
            self.graph_layer4 = GraphLayer_dgcl1_norm(self.pa)
            self.graph_layer5 = GraphLayer_dgcl1_norm(self.pa)
        else:
            self.graph_layer1 = GraphLayer_dgcl2_norm(0, 0)
            self.graph_layer2 = GraphLayer_dgcl2_norm(0, 0)
            self.graph_layer3 = GraphLayer_dgcl2_norm(0, 0)
            self.graph_layer4 = GraphLayer_dgcl2_norm(0, 0)
            self.graph_layer5 = GraphLayer_dgcl2_norm(0, 0)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False), self.bn10, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = self.graph_layer1(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer2(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer3(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer4(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x_local, idx = self.graph_layer4(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv10(x_local)
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x, self.k, local_idx, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x1, self.k, local_idx, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x2, self.k, local_idx, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x3, self.k, local_idx, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x + x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class HGCNN_geo(nn.Module):

    def __init__(self, args):
        super(HGCNN_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.par = 3 / 4
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dist):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN_geo(x_local, self.kl, local_idx, geod_dist, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_geo(x_local_1, self.kl, local_idx, geod_dist, local=True, idx=None)
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_geo(x_local_2, self.kl, local_idx, geod_dist, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_geo(x_local_3, self.kl, local_idx, geod_dist, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x, idx = get_graph_feature_HGCNN_geo(x, self.k, local_idx, geod_dist, local=False, idx=None)
        x = self.conv5(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x1, self.k, local_idx, geod_dist, local=False, idx=None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x2, self.k, local_idx, geod_dist, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x3, self.k, local_idx, geod_dist, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class HGCNN_norm_geo(nn.Module):

    def __init__(self, args):
        super(HGCNN_norm_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dist):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x, idx = get_graph_feature_HGCNN(x, self.k, local_idx, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local=False, idx=None)
        x = self.conv6(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class HGCNN(nn.Module):

    def __init__(self, args):
        super(HGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local_11.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x, idx = get_graph_feature_HGCNN(x, self.k, local_idx, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class HGCNN_cov(nn.Module):

    def __init__(self, args):
        super(HGCNN_cov, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        _, x_local = knn_HGCNN_cov(x, self.kl, local_idx, local=True)
        x_local, idx = get_graph_feature_HGCNN(x_local, self.kl, local_idx, local=True, idx=None)
        x_local_11 = self.conv1(x_local)
        x_local_1 = x_local_11.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        _, x_neww = knn_HGCNN_cov(x, self.k, local_idx, local=False)
        x_neww, idx = get_graph_feature_HGCNN(x_neww, self.k, local_idx, local=False, idx=None)
        x_neww_11 = self.conv5(x_neww)
        x1 = x_neww_11.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv6(x_neww)
        if res:
            x_neww = x_neww - x_neww_11
        x2 = x_neww.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv7(x_neww)
        x3 = x_neww.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv8(x_neww)
        x4 = x_neww.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class HGCNN_norm(nn.Module):

    def __init__(self, args):
        super(HGCNN_norm, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = get_graph_feature_HGCNN_norm(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_norm(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_norm(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = get_graph_feature_HGCNN_norm(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x, idx = get_graph_feature_HGCNN_norm(x, self.k, local_idx, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x1, self.k, local_idx, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x2, self.k, local_idx, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm(x3, self.k, local_idx, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Graphcnn_geo(nn.Module):

    def __init__(self, args):
        super(Graphcnn_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.pa = 2 / 3
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn41 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn10 = nn.BatchNorm2d(512)
        if args.self == False:
            self.graph_layer1 = GraphLayer_dgcl1_geo(self.pa)
            self.graph_layer2 = GraphLayer_dgcl1_geo(self.pa)
            self.graph_layer3 = GraphLayer_dgcl1_geo(self.pa)
            self.graph_layer4 = GraphLayer_dgcl1_geo(self.pa)
            self.graph_layer5 = GraphLayer_dgcl1_geo(self.pa)
        else:
            self.graph_layer1 = GraphLayer_dgcl2_geo(0, 0)
            self.graph_layer2 = GraphLayer_dgcl2_geo(0, 0)
            self.graph_layer3 = GraphLayer_dgcl2_geo(0, 0)
            self.graph_layer4 = GraphLayer_dgcl2_geo(0, 0)
            self.graph_layer5 = GraphLayer_dgcl2_geo(0, 0)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2)
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3)
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4)
        self.conv5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn5)
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6)
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7)
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False), self.bn10, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dis, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = self.graph_layer1(x_local, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer2(x_local_1, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer3(x_local_2, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer4(x_local_3, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x_local, idx = self.graph_layer5(x_local, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv10(x_local)
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x1, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x2, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_geo(x3, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x + x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Graphcnn_norm_geo(nn.Module):

    def __init__(self, args):
        super(Graphcnn_norm_geo, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.pa = 1 / 2
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.bn10 = nn.BatchNorm2d(512)
        if args.self == False:
            self.graph_layer1 = GraphLayer_dgcl1_norm_geo(self.pa)
            self.graph_layer2 = GraphLayer_dgcl1_norm_geo(self.pa)
            self.graph_layer3 = GraphLayer_dgcl1_norm_geo(self.pa)
            self.graph_layer4 = GraphLayer_dgcl1_norm_geo(self.pa)
            self.graph_layer5 = GraphLayer_dgcl1_norm_geo(self.pa)
        else:
            self.graph_layer1 = GraphLayer_dgcl2_norm_geo(0, 0)
            self.graph_layer2 = GraphLayer_dgcl2_norm_geo(0, 0)
            self.graph_layer3 = GraphLayer_dgcl2_norm_geo(0, 0)
            self.graph_layer4 = GraphLayer_dgcl2_norm_geo(0, 0)
            self.graph_layer5 = GraphLayer_dgcl2_norm_geo(0, 0)
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False), self.bn10, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, local_idx, geod_dis, res):
        batch_size = x.size(0)
        x_local = x.clone()
        x_local, idx = self.graph_layer1(x_local, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer2(x_local_1, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer3(x_local_2, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer4(x_local_3, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x_local, idx = self.graph_layer5(x_local, self.kl, local_idx, geod_dis, local=True, idx=None)
        x_local = self.conv10(x_local)
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm_geo(x, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv5(x)
        x11 = x
        x1 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm_geo(x1, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv6(x)
        if res:
            x = x - x11
        x2 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm_geo(x2, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv7(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x, idx = get_graph_feature_HGCNN_norm_geo(x3, self.k, local_idx, geod_dis, local=False, idx=None)
        x = self.conv8(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x + x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Graphcnn_cov(nn.Module):

    def __init__(self, args):
        super(Graphcnn_cov, self).__init__()
        self.args = args
        self.k = args.k
        self.kl = args.kl
        self.pa = 2 / 3
        output_channels = args.output_C
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn41 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn10_conv = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2)
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3)
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4)
        self.conv5 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1, bias=False), self.bn5)
        self.conv6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn6)
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn7)
        self.conv8 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False), self.bn10_conv, nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn9, nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn10 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        if args.self == False:
            self.graph_layer1 = GraphLayer_dgcl1(self.pa)
            self.graph_layer2 = GraphLayer_dgcl1(self.pa)
            self.graph_layer3 = GraphLayer_dgcl1(self.pa)
            self.graph_layer4 = GraphLayer_dgcl1(self.pa)
            self.graph_layer5 = GraphLayer_dgcl1(self.pa)
        else:
            self.graph_layer1 = GraphLayer_dgcl2(0, 0)
            self.graph_layer2 = GraphLayer_dgcl2(0, 0)
            self.graph_layer3 = GraphLayer_dgcl2(0, 0)
            self.graph_layer4 = GraphLayer_dgcl2(0, 0)
            self.graph_layer5 = GraphLayer_dgcl2(0, 0)

    def forward(self, x, local_idx, res):
        batch_size = x.size(0)
        _, x_local = knn_HGCNN_cov(x, self.kl, local_idx, local=True)
        x_local, idx = self.graph_layer1(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv1(x_local)
        x_local_11 = x_local
        x_local_1 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer2(x_local_1, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv2(x_local)
        if res:
            x_local = x_local - x_local_11
        x_local_2 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer3(x_local_2, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv3(x_local)
        x_local_3 = x_local.max(dim=-1, keepdim=False)[0]
        x_local, idx = self.graph_layer4(x_local_3, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv4(x_local)
        x_local_4 = x_local.max(dim=-1, keepdim=False)[0]
        x_local = torch.cat((x_local_1, x_local_2, x_local_3, x_local_4), dim=1)
        x_local, idx = self.graph_layer5(x_local, self.kl, local_idx, local=True, idx=None)
        x_local = self.conv10(x_local)
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        _, x_neww = knn_HGCNN_cov(x, self.k, local_idx, local=False)
        x_neww, idx = get_graph_feature_HGCNN(x_neww, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv5(x_neww)
        x_neww_11 = x_neww
        x1 = x_neww.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x1, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv6(x_neww)
        if res:
            x_neww = x_neww - x_neww_11
        x2 = x_neww.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x2, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv7(x_neww)
        x3 = x_neww.max(dim=-1, keepdim=False)[0]
        x_neww, idx = get_graph_feature_HGCNN(x3, self.k, local_idx, local=False, idx=None)
        x_neww = self.conv8(x_neww)
        x4 = x_neww.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.where(local_idx.to(device=x.device, dtype=torch.bool).unsqueeze(1), x + x_local, x)
        x = self.conv9(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn10(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn11(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points_pt2(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 10000000000.0
    farthest = torch.sum((xyz - xyz.mean(dim=1, keepdim=True)) ** 2, dim=-1).argmax(dim=1)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.where(sqrdists > radius ** 2, N, group_idx)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx = torch.where(mask, group_first, group_idx)
    return group_idx

class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = (xyz, points)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_pt2(xyz, new_xyz_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points_pt2(xyz, idx)
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], new_xyz.shape[1], 1, 3)
            if points is not None:
                grouped_points = index_points_pt2(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return (new_xyz.permute(0, 2, 1), new_points)

class PointNet2(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, xyz, res=False):
        B, _, _ = xyz.shape
        if xyz.size(1) > 3:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNet2_norm(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet2_norm, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, data, res=False):
        B, C, N = data.shape
        xyz = data[:, :3, :]
        norm = data[:, 3:, :]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNet2_cov(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet2_cov, self).__init__()
        self.args = args
        self.kl = args.kl
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=12, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, data, res=False):
        B, C, N = data.shape
        xyz = data[:, :3, :]
        cov_feat = get_cov_features(xyz, self.kl)
        l1_xyz, l1_points = self.sa1(xyz, cov_feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNetSetAbstraction_geo(nn.Module):

    def __init__(self, npoint, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction_geo, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, geod_dist, centroids_idx=None):
        """
        xyz: (B, 3, N)
        points: (B, C, N)
        geod_dist: (B, N, N)
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            if centroids_idx is None:
                centroids_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_pt2(xyz, centroids_idx)
            idx = query_geodesic_knn(self.nsample, geod_dist, centroids_idx)
            grouped_xyz = index_points_pt2(xyz, idx)
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], new_xyz.shape[1], 1, 3)
            if points is not None:
                grouped_points = index_points_pt2(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return (new_xyz.permute(0, 2, 1), new_points, centroids_idx)

class PointNet2_geo(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet2_geo, self).__init__()
        self.sa1 = PointNetSetAbstraction_geo(npoint=512, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction_geo(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction_geo(npoint=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, xyz, geod_dist, res=False):
        B, _, _ = xyz.shape
        xyz = xyz[:, :3, :]
        l1_xyz, l1_points, l1_idx = self.sa1(xyz, None, geod_dist)
        geod_dist_l1 = get_sub_geod_dist(geod_dist, l1_idx)
        l2_xyz, l2_points, l2_idx = self.sa2(l1_xyz, l1_points, geod_dist_l1)
        geod_dist_l2 = get_sub_geod_dist(geod_dist_l1, l2_idx)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points, geod_dist_l2)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNet2_norm_geo(nn.Module):

    def __init__(self, args, output_channels=4):
        super(PointNet2_norm_geo, self).__init__()
        self.sa1 = PointNetSetAbstraction_geo(npoint=512, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction_geo(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction_geo(npoint=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, data, geod_dist, res=False):
        B, C, N = data.shape
        xyz = data[:, :3, :]
        norm = data[:, 3:, :]
        l1_xyz, l1_points, l1_idx = self.sa1(xyz, norm, geod_dist)
        geod_dist_l1 = get_sub_geod_dist(geod_dist, l1_idx)
        l2_xyz, l2_points, l2_idx = self.sa2(l1_xyz, l1_points, geod_dist_l1)
        geod_dist_l2 = get_sub_geod_dist(geod_dist_l1, l2_idx)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points, geod_dist_l2)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
