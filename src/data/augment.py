from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import (
    random_euler_rotation,
    sample_vertex_groups,
    compute_face_normals,
    compute_vertex_normals,
    compute_face_sharpness,
)

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen

    edge_weight: float=0.0 # edge-aware sampling weight (0 for uniform)
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    # 在 src/data/augment.py 中修改 AugmentSample 的 apply 方法

    def apply(self, asset: Asset, **kwargs):
        assert asset.sampled_vertices is not None, "Please use PreSampledLazyAsset!"
        
        # O(1) 级别的极速随机采样
        total_points = asset.sampled_vertices.shape[0]
        idx = np.random.choice(total_points, self.num_samples, replace=False)
        
        asset.sampled_vertices = asset.sampled_vertices[idx]
        if asset.sampled_normals is not None:
            asset.sampled_normals = asset.sampled_normals[idx]

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        if pc is None:
            pc = asset.sampled_vertices_noisy
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale
        if asset.sampled_vertices_noisy is not None:
            asset.sampled_vertices_noisy = (asset.sampled_vertices_noisy - center) / scale
        if asset.sampled_normals is not None:
            norm = np.linalg.norm(asset.sampled_normals, axis=-1, keepdims=True) + 1e-12
            asset.sampled_normals = asset.sampled_normals / norm

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float

    noise_types: Optional[List[str]]=None

    noise_probs: Optional[List[float]]=None

    uniform_ratio: float=2.0

    impulse_prob: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise_types = self.noise_types or ["laplace"]
        if self.noise_probs is None:
            probs = np.ones(len(noise_types)) / len(noise_types)
        else:
            probs = np.array(self.noise_probs, dtype=np.float32)
            if len(probs) != len(noise_types):
                raise ValueError("noise_probs length must match noise_types length")
            probs = probs / probs.sum()
        noise_type = np.random.choice(noise_types, p=probs)
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_std, size=pc.shape)
        elif noise_type == "uniform":
            noise = np.random.uniform(-self.uniform_ratio * noise_std, self.uniform_ratio * noise_std, size=pc.shape)
        elif noise_type == "impulse":
            noise = np.zeros_like(pc)
            mask = np.random.rand(*pc.shape[:-1], 1) < self.impulse_prob
            impulse = np.random.uniform(-self.uniform_ratio, self.uniform_ratio, size=pc.shape) * noise_std
            noise = np.where(mask, impulse, noise)
        else:
            noise = np.random.laplace(0, noise_std, size=pc.shape)
        asset.sampled_vertices_noisy = pc + noise

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int
    
    train_cvm_network: bool

    scales: Optional[List[float]]=None

    scale_probs: Optional[List[float]]=None

    seed_strategy: str="random"
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        pc_normals = asset.sampled_normals
        
        assert pc is not None
        assert pc_noisy is not None
        
        N = pc_noisy.shape[0]
        
        scales = self.scales or [1.0]
        if self.scale_probs is None:
            scale_probs = np.ones(len(scales)) / len(scales)
        else:
            scale_probs = np.array(self.scale_probs, dtype=np.float32)
            if len(scale_probs) != len(scales):
                raise ValueError("scale_probs length must match scales length")
            scale_probs = scale_probs / scale_probs.sum()
        scale_counts = np.random.multinomial(self.num_patches, scale_probs)
        all_pat_A = []
        all_pat_B = []
        all_pat_t = []
        all_pat_normals = []
        tree = cKDTree(pc_noisy)
        for scale, count in zip(scales, scale_counts):
            if count == 0:
                continue
            if self.seed_strategy == "farthest":
                seed_idx = farthest_point_sampling_numpy(pc_noisy, count)
            else:
                seed_idx = np.random.permutation(N)[:count]   # (P,)
            seed_points = pc_noisy[seed_idx]                         # (P, 3)
            scaled_patch = max(self.patch_size, int(round(self.patch_size * scale)))
            _, nn_idx = tree.query(seed_points, k=scaled_patch)   # (P, M)
            if scaled_patch > self.patch_size:
                down_idx = np.random.choice(scaled_patch, self.patch_size, replace=False)
                nn_idx = nn_idx[:, down_idx]
            pat_A = pc_noisy[nn_idx]  # (P, M, 3)
            pat_B = pc[nn_idx]        # (P, M, 3)
            if pc_normals is not None:
                pat_normals = pc_normals[nn_idx]
            else:
                pat_normals = None
            l1, l2 = 1e-8, 1.0
            t = np.random.rand(count, self.patch_size, 1)
            t = (l2 - l1) * t + l1
            pat_t = t * pat_B + (1 - t) * pat_A
            seed_points_t = (
                t[:, 0:1, :] * pc[seed_idx][:, None, :] +
                (1 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
            )
            pat_A = pat_A - seed_points_t
            pat_B = pat_B - seed_points_t
            pat_t = pat_t - seed_points_t
            all_pat_A.append(pat_A)
            all_pat_B.append(pat_B)
            all_pat_t.append(pat_t)
            if pat_normals is not None:
                all_pat_normals.append(pat_normals)
        if asset.meta is None:
            asset.meta = {}
        if not all_pat_A:
            raise ValueError("No patches generated; check scales and num_patches settings.")
        asset.meta['pc_noisy'] = np.concatenate(all_pat_A, axis=0)
        asset.meta['pc_clean'] = np.concatenate(all_pat_B, axis=0)
        asset.meta['pc_mix'] = np.concatenate(all_pat_t, axis=0)
        if all_pat_normals:
            asset.meta['pc_normals'] = np.concatenate(all_pat_normals, axis=0)

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments

def farthest_point_sampling_numpy(points: np.ndarray, num_samples: int) -> np.ndarray:
    N = points.shape[0]
    if num_samples >= N:
        return np.arange(N)
    farthest = np.random.randint(0, N)
    selected = []
    dist = np.ones((N,)) * 1e10
    for _ in range(num_samples):
        selected.append(farthest)
        centroid = points[farthest]
        d = ((points - centroid) ** 2).sum(axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return np.array(selected, dtype=np.int64)