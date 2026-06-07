from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    # 彻底抛弃 Numpy，直接在 GPU 上生成随机排列
    return jt.randperm(n)[:m].int32()

def gather_neighbors(x, idx):
    """
    x: (B, N, C)
    idx: (B, N, K)
    return: (B, N, K, C)
    """
    B, N, K = idx.shape
    # 生成 Batch 维度的广播索引，彻底消灭 for 循环
    batch_idx = jt.arange(B).reshape(B, 1, 1).broadcast((B, N, K))
    return x[batch_idx, idx, :]

def chamfer_distance(pcl_a, pcl_b):
    dist_ab, _ = jt.misc.knn(pcl_a, pcl_b, 1)
    dist_ba, _ = jt.misc.knn(pcl_b, pcl_a, 1)
    return dist_ab.mean() + dist_ba.mean()

def point_to_plane_distance(pcl_pred, pcl_clean, normals):
    diff = pcl_pred - pcl_clean
    proj = (diff * normals).sum(dim=-1)
    return (proj ** 2).mean()

def structure_loss(pcl_pred, pcl_clean, k=8):
    if k <= 0:
        return jt.array(0.0)
    if k >= pcl_clean.shape[1]:
        k = int(pcl_clean.shape[1] - 1)
    _, nn_idx = jt.misc.knn(pcl_clean, pcl_clean, k + 1)
    nn_idx = nn_idx[:, :, 1:]
    clean_nn = gather_neighbors(pcl_clean, nn_idx)
    pred_nn = gather_neighbors(pcl_pred, nn_idx)
    rel_clean = clean_nn - pcl_clean.unsqueeze(2)
    rel_pred = pred_nn - pcl_pred.unsqueeze(2)
    return ((rel_pred - rel_clean) ** 2).sum(dim=-1).mean()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.structure_k = cfg.get('structure_k', 8)
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']

        # predict config
        self.predict_num_steps = cfg.get('predict_num_steps', 2)
        self.predict_step_size = cfg.get('predict_step_size', 1.0)
        self.predict_step_schedule = cfg.get('predict_step_schedule', 'linear')
        self.predict_adaptive_step = cfg.get('predict_adaptive_step', True)
        self.predict_iterations = cfg.get('predict_iterations', 1)
        self.predict_postprocess = cfg.get('predict_postprocess', False)
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 6)
        self.predict_seed_k_alpha = cfg.get('predict_seed_k_alpha', 1)
        self.postprocess_k = cfg.get('postprocess_k', 16)
        self.postprocess_strength = cfg.get('postprocess_strength', 0.1)
        self.postprocess_sigma = cfg.get('postprocess_sigma', 0.05)
        self.use_chamfer = cfg.get('use_chamfer', True)
        self.use_p2plane = cfg.get('use_p2plane', True)
        self.use_structure = cfg.get('use_structure', True)
        
        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim'],
            num_layers=cfg.get('edge_conv_layers', 3),
            use_residual=cfg.get('use_residual', True),
            use_attention=cfg.get('use_attention', False),
            attention_heads=cfg.get('attention_heads', 4),
            use_global=cfg.get('use_global', False),
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean, pc_normals=None):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        if pc_normals is not None:
            pc_normals = pc_normals[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore
        
        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        losses = {"loss": loss}
        pred_points = pc_noisy + pred_dir
        if self.use_chamfer:
            losses["chamfer"] = chamfer_distance(pred_points, pc_clean)
        if self.use_p2plane and pc_normals is not None:
            losses["p2plane"] = point_to_plane_distance(pred_points, pc_clean, pc_normals)
        if self.use_structure:
            losses["structure"] = structure_loss(pred_points, pc_clean, k=self.structure_k)
        return losses

    def denoise_langevin_dynamics(
        self,
        pcl_noisy,
        num_steps: int=4,
        step_size: float=1.0,
        step_schedule: str="linear",
        adaptive_step: bool=True,
    ):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                if step_schedule == "cosine":
                    step = step_size * 0.5 * (1 + jt.cos(jt.array(it / max(1, num_steps - 1)) * np.pi))
                elif step_schedule == "linear":
                    step = step_size * (1.0 - it / max(1, num_steps))
                else:
                    step = step_size
                if adaptive_step:
                    scale = jt.sqrt((pred_dir ** 2).sum(dim=-1, keepdims=True))
                    step = step * jt.tanh(scale) / (scale + 1e-6)
                pcl_next = pcl_next + step * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        pc_normals = batch.get('pc_normals', None)
        if pc_normals is not None:
            pc_normals = pc_normals.reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
            pc_normals=pc_normals,
        )
        return loss
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        # 保留 2 视角 TTA (兼顾高分与提速)
        angles = [0.0, np.pi]
        
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            tta_preds = [] 
            
            for angle in angles:
                cosval = np.cos(angle)
                sinval = np.sin(angle)
                rot_matrix = jt.array([
                    [cosval, 0, sinval],
                    [0, 1, 0],
                    [-sinval, 0, cosval]
                ]).float32()
                
                rotated_pc = jt.matmul(pc_noisy, rot_matrix)
                
                # 重新呼叫我们优化过的高速 patch 去噪
                pc_next = rotated_pc
                for it in range(self.predict_iterations):
                    pc_next = patch_based_denoise(
                        model=self,
                        pcl_noisy=pc_next,
                        patch_size=self.predict_patch_size,
                        seed_k=self.predict_seed_k,
                        seed_k_alpha=self.predict_seed_k_alpha,
                        num_steps=self.predict_num_steps,
                        step_size=self.predict_step_size,
                        step_schedule=self.predict_step_schedule,
                        adaptive_step=self.predict_adaptive_step,
                    )
                    if pc_next is None:
                        pc_next = rotated_pc
                        break
                
                inv_rot_matrix = rot_matrix.transpose(0, 1)
                aligned_pc_denoised = jt.matmul(pc_next, inv_rot_matrix)
                
                tta_preds.append(aligned_pc_denoised)
            
            # TTA 融合
            final_pc_denoised = jt.stack(tta_preds, dim=0).mean(dim=0)
            
            pc_denoised_np = final_pc_denoised.detach().numpy()
            res.append({"pc_denoised": pc_denoised_np})
            
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                entry = {
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                }
                if 'pc_normals' in b.meta:
                    entry["pc_normals"] = b.meta['pc_normals']
                res.append(entry)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return: dist, idx, nn
    """
    # ⚡ 提速点 1: 弃用巨型广播，直接调用 Jittor 底层 C++ 的 KNN 算子！
    dist_k, idx = jt.misc.knn(x, y, k)
    
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    num_steps: int=2,
    step_size: float=1.0,
    step_schedule: str="linear",
    adaptive_step: bool=True,
) -> jt.Var:
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    # ⚡ 纯随机 FPS (极大提速)
    seed_idx = jt.randperm(N)[:num_patches]
    seed_pnts = pcl_noisy[:, seed_idx, :]
    
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patches_denoised = []
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(
                curr,
                num_steps=num_steps,
                step_size=step_size,
                step_schedule=step_schedule,
                adaptive_step=adaptive_step,
            )
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    
    # ⚡ 纯 GPU 张量聚合 (不拉回 CPU)
    flat_idx = point_idxs.reshape(-1)
    flat_pts = patches_denoised.reshape(-1, 3)
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    weights = jt.exp(-patch_dists).reshape(-1, 1)
    
    weighted_pts = flat_pts * weights
    
    out_pts = jt.zeros((N, 3)).scatter_(0, flat_idx.unsqueeze(1).broadcast(weighted_pts.shape), weighted_pts, reduce='add')
    out_weights = jt.zeros((N, 1)).scatter_(0, flat_idx.unsqueeze(1), weights, reduce='add')
    
    pcl_out = out_pts / (out_weights + 1e-8)
    return pcl_out

def edge_aware_smooth(pcl, k=16, strength=0.1, sigma=0.05):
    """
    pcl: (N, 3)
    """
    if k <= 0:
        return pcl
    if pcl.shape[0] < k:
        k = pcl.shape[0]
    pcl = pcl.unsqueeze(0)  # (1, N, 3)
    _, _, nn = knn_points(pcl, pcl, k)
    nn = nn.squeeze(0)  # (N, k, 3)
    pcl = pcl.squeeze(0)
    dist = ((nn - pcl.unsqueeze(1)) ** 2).sum(dim=-1)
    weights = jt.exp(-dist / (sigma ** 2 + 1e-8))
    weights = weights / (weights.sum(dim=1, keepdims=True) + 1e-8)
    avg = (weights.unsqueeze(-1) * nn).sum(dim=1)
    return pcl + strength * (avg - pcl)
