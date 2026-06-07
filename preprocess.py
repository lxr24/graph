import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm
import multiprocessing as mp

# 导入你原有的数据处理函数
from src.data.utils import sample_surface, compute_face_normals, compute_vertex_normals, compute_face_sharpness

def process_single_obj(obj_path):
    npz_path = obj_path.replace('model_normalized.obj', 'pre_sampled_100k.npz')
    if os.path.exists(npz_path):
        return

    try:
        # 1. 加载网格
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # 2. 计算法向和边缘权重 (只算这一次！)
        face_normals = compute_face_normals(vertices, faces)
        vertex_normals = compute_vertex_normals(vertices, faces, face_normals)
        sharpness = compute_face_sharpness(vertices, faces, face_normals)
        face_weight = 1.0 + 1.0 * sharpness # 假设 edge_weight 为 1.0

        # 3. 采样 100,000 个密集点（作为训练时的“弹药库”）
        sampled_vertices, origin_face_index, _, _ = sample_surface(
            num_samples=100000, 
            vertices=vertices, 
            faces=faces, 
            face_weight=face_weight
        )
        sampled_normals = face_normals[origin_face_index]

        # 4. 保存为高压缩比的二进制格式
        np.savez_compressed(npz_path, vertices=sampled_vertices, normals=sampled_normals)
    except Exception as e:
        print(f"Error processing {obj_path}: {e}")

if __name__ == '__main__':
    # 获取所有的 obj 文件路径
    obj_files = glob.glob('dataset_train/shapenet/*/*/models/model_normalized.obj')
    print(f"Found {len(obj_files)} models to preprocess.")
    
    # 开启多进程飞速处理
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_obj, obj_files), total=len(obj_files)))