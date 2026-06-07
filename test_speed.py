import time
from tqdm import tqdm
import jittor as jt

from src.data.dataset import PCDatasetModule, DatasetConfig
from src.data.transform import Transform

# 1. 模拟数据路径配置
loader_config = DatasetConfig.parse(
    shuffle=True, batch_size=16, num_workers=8,
    datapath={
        'input_dataset_dir': './dataset_train', 
        'use_prob': True, 
        'num_files': 1000, 
        'loader': 'npz', 
        'data_name': 'models/pre_sampled_100k.npz', 
        'ignore_check': True, 
        'data_path': {'shapenet': [['./datalist/train.txt', 1.0]]}
    }
)

# 2. 给一个空的 Transform，防止 dataloader 返回 None
empty_transform = Transform(augments=[])

# 3. 初始化 Dataset，开启 debug=True 跳过复杂的 collate_fn 检查
dataset_module = PCDatasetModule(
    train_dataset_config=loader_config,
    train_transform=empty_transform,
    debug=True 
)

dataloader = dataset_module.train_dataloader()

if dataloader is None:
    print("❌ 错误：Dataloader 依然是 None，请检查配置！")
    exit()

print("🚀 开始测试纯数据加载速度 (仅测试硬盘读取 + 内存拷贝)...")
start_time = time.time()

for i, batch in enumerate(tqdm(dataloader, total=50)):
    if i >= 50:  # 我们只测前 50 个 batch
        break

elapsed = time.time() - start_time
print(f"\n✅ 50 个 Batch 加载总耗时: {elapsed:.2f} 秒")
print(f"⚡ 平均每个 Batch 耗时: {elapsed / 50:.3f} 秒")