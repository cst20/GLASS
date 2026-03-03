import collections
import json
import logging
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
import os
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from rqvaedatasets import EmbDataset
from models.rqvae import RQVAE

# ================= ⚙️ 配置区域 =================
# 1. 路径配置
MICRO_DIR = "/home/caoshiteng/TIGER/data/TAOBAO_MM/micro"
FILTER_IDS_PATH = os.path.join(MICRO_DIR, "filtered_item_ids.txt")

# 输入的全量 Parquet 文件
NEW_ITEM_EMB_PARQUET = "../data/TAOBAO_MM/rqvae_data/item_emb.parquet" 

# 2. 列名配置
COL_ITEM_ID = "item_id"     # 物品 ID 列名
COL_EMB = "embedding"       # 向量列名 (如果向量是单列 list 格式)

# 3. 模型配置
CODEBOOK = [64, 128, 128]
DATASET_NAME = "TAOBAO_MM"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自动寻找最新的 ckpt (根据目录结构 Jan-22-2026_13-54-25 等)
BASE_CKPT_DIR = f"./ckpt/taobao_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}"
def get_latest_ckpt(base_dir):
    if not os.path.exists(base_dir): return None
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs: return None
    subdirs.sort(reverse=True) # 假设文件夹命名规则支持按字符串排序得到最新时间
    return os.path.join(base_dir, subdirs[0], "best_collision_model.pth")

ckpt_path = get_latest_ckpt(BASE_CKPT_DIR)
OUTPUT_FILE = f"../data/{DATASET_NAME}/{DATASET_NAME}_t5_rqvae_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy"
# ==============================================

if not ckpt_path:
    raise FileNotFoundError(f"❌ 未找到 Checkpoint 目录: {BASE_CKPT_DIR}")

print(f"✅ Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args_model = ckpt["args"]
state_dict = ckpt["state_dict"]

# ================= 🛠️ 数据加载逻辑 (全量加载 + 标记 Target) =================
data_source = None
is_target_mask = None

# 1. 加载 Target Filter 列表
target_ids_set = set()
if os.path.exists(FILTER_IDS_PATH):
    with open(FILTER_IDS_PATH, 'r') as f:
        target_ids_set = set([line.strip() for line in f])
    print(f"📖 Target Domain 过滤列表已加载，包含 {len(target_ids_set)} 个 ID")
else:
    print(f"⚠️ Warning: Filter 文件不存在，将无法分析 Target 冲突率: {FILTER_IDS_PATH}")

# 2. 尝试从 Parquet 加载全量数据
if os.path.exists(NEW_ITEM_EMB_PARQUET):
    print(f"📖 读取全量 Parquet: {NEW_ITEM_EMB_PARQUET}")
    df = pd.read_parquet(NEW_ITEM_EMB_PARQUET)
    df[COL_ITEM_ID] = df[COL_ITEM_ID].astype(str)
    
    # 标记哪些行属于 Target Domain
    is_target_mask = df[COL_ITEM_ID].isin(target_ids_set).values
    print(f"   - 总样本数: {len(df)}")
    print(f"   - 命中 Target 样本数: {is_target_mask.sum()}")

    # 提取 Embedding 矩阵
    if COL_EMB in df.columns:
        emb_matrix = np.stack(df[COL_EMB].values)
    else:
        feat_cols = [c for c in df.columns if c != COL_ITEM_ID]
        emb_matrix = df[feat_cols].values.astype(np.float32)

    data_source = TensorDataset(torch.from_numpy(emb_matrix))
    data_source.dim = emb_matrix.shape[1]
else:
    print(f"⚠️ 未找到全量 Parquet，回退到 Checkpoint 原始路径: {args_model.data_path}")
    data_source = EmbDataset(args_model.data_path)

# ================= 🚀 初始化模型与推理 =================
model = RQVAE(in_dim=data_source.dim,
              num_emb_list=args_model.num_emb_list,
              e_dim=args_model.e_dim,
              layers=args_model.layers,
              dropout_prob=args_model.dropout_prob,
              bn=args_model.bn,
              loss_type=args_model.loss_type,
              quant_loss_weight=args_model.quant_loss_weight,
              kmeans_init=args_model.kmeans_init,
              kmeans_iters=args_model.kmeans_iters,
              sk_epsilons=args_model.sk_epsilons,
              sk_iters=args_model.sk_iters)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

data_loader = DataLoader(data_source, batch_size=4096, shuffle=False, num_workers=4)

all_indices_raw = []
all_indices_str = []

print(f"\n🚀 开始全量推理 (Device: {DEVICE})...")
with torch.no_grad():
    for d in tqdm(data_loader):
        if isinstance(d, list): d = d[0]
        d = d.to(DEVICE)
        
        # 获取原始语义索引 (不使用 Sinkhorn 强行去重)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        
        for index in indices:
            all_indices_raw.append(index.tolist())
            # 保存字符串形式方便统计冲突
            all_indices_str.append("-".join([str(int(x)) for x in index]))

all_indices_str = np.array(all_indices_str)

# ================= 📊 冲突率分析 (仅针对 Target Domain) =================
print("\n" + "="*40)
print(f"📊 冲突分析报告")
print("="*40)

if is_target_mask is not None:
    target_sids = all_indices_str[is_target_mask]
    num_target = len(target_sids)
    num_unique_target = len(set(target_sids))
    collision_cnt = num_target - num_unique_target
    
    print(f"【Target Domain 结果】")
    print(f"1. 样本总数: {num_target}")
    print(f"2. 唯一 SID 数: {num_unique_target}")
    print(f"3. 冲突数: {collision_cnt}")
    print(f"4. 冲突率 (Collision Rate): {collision_cnt / num_target:.4%}")
    
    # 统计最大冲突深度
    if num_target > 0:
        counts = collections.Counter(target_sids)
        print(f"5. 最大冲突深度: {max(counts.values())}")
else:
    print("⚠️ 由于缺少 ID 匹配信息，跳过 Target Domain 专项分析。")

# 全量数据参考
num_total = len(all_indices_str)
num_unique_total = len(set(all_indices_str))
print(f"\n【全量数据参考】")
print(f"1. 总样本数: {num_total}")
print(f"2. 全量冲突率: {(num_total - num_unique_total) / num_total:.4%}")

# ================= 💾 保存全量结果 =================
# 转换为 numpy 数组
codes_array = np.array(all_indices_raw)

# 按照要求：取消第四位自增去重，直接补 0
padding = np.zeros((codes_array.shape[0], 1), dtype=int)
codes_array_save = np.hstack((codes_array, padding))

print("\n" + "-" * 30)
print(f"💾 Saving FULL codes to: {OUTPUT_FILE}")
print(f"Final shape: {codes_array_save.shape}")
print(f"Sample (Last digit is always 0):\n{codes_array_save[:3]}")
np.save(OUTPUT_FILE, codes_array_save)
print("-" * 30)