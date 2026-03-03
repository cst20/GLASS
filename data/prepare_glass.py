import pandas as pd
import numpy as np
import os
import json
import glob
import gc
from tqdm import tqdm

# =============================================================================
# 全局配置参数（可根据实际环境修改）
# =============================================================================

BASE_DIR = "./TAOBAO_MM"                     # 数据根目录
CODEBOOK = [64, 128, 128]                    # RQVAE Codebook 尺寸
MAX_SEQ_LEN = 50                              # 历史行为硬截断长度
MIN_SEQ_LEN = 40                              # 有效历史最小长度
MAX_RETRIEVAL_LEN = 30                         # 检索历史保留长度

# 输入文件
TRAIN_SHARD_DIR = os.path.join(BASE_DIR, "micro/train")
TEST_SHARD_DIR = os.path.join(BASE_DIR, "micro/test")
ITEM_EMB_PATH = os.path.join(BASE_DIR, "rqvae_data/item_emb.parquet")
RQVAE_CODES_PATH = os.path.join(BASE_DIR, f'TAOBAO_MM_t5_rqvae_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy')

# 中间产物（码本向量与相似度矩阵）
CODEBOOK_VEC_PATH = os.path.join(BASE_DIR, f'codebook_vectors_layer1_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy')
CODEBOOK_SIM_PATH = os.path.join(BASE_DIR, f"codebook_similarity_matrix_{CODEBOOK[0]}.npy")

# 输出目录与索引文件
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, f"t5_data_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}")
USER_INDEX_PATH = os.path.join(BASE_DIR, f"user_history_groups_simsid_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}_codes.json")

# Step 4 的相似度分箱边界
BIN_BOUNDARIES = np.array([-1.0, -0.1, 0.0, 0.1, 1.0])

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


# =============================================================================
# 工具函数：资源生成与加载
# =============================================================================

def ensure_codebook_artifacts():
    """
    若本地缺少码本向量或相似度矩阵，则根据物品嵌入和 RQVAE 码重新计算并保存。
    """
    if os.path.exists(CODEBOOK_VEC_PATH) and os.path.exists(CODEBOOK_SIM_PATH):
        print("✅ [Artifacts] 码本向量与相似度矩阵已存在，跳过生成。")
        return

    print("\n" + "=" * 60)
    print("🛠️  [Artifacts] 开始生成码本向量与相似度矩阵...")
    print("=" * 60)

    # 检查必需文件
    if not os.path.exists(ITEM_EMB_PATH):
        raise FileNotFoundError(f"物品嵌入文件缺失：{ITEM_EMB_PATH}")
    if not os.path.exists(RQVAE_CODES_PATH):
        raise FileNotFoundError(f"RQVAE 码文件缺失：{RQVAE_CODES_PATH}")

    # 加载数据
    item_df = pd.read_parquet(ITEM_EMB_PATH)
    rqvae_codes = np.load(RQVAE_CODES_PATH)

    if len(item_df) != len(rqvae_codes):
        print(f"⚠️  警告：物品数 ({len(item_df)}) 与码数 ({len(rqvae_codes)}) 不一致！")

    # 提取物品嵌入矩阵
    item_embeddings = np.vstack(item_df['embedding'].values).astype(np.float32)
    emb_dim = item_embeddings.shape[1]

    # 聚合第一层码本向量
    print(f"   聚合第1层码本向量（共 {CODEBOOK[0]} 个码）...")
    codebook_vec = np.zeros((CODEBOOK[0], emb_dim), dtype=np.float32)
    codebook_cnt = np.zeros(CODEBOOK[0], dtype=np.int32)

    for i in tqdm(range(len(item_embeddings)), desc="   聚合进度"):
        layer1_code = rqvae_codes[i, 0]
        codebook_vec[layer1_code] += item_embeddings[i]
        codebook_cnt[layer1_code] += 1

    # 计算平均，空码用全局均值填充
    global_mean = np.mean(item_embeddings, axis=0)
    for code in range(CODEBOOK[0]):
        if codebook_cnt[code] > 0:
            codebook_vec[code] /= codebook_cnt[code]
        else:
            codebook_vec[code] = global_mean
            print(f"   - 码 {code} 无对应物品，使用全局均值填充。")

    np.save(CODEBOOK_VEC_PATH, codebook_vec)
    print(f"   ✅ 码本向量已保存：{CODEBOOK_VEC_PATH}")

    # 计算相似度矩阵（余弦相似度）
    print("   计算码本相似度矩阵...")
    norms = np.linalg.norm(codebook_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed_vec = codebook_vec / norms
    sim_matrix = np.dot(normed_vec, normed_vec.T)
    np.save(CODEBOOK_SIM_PATH, sim_matrix)
    print(f"   ✅ 相似度矩阵已保存：{CODEBOOK_SIM_PATH}")

    # 清理临时变量
    del item_df, item_embeddings, rqvae_codes, codebook_vec, sim_matrix
    gc.collect()
    print("✅ [Artifacts] 生成完成！\n")


def load_basic_resources():
    """
    加载物品 ID 到索引的映射表以及语义码（RQVAE codes）。
    返回：(hash_to_index dict, semantic_ids ndarray)
    """
    print("📥 [Resource] 加载基础资源...")
    item_df = pd.read_parquet(ITEM_EMB_PATH, columns=['item_id'])
    hash_to_index = {uid: idx for idx, uid in enumerate(item_df['item_id'])}
    print(f"   物品总数：{len(hash_to_index)}")

    if not os.path.exists(RQVAE_CODES_PATH):
        print(f"⚠️  警告：RQVAE 码文件不存在，将使用随机码（仅用于测试）！")
        semantic_ids = np.random.randint(0, 256, size=(len(item_df), 3)).astype(np.int16)
    else:
        semantic_ids = np.load(RQVAE_CODES_PATH)
        print(f"   语义码形状：{semantic_ids.shape}")

    return hash_to_index, semantic_ids


# =============================================================================
# 核心数据处理步骤
# =============================================================================

def step1_build_user_index(hash_to_index, semantic_ids):
    """
    遍历所有原始数据分片，为每个用户构建按第一层语义码分组的物品历史。
    结果保存为 JSON 文件（用于后续检索增强）。
    """
    print("\n" + "=" * 60)
    print("🚀 Step 1：构建用户历史分组索引（按第一层语义码）")
    print("=" * 60)

    train_files = sorted(glob.glob(os.path.join(TRAIN_SHARD_DIR, "*.parquet")))
    test_files = sorted(glob.glob(os.path.join(TEST_SHARD_DIR, "*.parquet")))
    all_files = train_files + test_files

    if not all_files:
        print("⚠️  未找到任何原始数据分片，Step 1 跳过。")
        return

    user_groups = {}  # {user_id: {layer1_code: [code_triplets]}}

    for file_path in tqdm(all_files, desc="   处理分片"):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"   读取失败 {file_path}：{e}")
            continue

        # 检查必要列
        if '129_1' not in df.columns or '150_2_180' not in df.columns:
            continue

        users = df['129_1'].values
        sequences = df['150_2_180'].values

        for user_raw, seq_raw in zip(users, sequences):
            user_id = str(user_raw)
            # 历史序列为除去最后50个的行为
            if isinstance(seq_raw, np.ndarray):
                seq_raw = seq_raw.tolist()
            if not isinstance(seq_raw, list):
                continue
            history = seq_raw[:-50]
            if not history:
                continue

            for item_hash in history:
                idx = hash_to_index.get(item_hash)
                if idx is None:
                    continue
                codes = semantic_ids[idx].tolist()  # [l1, l2, l3]
                l1 = int(codes[0])
                if l1 not in user_groups.setdefault(user_id, {}):
                    user_groups[user_id][l1] = []
                user_groups[user_id][l1].append(codes)

        del df
        gc.collect()

    print(f"   索引构建完成，共 {len(user_groups)} 个用户。")
    with open(USER_INDEX_PATH, 'w') as f:
        json.dump(user_groups, f)
    print(f"   ✅ 索引已保存：{USER_INDEX_PATH}")

    del user_groups
    gc.collect()


def _process_dataframe_rows(df, hash_to_index, semantic_ids):
    """
    处理单个 DataFrame，提取有效的（history, target）样本。
    返回：(样本列表, 统计字典)
    """
    processed = []
    stats = {'missing_target': 0, 'short_history': 0}

    users = df['129_1'].values
    sequences = df['150_2_180'].values
    targets = df['205'].values

    for user_raw, seq_raw, target_raw in zip(users, sequences, targets):
        # 目标物品必须可映射
        tgt_idx = hash_to_index.get(target_raw)
        if tgt_idx is None:
            stats['missing_target'] += 1
            continue
        target_codes = semantic_ids[tgt_idx].tolist()

        # 历史序列处理
        if isinstance(seq_raw, np.ndarray):
            seq_raw = seq_raw.tolist()
        if not isinstance(seq_raw, list):
            seq_raw = []
        short_seq = seq_raw[-MAX_SEQ_LEN:]  # 硬截断

        # 转换为语义码，只保留能映射的物品
        history_codes = []
        for item_hash in short_seq:
            idx = hash_to_index.get(item_hash)
            if idx is not None:
                history_codes.append(semantic_ids[idx].tolist())

        if len(history_codes) < MIN_SEQ_LEN:
            stats['short_history'] += 1
            continue

        processed.append({
            'user': str(user_raw),
            'history': history_codes,
            'target': target_codes
        })

    return processed, stats


def _process_folder_to_parquet(folder_path, file_pattern, hash_to_index, semantic_ids, out_filename):
    """
    处理一个文件夹下所有匹配的分片，生成一个合并的 Parquet 文件。
    """
    files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    if not files:
        return

    print(f"   📂 生成 {out_filename} ...")
    all_data = []
    total_stats = {'missing_target': 0, 'short_history': 0}

    for f in tqdm(files, desc="   读取分片"):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"   读取失败 {f}：{e}")
            continue

        rows, stats = _process_dataframe_rows(df, hash_to_index, semantic_ids)
        all_data.extend(rows)
        total_stats['missing_target'] += stats['missing_target']
        total_stats['short_history'] += stats['short_history']
        del df
        gc.collect()

    if not all_data:
        return

    final_df = pd.DataFrame(all_data)
    out_path = os.path.join(OUTPUT_DATA_DIR, out_filename)
    final_df.to_parquet(out_path)

    print(f"   💾 保存至：{out_path}")
    print(f"      样本数：{len(final_df)}")
    print(f"      丢弃（目标缺失）：{total_stats['missing_target']}")
    print(f"      丢弃（历史过短）：{total_stats['short_history']}")

    del all_data, final_df
    gc.collect()


def step2_create_basic_datasets(hash_to_index, semantic_ids):
    """
    从原始分片生成 train.parquet 和 test.parquet（包含 history 和 target 的语义码）。
    """
    print("\n" + "=" * 60)
    print(f"🚀 Step 2：生成基础训练/测试数据集（硬截断 {MAX_SEQ_LEN}，最小有效长度 {MIN_SEQ_LEN}）")
    print("=" * 60)

    _process_folder_to_parquet(TRAIN_SHARD_DIR, "train-shard-*.parquet",
                                hash_to_index, semantic_ids, "train.parquet")
    _process_folder_to_parquet(TEST_SHARD_DIR, "test-shard-*.parquet",
                               hash_to_index, semantic_ids, "test.parquet")
    print("✅ Step 2 完成！\n")


def step3_augment_train_data():
    """
    对训练集进行检索增强：为每个样本添加 same_sid_history 列，
    即从该用户历史中召回与目标物品第一层语义码相同的物品序列（截断至 MAX_RETRIEVAL_LEN）。
    结果保存为 train_sim_augmented.parquet。
    """
    print("\n" + "=" * 60)
    print("🚀 Step 3：训练数据检索增强（基于用户索引）")
    print("=" * 60)

    train_file = os.path.join(OUTPUT_DATA_DIR, "train.parquet")
    if not os.path.exists(train_file):
        print(f"⚠️  找不到 {train_file}，跳过 Step 3。")
        return

    # 加载用户索引
    if not os.path.exists(USER_INDEX_PATH):
        print(f"⚠️  用户索引文件 {USER_INDEX_PATH} 不存在，跳过。")
        return
    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)

    df = pd.read_parquet(train_file)
    retrieved_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="   检索历史"):
        user_id = str(row['user'])
        target = row['target']          # [l1, l2, l3]
        l1 = str(target[0]) if  target is not None and len(target) > 0 else None

        if l1 and user_id in user_index:
            full = user_index[user_id].get(l1, [])
            # 只保留最近 MAX_RETRIEVAL_LEN 条
            retrieved = full[-MAX_RETRIEVAL_LEN:]
        else:
            retrieved = []
        retrieved_list.append(retrieved)

    df['same_sid_history'] = retrieved_list
    out_path = os.path.join(OUTPUT_DATA_DIR, "train_sim_augmented.parquet")
    df.to_parquet(out_path)
    print(f"   💾 增强训练数据已保存：{out_path}")

    del df, user_index, retrieved_list
    gc.collect()
    print("✅ Step 3 完成！\n")


def step3_split_test_with_histories():
    """
    处理测试集：为 test.parquet 中的每个样本添加 sid_{}_history 列（对应每个第一层码的检索历史），
    然后随机打乱并等分为 valid_with_history.parquet 和 test_with_history.parquet。
    """
    print("\n" + "=" * 60)
    print("🚀 Step 3：测试集分割与检索历史添加")
    print("=" * 60)

    test_file = os.path.join(OUTPUT_DATA_DIR, "test.parquet")
    if not os.path.exists(test_file):
        print(f"⚠️  找不到 {test_file}，跳过。")
        return

    # 加载用户索引
    if not os.path.exists(USER_INDEX_PATH):
        print(f"⚠️  用户索引文件 {USER_INDEX_PATH} 不存在，跳过。")
        return
    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)

    df = pd.read_parquet(test_file)
    users = df['user'].astype(str).tolist()

    # 为每个第一层码生成历史列
    new_cols = {}
    for sid in tqdm(range(CODEBOOK[0]), desc="   生成各码历史列"):
        sid_str = str(sid)
        col_data = []
        for uid in users:
            hist = user_index.get(uid, {}).get(sid_str, [])
            col_data.append(hist[-MAX_RETRIEVAL_LEN:])
        new_cols[f'sid_{sid}_history'] = col_data

    df_aug = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # 随机打乱并均分
    df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)
    split = len(df_aug) // 2
    valid_df = df_aug.iloc[:split]
    test_df = df_aug.iloc[split:]

    valid_path = os.path.join(OUTPUT_DATA_DIR, "valid_with_history.parquet")
    test_path = os.path.join(OUTPUT_DATA_DIR, "test_with_history.parquet")
    valid_df.to_parquet(valid_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"   ✅ 验证集：{len(valid_df)} 条，测试集：{len(test_df)} 条")
    print(f"      保存至：{valid_path} 和 {test_path}")

    del df, df_aug, user_index, new_cols
    gc.collect()
    print("✅ Step 3 完成！\n")


def step4_add_sidtier_and_cleanup():
    """
    为最终保留的三个文件（valid_with_history, test_with_history, train_sim_augmented）
    添加 sidtier 特征（基于用户长期历史物品与码本的相似度分布），并删除中间文件 train.parquet 和 test.parquet。
    """
    print("\n" + "=" * 60)
    print("🚀 Step 4：添加 SIDTier 特征并清理中间文件")
    print("=" * 60)

    # 1. 加载物品嵌入和码本向量
    print("   加载物品嵌入和码本向量...")
    item_df = pd.read_parquet(ITEM_EMB_PATH)
    item_embeddings = np.vstack(item_df['embedding'].values).astype(np.float32)
    hash_to_idx = {uid: i for i, uid in enumerate(item_df['item_id'])}
    codebook_vec = np.load(CODEBOOK_VEC_PATH).astype(np.float32)
    print(f"   物品嵌入形状：{item_embeddings.shape}，码本形状：{codebook_vec.shape}")

    # 2. 构建用户长期历史索引（物品索引列表）
    print("   构建用户长期历史（物品索引）...")
    user_long_hist = {}  # user_id -> list of item indices
    all_shards = sorted(glob.glob(os.path.join(TRAIN_SHARD_DIR, "*.parquet"))) + \
                 sorted(glob.glob(os.path.join(TEST_SHARD_DIR, "*.parquet")))

    for fpath in tqdm(all_shards, desc="   读取原始分片"):
        try:
            df = pd.read_parquet(fpath)
        except:
            continue
        if '129_1' not in df.columns or '150_2_180' not in df.columns:
            continue

        users = df['129_1'].values
        seqs = df['150_2_180'].values

        for u, seq in zip(users, seqs):
            u = str(u)
            if isinstance(seq, np.ndarray):
                seq = seq.tolist()
            if not isinstance(seq, list):
                continue
            long_part = seq[:-50]  # 除去最后50个作为历史
            if not long_part:
                continue
            indices = [hash_to_idx[it] for it in long_part if it in hash_to_idx]
            if indices:
                # 若用户已有历史，则追加（不同分片可能包含同一用户的不同会话）
                user_long_hist.setdefault(u, []).extend(indices)

        del df
        gc.collect()

    print(f"   用户长期历史构建完成，共 {len(user_long_hist)} 个用户。")

    # 3. 定义 sidtier 计算函数（批量版）
    def compute_sidtier_batch(history_indices_list):
        """
        输入：每个元素为某个用户的历史物品索引列表（可空）
        输出：每个元素为一个长度为 CODEBOOK[0]*4 的特征向量
        """
        features_list = []
        for hist_idx in history_indices_list:
            if len(hist_idx) == 0:
                features_list.append(np.zeros(CODEBOOK[0] * 4, dtype=np.float32))
                continue

            hist_embs = item_embeddings[hist_idx]                     # (L, D)
            sims = np.dot(hist_embs, codebook_vec.T)                  # (L, C)
            sidtier = np.zeros(CODEBOOK[0] * 4, dtype=np.float32)

            for c in range(CODEBOOK[0]):
                col_sims = sims[:, c]
                counts, _ = np.histogram(col_sims, bins=BIN_BOUNDARIES)
                total = len(col_sims)
                if total > 0:
                    norm_counts = counts / total
                else:
                    norm_counts = np.zeros(4)
                sidtier[c*4:(c+1)*4] = norm_counts

            features_list.append(sidtier)
        return features_list

    # 4. 处理需要添加 sidtier 的文件
    target_files = [
        ("valid_with_history.parquet", "验证集"),
        ("test_with_history.parquet", "测试集"),
        ("train_sim_augmented.parquet", "增强训练集")
    ]

    batch_size = 500
    for fname, desc in target_files:
        fpath = os.path.join(OUTPUT_DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   ⚠️ 文件 {fname} 不存在，跳过。")
            continue

        print(f"\n   处理 {desc}：{fname}")
        df = pd.read_parquet(fpath)
        users = df['user'].astype(str).tolist()

        # 获取每个用户的长期历史索引列表（可能为空）
        hist_indices = [user_long_hist.get(uid, []) for uid in users]

        # 分批计算 sidtier
        all_sidtier = []
        for i in tqdm(range(0, len(users), batch_size), desc=f"   计算 sidtier"):
            batch = hist_indices[i:i+batch_size]
            batch_feat = compute_sidtier_batch(batch)
            all_sidtier.extend(batch_feat)

        df['sidtier'] = all_sidtier

        # 若存在 'sid' 列则拼接（本流程无此列，保留兼容）
        if 'sid' in df.columns:
            df['sidtier'] = df.apply(lambda r: np.concatenate([r['sid'], r['sidtier']]), axis=1)

        # 覆盖保存
        df.to_parquet(fpath, index=False)
        print(f"     已更新 sidtier 特征并保存。")
        del df
        gc.collect()

    # 5. 删除中间文件
    print("\n   清理中间文件...")
    for f in ["train.parquet", "test.parquet"]:
        p = os.path.join(OUTPUT_DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"     删除 {f}")

    # 释放大对象
    del item_df, item_embeddings, codebook_vec, user_long_hist
    gc.collect()
    print("✅ Step 4 完成！")


# =============================================================================
# 主执行流程
# =============================================================================

if __name__ == "__main__":
    print(f"🚀 [启动] 全流程数据处理开始")
    print(f"   配置：Codebook={CODEBOOK}, MAX_SEQ={MAX_SEQ_LEN}, MIN_SEQ={MIN_SEQ_LEN}")

    # 1. 生成必要的码本工件（若缺失）
    ensure_codebook_artifacts()

    # 2. 加载基础资源
    hash2idx, semantic_ids = load_basic_resources()

    # 3. 执行各步骤
    step1_build_user_index(hash2idx, semantic_ids)

    step2_create_basic_datasets(hash2idx, semantic_ids)

    # 释放不再需要的大对象（索引构建后仍需 hash2idx 和 semantic_ids，但 step2 已使用完毕）
    # 注意：step3 函数内部会自行加载索引文件，无需保留 hash2idx/semantic_ids
    del hash2idx, semantic_ids
    gc.collect()

    step3_augment_train_data()          # 新增：对训练集进行检索增强
    step3_split_test_with_histories()   # 处理测试集并添加检索历史

    step4_add_sidtier_and_cleanup()      # 添加 sidtier 并清理中间文件

    print("\n🎉 [完成] 所有步骤执行完毕！")