# 这个脚本实现了一个标准的 推荐系统离线数据处理 Pipeline，特别针对“微型化（Micro）”数据集的构建场景
# 它主要包含四个核心步骤：
# Step 1: 样本对齐与筛选 (Process Samples & Intersect)
# 功能：处理 train_samples 和 test_samples。
# 标签过滤：通过 check_is_click 函数筛选数据（当前代码保留点击的数据）。
# 用户采样：如果设置了 NUM_USERS，只保留前 N 个用户的样本。
# 物品交集 (Intersection)：这是一个非常关键的步骤。它计算了训练集和测试集中物品 ID 的交集。
# 输出：保存筛选后的 train/test_samples.parquet 和 filtered_item_ids.txt。

# Step 2: 用户特征对齐 (Process User Features)
# 功能：处理 user_features。
# 根据 Step 1 中确定的“有效用户列表”，从海量用户特征表中只提取这些用户的数据。
# 提取历史序列：解析 150_2_180 列（假设为历史行为序列），将所有序列中出现过的 Item ID 收集起来。
# 输出：保存 train/test_user_features.parquet。

# Step 3: 物品池构建与特征过滤 (Process Item Features)
# 功能：处理 item_features。
# 全量物品池 = Target Items (样本中的) + History Items (序列中的)。
# 原因：模型不仅需要对 Target Item 进行打分，还需要对用户历史序列中的 Item 进行 Embedding 查表。如果历史序列里的物品不在 Embedding 表里，模型会报错。
# 根据这个“全量物品池”，过滤原始的物品特征表。
# 输出：保存 item_features.parquet 和 final_item_ids.txt。

# Step 4: 宽表构建 (Build Wide Table)
# 功能：将样本、用户特征、物品特征拼成一张大宽表。
# DuckDB：使用 DuckDB 引擎执行 SQL JOIN。相比 Pandas 的 merge，DuckDB 在处理大文件 Join 时内存管理更优秀，且速度极快。
# 分片保存 (Sharding)：将结果按 SHARD_SIZE (50000行) 切分为多个小 Parquet 文件。这非常有利于 PyTorch/TensorFlow 的 DataLoader 并行读取数据。

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import os
import numpy as np
import gc

SOURCE_DIR = "./TAOBAO_MM/raw"
MICRO_DIR = "./TAOBAO_MM/micro"

# 用户筛选控制
#   - 整数 (e.g., 100000): 只保留前 N 个用户
#   - None             : 不筛选，保留全量用户
NUM_USERS = None  

# 物品筛选控制
#   - 整数: 限制交集物品的最大数量
#   - None: 不限制
NUM_ITEMS_SAMPLE = 10000
# 物品最小交互频次过滤 (只针对 Train)
MIN_ITEM_FREQ = 5

SHARD_SIZE = 50000 

# 【核心修改】这里改回读取原始的 samples 文件
FILES = {
    "train_sample": "train_samples.parquet",  # 原名: train_samples.parquet
    "test_sample":  "test_samples.parquet",   # 原名: test_samples.parquet
    "train_feat":   "train_user_features.parquet",
    "test_feat":    "test_user_features.parquet",
    "item_feat":    ["item_features.parquet", "scl_embedding_int8_p90.parquet"]
}

def check_is_click(val):
    """判断是否为点击 (label_0 == 0)"""
    if isinstance(val, (np.ndarray, list)):
        return val[0] == 0
    # 针对原始数据中可能存在的非list类型做兼容
    if isinstance(val, (int, float, np.number)):
        return val == 0
    return False

def save_parquet(df, path):
    if len(df) > 0:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        print(f"    💾 保存成功: {path} (Rows: {len(df)})")
    else:
        print(f"    ⚠️ 警告: 数据为空，未保存 {path}")


def step1_process_samples_and_intersect():
    print("\n🔵 [Step 1] 读取原始样本 -> 筛选用户/点击 -> 计算物品交集...")
    os.makedirs(MICRO_DIR, exist_ok=True)
    
    data_cache = {} 
    valid_users = {"train": set(), "test": set()}
    
    # --- 1.1 读取 & 预处理 ---
    for mode, input_file in [("train", FILES["train_sample"]), ("test", FILES["test_sample"])]:
        src_path = os.path.join(SOURCE_DIR, input_file)
        if not os.path.exists(src_path):
            print(f"    ❌ 错误: 找不到原始文件 {src_path}")
            return None, None

        print(f"    📖 读取原始文件: {input_file} ...")
        df = pd.read_parquet(src_path)
        
        # A. 过滤点击
        if "label_0" in df.columns:
            df = df[df["label_0"].apply(check_is_click)]
        
        # B. 过滤用户 (Top N)
        if "129_1" in df.columns:
            unique_users = df["129_1"].unique()
            if NUM_USERS is not None and isinstance(NUM_USERS, int) and NUM_USERS > 0:
                selected_users = unique_users[:NUM_USERS] if len(unique_users) > NUM_USERS else unique_users
            else:
                selected_users = unique_users
            
            df = df[df["129_1"].isin(selected_users)]
            valid_users[mode] = set(selected_users) 
        
        data_cache[mode] = df
        print(f"    ✅ {mode} 预处理暂存: {len(df)} 行")
    # ========================================================
    # 🛑 【新增逻辑】在此处过滤 Train 中交互次数 < 5 的物品
    # ========================================================
    if MIN_ITEM_FREQ is not None and MIN_ITEM_FREQ > 1:
        print(f"    📉 执行低频物品过滤 (Min Freq >= {MIN_ITEM_FREQ})...")
        train_df = data_cache["train"]
        
        # 1. 统计 Train 中每个 Item 的出现次数
        item_counts = train_df["205"].value_counts()
        
        # 2. 找到满足阈值的 Item ID
        high_freq_items = item_counts[item_counts >= MIN_ITEM_FREQ].index
        
        # 3. 打印一下过滤了多少
        print(f"       原始物品数: {len(item_counts)} -> 高频物品数: {len(high_freq_items)}")
        
        # 4. 过滤 Train DataFrame
        # 只有在 high_freq_items 里的行才保留
        data_cache["train"] = train_df[train_df["205"].isin(high_freq_items)]
        
        print(f"       过滤后 Train 样本数: {len(data_cache['train'])}")
    # --- 1.2 计算物品交集 ---
    print("    🔗 计算 Train/Test 物品交集...")
    train_items = set(data_cache["train"]["205"].unique())
    test_items = set(data_cache["test"]["205"].unique())
    
    common_items = train_items.intersection(test_items)
    print(f"    📊 交集物品数量: {len(common_items)}")
    
    if len(common_items) == 0:
        print("    ❌ 严重错误: Train 和 Test 没有公共物品！")
        return None, None

    # (可选) 采样
    if NUM_ITEMS_SAMPLE is not None and len(common_items) > NUM_ITEMS_SAMPLE:
        print(f"    ✂️ [配置生效] 随机采样保留 {NUM_ITEMS_SAMPLE} 个物品...")
        np.random.seed(42)
        common_items = set(np.random.choice(list(common_items), size=NUM_ITEMS_SAMPLE, replace=False))

    # ==========================================
    # ✅ [新增逻辑] 保存 filtered_item_ids.txt
    # ==========================================
    filtered_txt_path = os.path.join(MICRO_DIR, "filtered_item_ids.txt")
    print(f"    💾 保存 Target 交集物品列表到: {filtered_txt_path}")
    try:
        sorted_items = sorted(list(common_items))
        with open(filtered_txt_path, 'w', encoding='utf-8') as f:
            for item_id in sorted_items:
                f.write(f"{item_id}\n")
    except Exception as e:
        print(f"    ⚠️ 保存 filtered_item_ids.txt 失败: {e}")

    # --- 1.3 过滤 DataFrame 并保存 ---
    target_item_ids = common_items
    
    for mode in ["train", "test"]:
        df = data_cache[mode]
        df_final = df[df["205"].isin(common_items)]
        
        save_parquet(df_final, os.path.join(MICRO_DIR, f"{mode}_samples.parquet"))
        
        valid_users[mode] = set(df_final["129_1"].unique())
        del df, df_final
        gc.collect()

    return valid_users, target_item_ids

def step2_process_user_features(valid_users):
    print("\n🔵 [Step 2] 处理用户特征 & 提取历史序列物品...")
    history_item_ids = set()
    
    for mode, input_file in [("train", FILES["train_feat"]), ("test", FILES["test_feat"])]:
        src_path = os.path.join(SOURCE_DIR, input_file)
        out_path = os.path.join(MICRO_DIR, f"{mode}_user_features.parquet")
        
        if not os.path.exists(src_path): 
            print(f"    ⚠️ 跳过 {input_file} (文件不存在)")
            continue
            
        print(f"    📖 读取特征: {input_file} ...")
        df = pd.read_parquet(src_path)
        
        if "129_1" in df.columns and mode in valid_users:
            original_len = len(df)
            df = df[df["129_1"].isin(valid_users[mode])]
            print(f"       用户对齐: {original_len} -> {len(df)}")
            
            if "150_2_180" in df.columns:
                seq_items = df["150_2_180"].explode().unique()
                seq_items = [x for x in seq_items if pd.notna(x)]
                history_item_ids.update(seq_items)
            
            save_parquet(df, out_path)
        
        del df
        gc.collect()
        
    print(f"    📦 历史序列包含物品数: {len(history_item_ids)}")
    return history_item_ids

def step3_process_item_features(target_ids, history_ids):
    print("\n🔵 [Step 3] 生成物品 Embedding 表...")
    final_item_ids = target_ids.union(history_ids)
    print(f"    ∑ 最终物品池大小: {len(final_item_ids)} (Target: {len(target_ids)} + History: {len(history_ids)})")
    
    # 保存 ID 列表
    with open(os.path.join(MICRO_DIR, "final_item_ids.txt"), "w") as f:
        for iid in final_item_ids: f.write(f"{iid}\n")

    for filename in FILES["item_feat"]:
        src_path = os.path.join(SOURCE_DIR, filename)
        out_path = os.path.join(MICRO_DIR, filename)
        
        if os.path.exists(src_path):
            print(f"    ✂️ 过滤文件: {filename} ...")
            df = pd.read_parquet(src_path)
            if "205" in df.columns:
                df = df[df["205"].isin(final_item_ids)]
                save_parquet(df, out_path)
            del df
            gc.collect()

def step4_build_wide_table():
    print("\n🔵 [Step 4] DuckDB 生成宽表 (Join)...")
    con = duckdb.connect()
    
    for mode in ["train", "test"]:
        sample_file = os.path.join(MICRO_DIR, f"{mode}_samples.parquet")
        user_file = os.path.join(MICRO_DIR, f"{mode}_user_features.parquet")
        item_file = os.path.join(MICRO_DIR, "item_features.parquet")
        
        if not (os.path.exists(sample_file) and os.path.exists(user_file)):
            print(f"    ⚠️ 跳过 {mode}: 中间文件不全")
            continue

        output_dir = os.path.join(MICRO_DIR, mode)
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"    🏗️ 正在处理 {mode} 集...")
        
        # 确保 item_features 存在，如果不存在左连接可能会有问题，但 DuckDB 允许 left join 即使表空
        # 这里为了稳健，如果 item_file 不存在，可以考虑只 join user
        item_join_clause = f"LEFT JOIN '{item_file}' AS i ON s.\"205\" = i.\"205\"" if os.path.exists(item_file) else ""
        item_select_clause = ", i.\"206\", i.\"213\", i.\"214\"" if os.path.exists(item_file) else ""

        query = f"""
        SELECT 
            s."label_0", s."129_1",
            u."130_1", u."130_2", u."130_3", u."130_4", u."130_5",
            u."150_2_180", u."151_2_180", 
            s."205"
            {item_select_clause}
        FROM '{sample_file}' AS s
        LEFT JOIN '{user_file}' AS u ON s."129_1" = u."129_1"
        {item_join_clause}
        """
        
        try:
            reader = con.query(query).fetch_arrow_reader(batch_size=SHARD_SIZE)
            shard_id = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                    if len(batch) == 0: break
                    table = pa.Table.from_batches([batch])
                    pq.write_table(table, os.path.join(output_dir, f"{mode}-shard-{shard_id:06d}.parquet"), compression='ZSTD')
                    shard_id += 1
                except StopIteration: break
            print(f"    🎉 {mode} 完成: {shard_id} 个分片")
        except Exception as e:
            print(f"    ❌ DuckDB 执行失败: {e}")

if __name__ == "__main__":
    valid_users_dict, target_items = step1_process_samples_and_intersect()
    if valid_users_dict:
        history_items = step2_process_user_features(valid_users_dict)
        step3_process_item_features(target_items, history_items)
        step4_build_wide_table()
        print("\n✅ 任务全部完成")