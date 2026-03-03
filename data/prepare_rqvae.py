import pandas as pd
import numpy as np
import os

# ================= 配置 =================
# 输入文件路径
NUM_USERS=10000
INPUT_PARQUET = "./TAOBAO_MM/micro/scl_embedding_int8_p90.parquet"

# 输出目录
OUTPUT_DIR = "./TAOBAO_MM/rqvae_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件名
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "item_emb.parquet")

def process_data_to_parquet():
    print(f"🚀 Loading parquet from: {INPUT_PARQUET}")
    
    # 1. 读取 Parquet
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 2. 检查列名
    id_col = '205'      # 原始 Item ID
    emb_col = '205_c'   # 原始 Embedding (Int8 list)
    
    if id_col not in df.columns:
        raise ValueError(f"找不到 ItemID 列 {id_col}，当前列: {df.columns}")
        
    print(f"   Original Rows: {len(df)}")
    # 3. 按 Item ID 排序 (非常重要！)
    # RQ-VAE 训练后生成的 ID 是基于行号的，所以必须保证输入文件的顺序是固定的（按ID排序）
    print("   Sorting by Item ID...")
    df = df.sort_values(by=id_col).reset_index(drop=True)
    
    # 4. 提取并转换 Embedding
    print("   Processing Embeddings (Int8 -> Float32)...")
    
    # 将 DataFrame 的 list 列堆叠成 Numpy 矩阵
    emb_matrix = np.stack(df[emb_col].values)
    
    # 判断并执行反量化
    if emb_matrix.dtype == np.int8 or emb_matrix.dtype == np.uint8:
        print("   Detected Int8. Dividing by 127.0 to restore Float32...")
        # 转换：除以 127 归一化到 [-1, 1]
        emb_matrix = emb_matrix.astype(np.float32) / 127.0
    else:
        print("   Detected Float. Keeping as Float32.")
        emb_matrix = emb_matrix.astype(np.float32)

    # 5. 将处理好的矩阵放回 DataFrame
    # Parquet 存储向量的最佳方式是 List[float]
    # 我们把 numpy 矩阵的每一行转回 list 赋值给新列
    df['embedding'] = list(emb_matrix)
    
    # 6. 构建最终的 DataFrame
    # 只保留 ID 和 Embedding 两列，并重命名为标准名称
    final_df = df[[id_col, 'embedding']].rename(columns={id_col: 'item_id'})
    
    # 7. 保存为 Parquet
    print(f"💾 Saving to Parquet: {OUTPUT_PARQUET}")
    final_df.to_parquet(OUTPUT_PARQUET)
    
    # 8. 验证
    print("✅ Done!")
    print(f"   Shape: {final_df.shape}")
    print(f"   Columns: {final_df.columns.tolist()}")
    print(f"   Example ID: {final_df.iloc[0]['item_id']}")
    print(f"   Example Emb Type: {type(final_df.iloc[0]['embedding'])}")

if __name__ == "__main__":
    process_data_to_parquet()