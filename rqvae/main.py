import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
import argparse
import random
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from trainer import Trainer
import logging
import sys

from models.rqvae import RQVAE 

def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance RQVAE Training")

    # ================= 核心训练参数 =================
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--learner', type=str, default='adamw', help='optimizer: adam, sgd, adamw')  
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='scheduler type')     
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')        
    parser.add_argument('--save_limit', type=int, default=3, help='max checkpoints to keep')          
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=20384)
    
    # ================= 模型架构参数 =================
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=CODEBOOK)

    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    
    # ================= RQVAE 特有参数 =================
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)

    # ================= 路径 =================
    parser.add_argument("--dataset", type=str, default="taobao-mm", choices=['taobao-mm', 'kuairec'])
    parser.add_argument("--data_path", type=str, default="../data/TAOBAO_MM/rqvae_data/item_emb.parquet")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/taobao_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()
# ================= 工具函数 =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def setup_logger(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(save_dir, "train.log"))
        ]
    )
class GPUDataLoader:
    """
    一个简单的 DataLoader，专门配合全量加载到 GPU 的数据。
    它使得 Trainer 中的 `for data in iter_data` 能高效运行。
    """
    def __init__(self, data_tensor, batch_size, shuffle=True):
        self.data = data_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = data_tensor.device
        
    def __iter__(self):
        n = self.data.shape[0]
        # 在 GPU 上生成索引
        if self.shuffle:
            indices = torch.randperm(n, device=self.device)
        else:
            indices = torch.arange(n, device=self.device)
            
        for i in range(0, n, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.data[batch_indices]
            
    def __len__(self):
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size
if __name__ == '__main__':
    # 1. 设置
    CODEBOOK=[64,128,128]
    args = parse_args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    setup_logger(args.ckpt_dir)
    
    logger = logging.getLogger()
    logger.info(f"Using Device: {args.device}")

    # 2. 加载数据 (全量进显存)
    import pandas as pd
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    # 假设 embedding 列是 array 或 list
    if 'embedding' in df.columns:
        data_numpy = np.stack(df['embedding'].values)
    else:
        # 如果是纯数值列
        data_numpy = df.select_dtypes(include=[np.number]).values
        
    data_tensor = torch.from_numpy(data_numpy).float().to(args.device)
    logger.info(f"Data shape: {data_tensor.shape}")

    # 3. 封装数据为 Batch 迭代器
    # 这里我们不用 torch.utils.data.DataLoader，因为它在多 worker 下对 GPU tensor 支持不好
    # 使用自定义的 GPUDataLoader 完美契合你的 Trainer 逻辑
    train_loader = GPUDataLoader(data_tensor, args.batch_size, shuffle=True)

    # 4. 初始化模型
    # 注意：确保 RQVAE 类实现了 compute_loss 和 get_indices 方法
    from models.rqvae import RQVAE 
    input_dim = data_tensor.shape[1]
    
    model = RQVAE(in_dim=input_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    
    # 5. 调用你的 Trainer
    trainer = Trainer(args, model, data_num=len(train_loader))
    
    # 开始训练
    logger.info("Starting Training...")
    best_loss, best_coll = trainer.fit(train_loader)
    logger.info(f"Training Finished. Best Loss: {best_loss}, Best Collision: {best_coll}")