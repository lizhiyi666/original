"""
训练约束分类器。
用法: python train_classifier.py --run_id <run_id> --epochs 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from constraint_classifier import ConstraintClassifier, compute_constraint_label
from evaluate_utils import get_task, get_run_data


def train_classifier(args):
    data_name, seed, run_path = get_run_data(args.run_id, "wandb")
    task, datamodule = get_task(run_path, data_root="./")

    dd = task.discrete_diffusion
    dd.eval()
    device = next(dd.parameters()).device

    # 导入 index_to_log_onehot
    from discrete_diffusion.diffusion_transformer import index_to_log_onehot

    classifier = ConstraintClassifier(
        num_classes=dd.num_classes,
        type_classes=dd.type_classes,
        num_spectial=dd.num_spectial,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_timesteps=dd.num_timesteps,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training classifier for {args.epochs} epochs on device={device}")
    print(f"  num_classes={dd.num_classes}, type_classes={dd.type_classes}, "
          f"num_spectial={dd.num_spectial}, num_timesteps={dd.num_timesteps}")

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_pos = 0
        total_neg = 0

        for batch in tqdm(datamodule.train_dataloader(), desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            B = batch.batch_size

            # 检查必要数据
            if not hasattr(batch, 'po_matrix') or batch.po_matrix is None:
                continue
            if not hasattr(batch, 'category_mask') or batch.category_mask is None:
                continue

            # x_0: ground truth token indices
            # 需要构造 content token（与 sample_fast 类似的布局）
            # 从 batch 中获取 marks（即 category + poi 交错序列）
            if hasattr(batch, 'marks') and batch.marks is not None:
                x_start = batch.marks.long()  # [B, L]
            else:
                continue

            L = x_start.shape[1]

            # 随机采样时间步
            t = torch.randint(0, dd.num_timesteps, (B,), device=device)

            # 加噪: x_0 → x_t
            try:
                log_x_start = index_to_log_onehot(x_start, dd.num_classes)
                log_x_t = dd.q_sample(log_x_start, t, batch)
            except Exception as e:
                print(f"[WARN] q_sample failed: {e}, skipping batch")
                continue

            # 计算 label
            labels = compute_constraint_label(
                x_start, batch.po_matrix,
                batch.category_mask,
                dd.num_spectial, dd.type_classes,
            )

            total_pos += (labels == 1.0).sum().item()
            total_neg += (labels == 0.0).sum().item()

            # 前向传播
            logits = classifier(log_x_t.detach(), t, batch.category_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            predictions = (logits > 0).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += B

        scheduler.step()

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                  f"Pos={total_pos}, Neg={total_neg}, Total={total_samples}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: No valid batches processed!")
            print("  Check: does train data have po_matrix and category_mask?")

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"constraint_classifier_{args.run_id}.pt")
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'config': {
            'num_classes': dd.num_classes,
            'type_classes': dd.type_classes,
            'num_spectial': dd.num_spectial,
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'num_timesteps': dd.num_timesteps,
            'dropout': args.dropout,
        }
    }, save_path)
    print(f"Classifier saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    train_classifier(args)