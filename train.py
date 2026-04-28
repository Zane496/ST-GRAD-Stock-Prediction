# train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from models.st_grad import ST_GRAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(npz_path="data/processed/stock_dataset.npz", adj_path="data/processed/normalized_adj_matrix.npy"):
    """Load preprocessed dataset and adjacency matrix."""
    if not os.path.exists(npz_path) or not os.path.exists(adj_path):
        raise FileNotFoundError("Processed data not found. Please ensure data files exist.")

    data = np.load(npz_path, allow_pickle=True)
    X_train = torch.FloatTensor(data['X_train']).permute(0, 3, 1, 2)
    y_train = torch.FloatTensor(data['y_train']).squeeze(-1)
    X_val = torch.FloatTensor(data['X_val']).permute(0, 3, 1, 2)
    y_val = torch.FloatTensor(data['y_val']).squeeze(-1)

    A = torch.FloatTensor(np.load(adj_path)).to(device)
    return X_train, X_val, y_train, y_val, A


def evaluate(model, data_loader, criterion, A):
    """Evaluate the model and return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, A)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }


def train_model(args):
    """Main training routine."""
    X_train, X_val, y_train, y_val, A = load_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = ST_GRAD(X_train.size(1), 1, X_train.size(3), X_train.size(2),
                    args.hidden_channels, args.num_layers, args.diffusion_steps, args.embed_dim).to(device)

    pos_weight = torch.tensor([(1 - torch.mean(y_train)) / (torch.mean(y_train) + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_f1 = 0.0
    best_metrics = None  # 用于保存最佳轮次的完整指标
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "st_grad_best_model.pth")

    print(f"--- Training Started ---")
    for epoch in range(args.epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch, A), y_batch)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, criterion, A)
        scheduler.step(val_metrics['f1'])

        # 当 F1 提高时，更新并保存当前的完整指标
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics  # 记录当前全套指标
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

    # --- 打印最终结果字典 ---
    if best_metrics:
        # 将单次运行结果适配到你的模板格式中
        print("\n" + "=" * 35)
        print("  FINAL PERFORMANCE METRICS (BEST)")
        print("=" * 35)
        print(f"  ACC    : {best_metrics['accuracy']:.4f}")
        print(f"  PRE    : {best_metrics['precision']:.4f}")
        print(f"  REC    : {best_metrics['recall']:.4f}")
        print(f"  F1     : {best_metrics['f1']:.4f}")
        print(f"  AUC    : {best_metrics['auc']:.4f}")
        print("=" * 35)

    print(f"Training Done. Best F1 saved at: {best_path}")


def test_model(args):
    """Load a specific saved model and evaluate it on the validation set."""
    if not os.path.exists(args.load_path):
        print(f"Error: Model file not found at {args.load_path}")
        return

    X_train, X_val, y_train, y_val, A = load_data()
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = ST_GRAD(X_train.size(1), 1, X_train.size(3), X_train.size(2),
                    args.hidden_channels, args.num_layers, args.diffusion_steps, args.embed_dim).to(device)

    print(f"Loading weights from {args.load_path}...")
    model.load_state_dict(torch.load(args.load_path, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model, val_loader, criterion, A)

    print("\n=== Model Performance ===")
    for k, v in metrics.items():
        if k.lower() != 'loss':  # 过滤掉 loss
            print(f"{k.capitalize():10}: {v:.4f}")