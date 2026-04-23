""" Utility functions
"""
import os
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

cuda = True if torch.cuda.is_available() else False


def cosine_dist(x1, x2=None):
    # Cosine distance between feature matrices.
    x2 = x1 if x2 is None else x2
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)
    return 1 - torch.mm(x1, x2.T)


def build_knn(feats, k):
    # Build an undirected kNN graph for one view.
    N = feats.shape[0]
    k = min(k, N - 1)
    dist = cosine_dist(feats)
    dist.fill_diagonal_(float('inf'))
    _, idx = dist.topk(k, largest=False, dim=1)
    row = torch.arange(N, device=feats.device).unsqueeze(1).expand(-1, k).flatten()
    col = idx.flatten()
    ei = torch.stack([row, col], 0)
    ei = torch.cat([ei, ei.flip(0)], 1)
    ei = torch.unique(ei, dim=1)
    return ei


def build_trte_graphs(data_tr, data_te, k, device='cpu'):
    # Build train-only and train-target graphs for each view.
    V = len(data_tr)
    N_tr = data_tr[0].shape[0]
    N_te = data_te[0].shape[0]
    adj_tr_l, adj_te_l = [], []
    for v in range(V):
        tr = data_tr[v] if isinstance(data_tr[v], torch.Tensor) else torch.FloatTensor(data_tr[v]).to(device)
        te = data_te[v] if isinstance(data_te[v], torch.Tensor) else torch.FloatTensor(data_te[v]).to(device)
        adj_tr = build_knn(tr, k)
        adj_tr_l.append(adj_tr.to(device))

        # Add bidirectional edges between train and target nodes.
        d_te2tr = cosine_dist(te, tr)
        d_tr2te = d_te2tr.T
        k_te = min(k, N_te)
        k_tr = min(k, N_tr)
        _, idx_tr2te = d_tr2te.topk(k_te, largest=False, dim=1)
        _, idx_te2tr = d_te2tr.topk(k_tr, largest=False, dim=1)
        r1 = torch.arange(N_tr, device=device).unsqueeze(1).expand(-1, k_te).flatten()
        c1 = idx_tr2te.flatten() + N_tr
        r2 = torch.arange(N_te, device=device).unsqueeze(1).expand(-1, k_tr).flatten() + N_tr
        c2 = idx_te2tr.flatten()
        e_cross = torch.cat([torch.stack([r1, c1], 0), torch.stack([r2, c2], 0)], 1)
        adj_te = torch.cat([adj_tr.clone(), e_cross], 1)
        adj_te = torch.cat([adj_te, adj_te.flip(0)], 1)
        adj_te = torch.unique(adj_te, dim=1)
        adj_te_l.append(adj_te.to(device))
    return adj_tr_l, adj_te_l, N_tr, N_te


def load_splits(data_folder, view_list):
    # Load one prepared train-val-test split.
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_val = np.loadtxt(os.path.join(data_folder, "labels_val.csv"), delimiter=',').astype(int)
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)
    data_tr, data_val, data_te = [], [], []
    for v in view_list:
        data_tr.append(np.nan_to_num(np.loadtxt(os.path.join(data_folder, str(v) + "_tr.csv"), delimiter=','), nan=0.0, posinf=0.0, neginf=0.0))
        data_val.append(np.nan_to_num(np.loadtxt(os.path.join(data_folder, str(v) + "_val.csv"), delimiter=','), nan=0.0, posinf=0.0, neginf=0.0))
        data_te.append(np.nan_to_num(np.loadtxt(os.path.join(data_folder, str(v) + "_te.csv"), delimiter=','), nan=0.0, posinf=0.0, neginf=0.0))
    return data_tr, data_val, data_te, labels_tr, labels_val, labels_te


def prepare_tensors(data_list, labels, device, train_stats=None):
    # Convert arrays to tensors and standardize features.
    tensors, stats = [], []
    for i, d in enumerate(data_list):
        t = torch.FloatTensor(d).to(device) if not isinstance(d, torch.Tensor) else d.to(device)
        if train_stats is not None:
            mu, std = train_stats[i]
            mu, std = mu.to(device), std.to(device)
        else:
            mu = t.mean(0, keepdim=True)
            std = t.std(0, unbiased=False, keepdim=True).clamp_min_(1e-8)
        t = (t - mu) / std
        tensors.append(t)
        stats.append((mu.detach().clone(), std.detach().clone()))
    labels_t = torch.LongTensor(labels).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
    return tensors, labels_t, stats


def configure_torch_runtime(device=None, allow_tf32=True, matmul_precision="high",
                            cudnn_benchmark=True):
    # Configure PyTorch runtime options.
    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except (AttributeError, RuntimeError):
            pass

    if device is None:
        device = torch.device('cuda' if cuda else 'cpu')
    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def cal_class_weight(labels, K):
    # Compute smoothed inverse-frequency class weights.
    cnt = np.bincount(labels.astype(int), minlength=K)
    w = np.sqrt(cnt.max() / (cnt + 1e-8))
    return w / w.sum() * K


def compute_metrics(labels, preds, probs=None, K=2):
    # Compute classification metrics.
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if probs is not None and isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1_m = f1_score(labels, preds, average='macro')
    f1_w = f1_score(labels, preds, average='weighted')
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    auc = 0.5
    if probs is not None:
        try:
            if K == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class='ovr')
        except (ValueError, TypeError):
            # Fall back when AUC is undefined.
            auc = 0.5
    return {
        'accuracy': acc,
        'f1_macro': f1_m,
        'macro_recall': macro_recall,
        'f1_weighted': f1_w,
        'auc': auc,
    }


def save_run_outputs(output_dir, summary, labels, preds, probs):
    # Save summary and per-sample predictions.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_index", "label", "prediction"]
        if probs is not None:
            header.extend([f"prob_{i}" for i in range(probs.shape[1])])
        writer.writerow(header)
        for i, (label, pred) in enumerate(zip(labels, preds)):
            row = [i, int(label), int(pred)]
            if probs is not None:
                row.extend([float(x) for x in probs[i]])
            writer.writerow(row)


def compute_gradient_norm(model, return_param_norms=False):
    # Compute the global gradient norm.
    total_norm_sq = None
    param_norms = {} if return_param_norms else None

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm_sq = param.grad.detach().pow(2).sum()
        total_norm_sq = grad_norm_sq if total_norm_sq is None else total_norm_sq + grad_norm_sq
        if return_param_norms:
            param_norms[name] = float(grad_norm_sq.sqrt().detach().cpu())

    total_norm = 0.0 if total_norm_sq is None else float(total_norm_sq.sqrt().detach().cpu())
    return total_norm, (param_norms or {})


def check_gradient_health(model, verbose=False):
    # Optionally inspect gradients for numerical issues.
    if not verbose:
        return True, []

    issues = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.data
        if torch.isnan(grad).any():
            issues.append(f"{name}: NaN gradient")
        if torch.isinf(grad).any():
            issues.append(f"{name}: Inf gradient")
        grad_norm = grad.norm(2).item()
        if grad_norm > 100:
            issues.append(f"{name}: large gradient ({grad_norm:.2f})")
        if grad_norm < 1e-7 and param.requires_grad:
            issues.append(f"{name}: tiny gradient ({grad_norm:.2e})")
    return len(issues) == 0, issues


def adaptive_gradient_clipping(model, max_norm=5.0, grad_history=None, adaptive=True,
                               precomputed_norm=None):
    # Clip gradients with a history-aware threshold.
    current_norm = precomputed_norm if precomputed_norm is not None else compute_gradient_norm(model)[0]

    if adaptive and grad_history is not None and len(grad_history) >= 10:
        mean_norm = np.mean(grad_history[-10:])
        std_norm = np.std(grad_history[-10:])
        adaptive_max = max(max_norm, mean_norm + 2 * std_norm)
    else:
        adaptive_max = max_norm

    was_clipped = current_norm > adaptive_max
    if was_clipped:
        torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_max)

    if grad_history is None:
        grad_history = []
    grad_history.append(current_norm)
    if len(grad_history) > 100:
        grad_history = grad_history[-100:]

    return adaptive_max, was_clipped, grad_history


DATASET_KNN_K = {}
KNN_K_DEFAULT = 15


def resolve_knn_k(dataset, override=None):
    # Resolve the kNN neighbor count for a dataset.
    if override is not None:
        return int(override)
    ds = dataset.upper() if dataset else None
    if ds and ds in DATASET_KNN_K:
        return int(DATASET_KNN_K[ds])
    return KNN_K_DEFAULT
