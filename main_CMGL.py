""" Run full 5-fold CMGL experiment on GS-BRCA
"""

from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np
import torch

from train_test import train_mrf, train_gnn, warmup_knn_k
from utils import configure_torch_runtime

NUM_CLASS = 5
VIEW_LIST = [1, 2, 3, 4]
LR_MRF = 5e-4
LR_GNN = 5e-4
NUM_EPOCH_MRF = 150
NUM_EPOCH_GNN = 500
NUM_FOLDS = 5
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    dataset_name = "GS-BRCA"
    root_dir = Path(__file__).resolve().parent
    output_root = root_dir / "results"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configure_torch_runtime(device, allow_tf32=True, matmul_precision="high")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_results = []
    for fold in range(1, NUM_FOLDS + 1):
        print("\nFold {}/{}".format(fold, NUM_FOLDS))
        set_seed(SEED)
        data_folder = str(root_dir / "GS-BRCA" / "fold{}".format(fold))
        output_dir = output_root / run_tag / "fold{}".format(fold)

        mrf = train_mrf(data_folder, VIEW_LIST, NUM_CLASS, LR_MRF, NUM_EPOCH_MRF)

        print("  Warm-up KNN k selection...")
        selected_k = warmup_knn_k(mrf, LR_GNN, NUM_EPOCH_GNN,
                                  dataset_name=dataset_name)
        set_seed(SEED)
        summary = train_gnn(mrf, LR_GNN, NUM_EPOCH_GNN, selected_k,
                            dataset_name=dataset_name, output_dir=output_dir)
        summary["fold"] = fold
        fold_results.append(summary)

    metric_keys = ["accuracy", "f1_macro"]
    all_keys = ["accuracy", "f1_macro", "macro_recall", "f1_weighted", "auc"]
    vals = {k: [r.get(k, 0.0) for r in fold_results] for k in all_keys}
    print("\n5-Fold Summary")
    print("  accuracy:  {:.4f} +/- {:.4f}".format(np.mean(vals["accuracy"]), np.std(vals["accuracy"])))
    print("  f1_macro:  {:.4f} +/- {:.4f}".format(np.mean(vals["f1_macro"]), np.std(vals["f1_macro"])))

    agg = {"dataset": dataset_name, "num_folds": NUM_FOLDS, "folds": fold_results}
    for k in all_keys:
        agg[k + "_mean"] = float(np.mean(vals[k]))
        agg[k + "_std"] = float(np.std(vals[k]))
    agg_path = output_root / run_tag / "5fold_summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print("\nSaved to {}".format(agg_path))
