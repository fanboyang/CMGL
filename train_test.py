""" Training and testing of the model
"""
import copy
from contextlib import nullcontext
import torch
from models import init_model_dict, GNNStage
from losses import get_mrf_loss, get_gnn_loss
from utils import (build_trte_graphs, load_splits, prepare_tensors,
                   cal_class_weight, compute_metrics,
                   compute_gradient_norm, check_gradient_health,
                   adaptive_gradient_clipping, save_run_outputs)

cuda = True if torch.cuda.is_available() else False


class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4, mode='max'):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best, self.stop, self.best_ep = 0, None, False, 0
    def __call__(self, score, epoch):
        if self.best is None:
            self.best, self.best_ep = score, epoch
            return False
        imp = score > self.best + self.min_delta if self.mode == 'max' else score < self.best - self.min_delta
        if imp:
            self.best, self.best_ep, self.counter = score, epoch, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_mrf(data_folder, view_list, num_class, lr_mrf, num_epoch_mrf,
              strategy="intersection"):
    test_interval = 50
    num_view = len(view_list)
    device = torch.device('cuda' if cuda else 'cpu')
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    scaler_mrf = None
    grad_hist_mrf = []

    data_tr_raw, data_val_raw, data_te_raw, labels_tr, labels_val, labels_te = \
        load_splits(data_folder, view_list)
    data_tr, labels_tr_t, tr_stats = prepare_tensors(data_tr_raw, labels_tr, device)
    data_val, _, _ = prepare_tensors(data_val_raw, labels_val, device, tr_stats)
    data_te, _, _ = prepare_tensors(data_te_raw, labels_te, device, tr_stats)
    cw = torch.FloatTensor(cal_class_weight(labels_tr, num_class)).to(device)
    dim_list = [x.shape[1] for x in data_tr]

    model_dict = init_model_dict(num_view, num_class, dim_list, strategy=strategy)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    optim_mrf = torch.optim.Adam(model_dict["MRF"].parameters(), lr=lr_mrf)
    sched_mrf = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_mrf, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    print("\nTraining MRF...")
    for epoch in range(1, num_epoch_mrf + 1):
        model_dict["MRF"].train()
        optim_mrf.zero_grad()
        amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            out = model_dict["MRF"](data_tr)
            loss = get_mrf_loss(out, labels_tr_t, num_class, epoch)
        if scaler_mrf is not None:
            scaler_mrf.scale(loss).backward()
            scaler_mrf.unscale_(optim_mrf)
        else:
            loss.backward()
        grad_norm_before, _ = compute_gradient_norm(model_dict["MRF"])
        _, _ = check_gradient_health(model_dict["MRF"], verbose=False)
        _, _, grad_hist_mrf = adaptive_gradient_clipping(
            model_dict["MRF"], max_norm=5.0, grad_history=grad_hist_mrf,
            adaptive=True, precomputed_norm=grad_norm_before)
        if scaler_mrf is not None:
            scaler_mrf.step(optim_mrf)
            scaler_mrf.update()
        else:
            optim_mrf.step()
        sched_mrf.step(loss.item())
        if epoch % test_interval == 0:
            print("  MRF Epoch {:d} | Loss: {:.4f}".format(epoch, loss.item()))

    model_dict["MRF"].eval()
    with torch.no_grad():
        conf_tr = model_dict["MRF"](data_tr)['classification_confidence'].detach().clone()
        conf_val = model_dict["MRF"](data_val)['classification_confidence'].detach().clone()
        conf_te = model_dict["MRF"](data_te)['classification_confidence'].detach().clone()

    return {
        'data_tr': data_tr, 'data_val': data_val, 'data_te': data_te,
        'labels_tr': labels_tr, 'labels_val': labels_val, 'labels_te': labels_te,
        'labels_tr_t': labels_tr_t,
        'conf_tr': conf_tr, 'conf_val': conf_val, 'conf_te': conf_te,
        'cw': cw, 'dim_list': dim_list, 'num_view': num_view,
        'num_class': num_class, 'device': device,
        'use_amp': use_amp, 'amp_dtype': amp_dtype, 'strategy': strategy,
    }


def train_gnn(mrf, lr_gnn, num_epoch_gnn, knn_k,
              dataset_name=None, es_patience=50, output_dir=None,
              warmup_mode=False):
    num_view = mrf['num_view']
    num_class = mrf['num_class']
    device = mrf['device']
    use_amp = mrf['use_amp']
    amp_dtype = mrf['amp_dtype']
    data_tr, data_val, data_te = mrf['data_tr'], mrf['data_val'], mrf['data_te']
    labels_tr, labels_val, labels_te = mrf['labels_tr'], mrf['labels_val'], mrf['labels_te']
    labels_tr_t = mrf['labels_tr_t']
    conf_tr, conf_val, conf_te = mrf['conf_tr'], mrf['conf_val'], mrf['conf_te']
    cw, dim_list, strategy = mrf['cw'], mrf['dim_list'], mrf['strategy']
    scaler_gnn = None
    grad_hist_gnn = []

    adj_tr_l, adj_val_l, n_tr, n_val = build_trte_graphs(data_tr, data_val, knn_k, device)
    _, adj_te_l, _, n_te = build_trte_graphs(data_tr, data_te, knn_k, device)

    data_all = [torch.cat([data_tr[v], data_val[v]], 0) for v in range(num_view)]
    tr_idx = torch.arange(n_tr, device=device)
    val_idx = torch.arange(n_tr, n_tr + n_val, device=device)
    conf_all = torch.cat([conf_tr, conf_val], 0)
    conf_test_all = torch.cat([conf_tr, conf_te], 0)

    model = GNNStage(num_view, num_class, dim_list, strategy=strategy)
    if cuda:
        model.cuda()
    optim_gnn = torch.optim.Adam(model.parameters(), lr=lr_gnn, weight_decay=5e-4)
    sched_gnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_gnn, mode="max", factor=0.5, patience=30, min_lr=1e-6)
    es = EarlyStopping(patience=es_patience, mode='max')
    best_state, best_metrics, best_epoch = None, None, 0

    print("\nTraining GNN...")
    for epoch in range(1, num_epoch_gnn + 1):
        model.train()
        optim_gnn.zero_grad()
        amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            out = model(data_all, adj_tr_l, conf_all)
            masked_out = {k: v[tr_idx] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n_tr + n_val else v
                          for k, v in out.items()}
            loss = get_gnn_loss(masked_out, labels_tr_t, cw)
        if scaler_gnn is not None:
            scaler_gnn.scale(loss).backward()
            scaler_gnn.unscale_(optim_gnn)
        else:
            loss.backward()
        grad_norm_before, _ = compute_gradient_norm(model)
        _, _ = check_gradient_health(model, verbose=False)
        _, _, grad_hist_gnn = adaptive_gradient_clipping(
            model, max_norm=5.0, grad_history=grad_hist_gnn,
            adaptive=True, precomputed_norm=grad_norm_before)
        if scaler_gnn is not None:
            scaler_gnn.step(optim_gnn)
            scaler_gnn.update()
        else:
            optim_gnn.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data_all, adj_val_l, conf_all)
            val_prob = val_out['prob'][val_idx]
            val_preds = val_prob.argmax(1)
        val_m = compute_metrics(labels_val, val_preds, val_prob, num_class)
        val_f1 = val_m['f1_macro']
        sched_gnn.step(val_f1)

        if best_metrics is None or val_f1 > best_metrics.get('f1_macro', -1):
            best_metrics = val_m.copy()
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if es(val_f1, epoch):
            print("  Early stop {:d}".format(epoch))
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if warmup_mode:
        return best_metrics or {}

    model.eval()
    data_test_all = [torch.cat([data_tr[v], data_te[v]], 0) for v in range(num_view)]
    te_mask = torch.arange(n_tr, n_tr + n_te, device=device)
    with torch.no_grad():
        te_out = model(data_test_all, adj_te_l, conf_test_all)
        te_prob = te_out['prob'][te_mask].cpu().numpy()
    te_labels = labels_te
    te_preds = te_prob.argmax(1)
    te_metrics = compute_metrics(te_labels, te_preds, te_prob, num_class)
    te_summary = {
        "dataset": dataset_name,
        "best_epoch": int(best_epoch),
        "num_classes": int(num_class),
        "knn_k": int(knn_k),
        **{k: float(v) for k, v in te_metrics.items()},
    }

    if output_dir is not None:
        save_run_outputs(output_dir, te_summary, te_labels, te_preds, te_prob)

    print("  Test ACC: {:.3f}  F1-macro: {:.3f}".format(te_metrics['accuracy'], te_metrics['f1_macro']))

    return te_summary


def warmup_knn_k(mrf, lr_gnn, num_epoch_gnn,
                 dataset_name=None, es_patience=50,
                 candidates=None):
    import io, contextlib
    if candidates is None:
        candidates = [7, 11, 15, 19, 23]
    results = []
    for k in candidates:
        with contextlib.redirect_stdout(io.StringIO()):
            val_m = train_gnn(mrf, lr_gnn, num_epoch_gnn, k,
                              dataset_name=dataset_name,
                              es_patience=es_patience,
                              warmup_mode=True)
        f1 = val_m.get('f1_macro', 0.0)
        results.append((k, f1))
        print("  warmup k={:d}".format(k))
    best_k, best_f1 = select_best_k(results)
    print("  => selected k={:d}".format(best_k))
    return best_k


def select_best_k(results):
    
    return max(results, key=lambda x: (x[1], x[0]))
