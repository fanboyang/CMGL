""" Loss functions
"""
import torch
import torch.nn.functional as F


def edl_loss(evi_list, labels, K, epoch, anneal_step=50):
    # Evidential loss for the MRF stage.
    tgt = F.one_hot(labels, K).float()
    ac = min(1.0, epoch / anneal_step)
    total = 0
    for e in evi_list:
        a = e + 1
        S = a.sum(1, keepdim=True)
        A = (tgt * (torch.digamma(S) - torch.digamma(a))).sum(1, keepdim=True)
        ka = (a - 1) * (1 - tgt) + 1
        sa = ka.sum(1, keepdim=True)
        kl = (torch.lgamma(sa) - torch.lgamma(torch.tensor(float(K), device=e.device))
              - torch.lgamma(ka).sum(1, keepdim=True)
              + ((ka - 1) * (torch.digamma(ka) - torch.digamma(sa))).sum(1, keepdim=True))
        total += torch.mean(A + ac * kl)
    return total / len(evi_list)


def ce_smooth(logits, labels, smoothing=0.1, cw=None):
    # Cross-entropy with label smoothing.
    K = logits.shape[1]
    sm = torch.full_like(logits, smoothing / (K - 1))
    sm.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    lp = F.log_softmax(logits, 1)
    loss = -(sm * lp).sum(1)
    if cw is not None:
        loss = loss * cw[labels]
    return loss.mean()


def conf_diversity_loss(conf, ent_ratio=0.3, gap_min=0.15):
    # Encourage informative confidence distributions.
    M = conf.shape[1]
    ent = -(conf * torch.log(conf + 1e-8)).sum(1)
    me = torch.log(torch.tensor(float(M), device=conf.device))
    d = F.relu(ent - ent_ratio * me).mean()
    mx, _ = conf.max(1)
    mn, _ = conf.min(1)
    return d + 2.0 * F.relu(gap_min - (mx - mn)).mean()


def sup_contrastive(emb, labels, temp=0.1):
    # Supervised contrastive loss on embeddings.
    N = emb.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=emb.device)
    en = F.normalize(emb, dim=1)
    sim = en @ en.T / temp
    lv = labels.view(-1, 1)
    msk = (lv == lv.T).float()
    diag = torch.eye(N, device=emb.device)
    msk = msk * (1 - diag)
    exp_l = torch.exp(sim) * (1 - diag)
    lp = sim - torch.log(exp_l.sum(1, keepdim=True) + 1e-8)
    pc = msk.sum(1)
    ml = (msk * lp).sum(1) / (pc + 1e-8)
    v = pc > 0
    if not v.any():
        return torch.tensor(0.0, device=emb.device)
    return -ml[v].mean()


def get_mrf_loss(mrf_out, labels, K, epoch, anneal_step=50,
                 w_edl=1.5, w_cls=1.5, w_div=1.0):
    # Total loss for the MRF stage.
    loss_edl = edl_loss(mrf_out['evidence_list'], labels, K, epoch, anneal_step)
    bel = torch.stack(mrf_out['belief_list'], 1)
    conf = mrf_out['classification_confidence']
    fb = (conf.unsqueeze(-1) * bel).sum(1)
    loss_cls = F.nll_loss(torch.log(fb + 1e-8), labels)
    loss_div = conf_diversity_loss(conf)
    return w_edl * loss_edl + w_cls * loss_cls + w_div * loss_div


def get_gnn_loss(gnn_out, labels, cw=None, smoothing=0.1,
                 w_cls=3.0, w_con=1.0):
    # Total loss for the GNN stage.
    loss_cls = ce_smooth(gnn_out['logits'], labels, smoothing, cw)
    loss_con = sup_contrastive(gnn_out['embeddings'], labels) if gnn_out.get('embeddings') is not None else 0.0
    return w_cls * loss_cls + w_con * loss_con
