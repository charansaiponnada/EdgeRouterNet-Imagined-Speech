import os
import random
import numpy as np
import torch
import time
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_run_id():
    return time.strftime("%Y%m%d_%H%M%S")

def safe_cuda_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            if isinstance(r, dict):
                f.write(",".join(str(r.get(h, "")) for h in header) + "\n")
            else:
                f.write(",".join(str(x) for x in r) + "\n")

def metrics_from_preds(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))

    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    return {
        "acc": acc,
        "bacc": bacc,
        "macroF1": float(np.mean(f1)),
        "weightedF1": float(np.sum(f1 * sup) / (np.sum(sup) + 1e-12)),
        "per_class": {"prec": pr.tolist(), "rec": rc.tolist(), "f1": f1.tolist(), "sup": sup.tolist()},
    }

def build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep) / float(max(1, warmup_epochs))
        progress = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_crop_starts(TA, crop_stride):
    return list(range(0, int(TA) - 256 + 1, int(crop_stride)))

def softmax_weights_from_scores(score_dict, alpha=10.0):
    names = sorted(list(score_dict.keys()))
    s = np.array([score_dict[n] for n in names])
    w = np.exp(alpha * s) / np.sum(np.exp(alpha * s))
    return {n: float(v) for n, v in zip(names, w)}

import torch.nn.functional as F

def route_probs_to_5(len_logits, short_logits, long_logits, gate_temp=1.0, head_temp=1.0):
    p_len   = torch.softmax(len_logits / gate_temp, dim=1)
    p_short = torch.softmax(short_logits / head_temp, dim=1)
    p_long  = torch.softmax(long_logits  / head_temp, dim=1)

    B = p_len.size(0)
    p5 = torch.zeros((B, 5), device=len_logits.device)
    
    p5[:, 0] = p_len[:, 1] * p_long[:, 0]
    p5[:, 1] = p_len[:, 1] * p_long[:, 1]
    p5[:, 3] = p_len[:, 1] * p_long[:, 2]
    p5[:, 2] = p_len[:, 0] * p_short[:, 0]
    p5[:, 4] = p_len[:, 0] * p_short[:, 1]
    return p5

@torch.no_grad()
def trial_probs_one_model_direct(model, arc, x_cta, crop_starts,
                                 keep_frac=1.0, use_conf_filter=False,
                                 beta_arc=0.55, gate_temp=1.0, head_temp=1.0, t_arc=1.0):
    crops = np.stack([x_cta[:, s:s+256] for s in crop_starts], axis=0)
    xb = torch.from_numpy(crops).float().to(next(model.parameters()).device)
    
    len_l, sh_l, lo_l, emb = model(xb)
    
    if (not use_conf_filter) or (len(crop_starts) <= 1):
        p_route = route_probs_to_5(len_l, sh_l, lo_l, gate_temp, head_temp)
        # ArcFace weight is in arc.weight
        W = F.normalize(arc.weight, dim=1)
        logits_arc = F.linear(F.normalize(emb, dim=1), W) * float(arc.s)
        p_arc = torch.softmax(logits_arc / t_arc, dim=1)
        p = (1.0 - beta_arc) * p_route + beta_arc * p_arc
        return p.mean(dim=0)

    # With confidence filter
    p_route_f = route_probs_to_5(len_l, sh_l, lo_l, 1.0, 1.0)
    W = F.normalize(arc.weight, dim=1)
    logits_arc_f = F.linear(F.normalize(emb, dim=1), W) * float(arc.s)
    p_arc_f = torch.softmax(logits_arc_f / 1.0, dim=1)
    p_f = (1.0 - beta_arc) * p_route_f + beta_arc * p_arc_f
    
    # margin
    top2 = torch.topk(p_f, k=2, dim=1).values
    m = top2[:, 0] - top2[:, 1]
    
    kkeep = max(1, int(math.ceil(keep_frac * len(crop_starts))))
    idx_keep = torch.topk(m, k=kkeep, largest=True).indices
    
    p_route = route_probs_to_5(len_l[idx_keep], sh_l[idx_keep], lo_l[idx_keep], gate_temp, head_temp)
    logits_arc = F.linear(F.normalize(emb[idx_keep], dim=1), W) * float(arc.s)
    p_arc = torch.softmax(logits_arc / t_arc, dim=1)
    p = (1.0 - beta_arc) * p_route + beta_arc * p_arc
    return p.mean(dim=0)
