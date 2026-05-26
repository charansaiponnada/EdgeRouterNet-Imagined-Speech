import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import sys

from .model import EdgeRouterNet, ArcMarginProduct
from .dataset import AllCropsDS, slice_window, EEGTransform
from .utils import (
    set_seed, safe_cuda_empty_cache, count_params, write_csv,
    metrics_from_preds, build_warmup_cosine_scheduler, route_probs_to_5,
    build_crop_starts, trial_probs_one_model_direct, softmax_weights_from_scores
)
from .config import (
    LENGTH_MAP, SHORT_REMAP, LONG_REMAP, NUM_CLASSES, CROP_LEN,
    RANDOM_STATE, CALIB_FRAC, ENSEMBLE_CONFIGS, PROBE_MEMBER, class_names,
    ARC_S, ARC_M, LAMBDA_ARC, ARC_START_EPOCH,
    EARLYSTOP_MIN_EPOCH, EARLYSTOP_PATIENCE, EVAL_EVERY_EPOCHS,
    CHEAP_EVAL_MAX_TRIALS, FULL_EVAL_EVERY, EMA_MOM,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WD,
    MIN_LR_RATIO, WARMUP_EPOCHS
)

def train_one_member(cfg, X_train_sub, y_train_sub, X_calib, y_calib,
                     TA, fold_tag, ab,
                     member_keep_frac,
                     run_dir, device, use_aug=True):
    
    set_seed(cfg["seed"])
    name = cfg["name"]
    crop_starts = build_crop_starts(TA, cfg["crop_stride"])

    model = EdgeRouterNet(
        kernels=cfg["kernels"],
        disable_csa=ab["disable_csa"],
        disable_eca=ab["disable_eca"],
        disable_ms2=ab["disable_ms2"],
    ).to(device)
    arc = ArcMarginProduct(in_features=model.emb_dim, out_features=5, s=ARC_S, m=ARC_M).to(device)

    print(f"\n[{fold_tag}] Member {name} | TA={TA} crops/trial={len(crop_starts)} keep_frac={member_keep_frac:.2f} aug={use_aug}")
    print(f"  params(model)={count_params(model)} params(arc)={count_params(arc)}")

    transform = EEGTransform() if use_aug else None
    ds = AllCropsDS(X_train_sub, y_train_sub, crop_starts, LENGTH_MAP, SHORT_REMAP, LONG_REMAP, transform=transform)
    loader = DataLoader(
        ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device=="cuda")
    )

    opt = torch.optim.AdamW(list(model.parameters()) + list(arc.parameters()), lr=DEFAULT_LR, weight_decay=DEFAULT_WD)
    sch = build_warmup_cosine_scheduler(opt, epochs=DEFAULT_EPOCHS, warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO)
    ce_none = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_ema, best_acc = -1.0, -1.0
    ema, no_improve = None, 0
    ckpt_path = os.path.join(run_dir, f"{fold_tag}_member_{name}_best.pt")
    hist_rows = []

    ep_bar = tqdm(range(1, DEFAULT_EPOCHS+1), desc=f"[{fold_tag}|{name}]", leave=True, dynamic_ncols=True, file=sys.stderr)

    for ep in ep_bar:
        t_ep0 = time.time()
        model.train(); arc.train()

        for xb, y_full, y_len, y_route in loader:
            xb, y_full, y_len, y_route = xb.to(device), y_full.to(device), y_len.to(device), y_route.to(device)
            opt.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                len_l, sh_l, lo_l, emb = model(xb)
                loss_len = F.cross_entropy(len_l, y_len)
                
                p_len = torch.softmax(len_l, dim=1)
                is_short, is_long = (y_len == 0), (y_len == 1)
                
                loss_short = xb.new_tensor(0.0)
                if is_short.any():
                    ce_s = ce_none(sh_l[is_short], y_route[is_short])
                    loss_short = (p_len[is_short, 0].detach() * ce_s).mean()

                loss_long = xb.new_tensor(0.0)
                if is_long.any():
                    ce_l = ce_none(lo_l[is_long], y_route[is_long])
                    loss_long = (p_len[is_long, 1].detach() * ce_l).mean()

                p5 = route_probs_to_5(len_l, sh_l, lo_l)
                loss_5 = -torch.log(p5.gather(1, y_full.view(-1,1)) + 1e-9).mean()

                loss = loss_len + loss_short + loss_long + loss_5
                if ep >= ARC_START_EPOCH:
                    arc_logits = arc(emb, y_full)
                    loss += LAMBDA_ARC * F.cross_entropy(arc_logits, y_full)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        
        sch.step()

        if ep % EVAL_EVERY_EPOCHS == 0:
            model.eval(); arc.eval()
            yt, yp = [], []
            n_eval = X_calib.shape[0] if (ep % FULL_EVAL_EVERY == 0 or ep == DEFAULT_EPOCHS) else min(X_calib.shape[0], CHEAP_EVAL_MAX_TRIALS)
            
            with torch.no_grad():
                for i in range(n_eval):
                    p = trial_probs_one_model_direct(model, arc, X_calib[i], crop_starts, keep_frac=member_keep_frac, beta_arc=ab["beta_arc"])
                    yp.append(int(torch.argmax(p).item()))
                    yt.append(int(y_calib[i]))
            
            calib_acc = accuracy_score(yt, yp)
            ema = calib_acc if ema is None else EMA_MOM * ema + (1.0 - EMA_MOM) * calib_acc
            
            if ema > best_ema + 1e-12:
                best_ema, best_acc, no_improve = ema, calib_acc, 0
                torch.save({"model": model.state_dict(), "arc": arc.state_dict(), "best_calib_acc": best_acc, "TA": TA, "crop_starts": crop_starts}, ckpt_path)
            else:
                no_improve += 1
            
            ep_bar.set_postfix(acc=round(calib_acc,4), ema=round(ema,4), ni=no_improve)
            if ep >= EARLYSTOP_MIN_EPOCH and no_improve >= EARLYSTOP_PATIENCE:
                print(f" Early stop at {ep}")
                break

        hist_rows.append({"epoch": ep, "calib_acc": calib_acc, "ema": ema, "sec": time.time()-t_ep0})

    write_csv(os.path.join(run_dir, f"{fold_tag}_member_{name}_trainlog.csv"), hist_rows, ["epoch","calib_acc","ema","sec"])
    return model, arc, crop_starts, best_acc, ckpt_path

def build_cache_for_member(desc, model, arc, crop_starts, X_eval):
    model.eval(); arc.eval()
    cache = []
    with torch.no_grad():
        for i in range(X_eval.shape[0]):
            crops = np.stack([X_eval[i, :, s:s+256] for s in crop_starts], axis=0)
            xb = torch.from_numpy(crops).float().to(next(model.parameters()).device)
            len_l, sh_l, lo_l, emb = model(xb)
            cache.append({
                "len_l": len_l.cpu(), "sh_l": sh_l.cpu(), "lo_l": lo_l.cpu(), "emb": emb.cpu(),
                "arc_w": arc.weight.detach().cpu(), "arc_s": float(arc.s)
            })
    return cache

def eval_ensemble_from_cache(caches_by_name, y_eval, keep_fracs_by_member, use_conf_filter, ensemble_weights, beta_arc):
    active = sorted(list(caches_by_name.keys()))
    probs_all = np.zeros((len(y_eval), NUM_CLASSES))
    for i in range(len(y_eval)):
        p_ens = np.zeros(NUM_CLASSES)
        for name in active:
            c = caches_by_name[name][i]
            p_trial = (1.0 - beta_arc) * route_probs_to_5(c["len_l"], c["sh_l"], c["lo_l"]).numpy() + \
                      beta_arc * torch.softmax(F.linear(F.normalize(c["emb"], dim=1), F.normalize(c["arc_w"], dim=1)) * c["arc_s"], dim=1).numpy()
            
            if use_conf_filter:
                m = np.sort(p_trial, axis=1)[:, -1] - np.sort(p_trial, axis=1)[:, -2]
                kkeep = max(1, int(math.ceil(keep_fracs_by_member[name] * len(m))))
                p_member = p_trial[np.argsort(m)[-kkeep:]].mean(axis=0)
            else:
                p_member = p_trial.mean(axis=0)
            p_ens += ensemble_weights[name] * p_member
        probs_all[i] = p_ens / (p_ens.sum() + 1e-9)
    return metrics_from_preds(y_eval, probs_all.argmax(axis=1)), confusion_matrix(y_eval, probs_all.argmax(axis=1), labels=list(range(NUM_CLASSES)))

def run_5fold_cv(ab, X_full, y_all, run_dir, device):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_accs, fold_rows = [], []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_all), 1):
        fold_tag = f"fold{fold_idx:02d}"
        X_train, y_train = X_full[train_idx], y_all[train_idx]
        X_val, y_val = X_full[val_idx], y_all[val_idx]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=CALIB_FRAC, random_state=RANDOM_STATE+fold_idx)
        tr_idx, cal_idx = next(sss.split(X_train, y_train))
        X_tr, y_tr, X_cal, y_cal = X_train[tr_idx], y_train[tr_idx], X_train[cal_idx], y_train[cal_idx]
        
        # Windows & Norm
        X_tr = slice_window(X_tr, 0, 1024); X_cal = slice_window(X_cal, 0, 1024); X_val = slice_window(X_val, 0, 1024)
        mu, sd = X_tr.mean(axis=(0,2), keepdims=True), X_tr.std(axis=(0,2), keepdims=True) + 1e-6
        X_tr, X_cal, X_val = (X_tr-mu)/sd, (X_cal-mu)/sd, (X_val-mu)/sd
        
        trained = []
        for cfg in ENSEMBLE_CONFIGS:
            m, a, cs, acc, _ = train_one_member(cfg, X_tr, y_tr, X_cal, y_cal, 1024, fold_tag, ab, 0.60, run_dir, device)
            trained.append((cfg["name"], m, a, cs))
            
        caches_val = {n: build_cache_for_member(n, m, a, cs, X_val) for n, m, a, cs in trained}
        met, cm = eval_ensemble_from_cache(caches_val, y_val, {n:0.60 for n in caches_val}, True, {n:1/3 for n in caches_val}, ab["beta_arc"])
        fold_accs.append(met["acc"])
        print(f"[{fold_tag}] Val Acc: {met['acc']:.4f}")
        fold_rows.append({"fold": fold_tag, "acc": met["acc"], "bacc": met["bacc"]})
        
    write_csv(os.path.join(run_dir, "cv_summary.csv"), fold_rows, ["fold", "acc", "bacc"])
    return {"acc_mean": np.mean(fold_accs), "acc_std": np.std(fold_accs)}

def finalfit_train_bundle(ab, X_train_full, y_train, run_dir, device):
    X_win = slice_window(X_train_full, 0, 1024)
    mu, sd = X_win.mean(axis=(0,2), keepdims=True), X_win.std(axis=(0,2), keepdims=True) + 1e-6
    X_win = (X_win - mu)/sd
    
    trained_members = {}
    for cfg in ENSEMBLE_CONFIGS:
        _, _, cs, acc, cp = train_one_member(cfg, X_win, y_train, X_win[:20], y_train[:10], 1024, "final", ab, 0.60, run_dir, device)
        trained_members[cfg["name"]] = {"ckpt": cp, "crop_starts": cs}
        
    bundle = {"ablation": ab, "norm": {"mu": mu.tolist(), "sd": sd.tolist()}, "members": trained_members}
    bp = os.path.join(run_dir, "finalfit_bundle.json")
    with open(bp, "w") as f: json.dump(bundle, f, indent=2)
    return bp
