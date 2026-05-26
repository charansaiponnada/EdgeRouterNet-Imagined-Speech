import os
import argparse
import torch
from src.dataset import load_split_full
from src.train import run_5fold_cv, finalfit_train_bundle
from src.utils import now_run_id, set_seed
from src.config import RANDOM_STATE

def main():
    parser = argparse.ArgumentParser(description="BCI Imagined Speech Decoding: EdgeRouterNet Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Training set directory (.mat files)")
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory for checkpoints and logs")
    parser.add_argument("--mode", type=str, choices=["cv", "final", "both"], default="both", help="cv: 5-fold CV, final: Full fit, both: Both")
    parser.add_argument("--tag", type=str, default="BCI_RUN", help="Experiment tag")
    parser.add_argument("--beta_arc", type=float, default=0.55, help="ArcFace weight factor")
    args = parser.parse_args()
    
    set_seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.out_dir, f"{now_run_id()}_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    
    ab = {"tag": args.tag, "disable_csa": False, "disable_eca": False, "disable_ms2": False, "beta_arc": args.beta_arc}
    
    print(f"--- BCI Imagined Speech Pipeline | Mode: {args.mode} | Device: {device} ---")
    X, y, _ = load_split_full(args.data_dir, "TRAIN", L_need=1152)
    if X is None: return print("Error: No data found.")

    if args.mode in ["cv", "both"]:
        res = run_5fold_cv(ab, X, y, run_dir, device)
        print(f"\n[CV COMPLETE] Mean Acc: {res['acc_mean']:.4f} \u00b1 {res['acc_std']:.4f}")

    if args.mode in ["final", "both"]:
        bp = finalfit_train_bundle(ab, X, y, run_dir, device)
        print(f"\n[FINALFIT COMPLETE] Bundle: {bp}")

if __name__ == "__main__": main()
