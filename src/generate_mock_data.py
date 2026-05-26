import os
import argparse
import numpy as np
from scipy.io import savemat

def generate_mock_mat(fp, n_trials=50, n_ch=64, n_times=1500):
    # Mock KaraOne .mat structure
    # epo.x: [times, channels, trials]
    # epo.y: [classes, trials]
    # epo.t: [times, 1]
    
    x = np.random.randn(n_times, n_ch, n_trials).astype(np.float32)
    y = np.zeros((5, n_trials))
    labels = np.random.randint(0, 5, n_trials)
    for i, l in enumerate(labels):
        y[l, i] = 1
        
    t = np.linspace(-0.5, 5.0, n_times).reshape(-1, 1)
    
    epo = {
        "x": x,
        "y": y,
        "t": t
    }
    
    savemat(fp, {"epo": epo})

def main():
    parser = argparse.ArgumentParser(description="Generate mock KaraOne data")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_files", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n_files):
        fp = os.path.join(args.out_dir, f"mock_subject_{i:02d}.mat")
        generate_mock_mat(fp)
        print(f"Generated {fp}")

if __name__ == "__main__":
    main()
