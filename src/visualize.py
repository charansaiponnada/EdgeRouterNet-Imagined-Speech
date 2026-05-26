import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import EdgeRouterNet, ArcMarginProduct
from src.dataset import load_one_mat_full
from src.utils import generate_saliency_map, set_seed
from src.config import class_names

def main():
    parser = argparse.ArgumentParser(description="Visualize EEG Saliency Maps")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to a sample .mat file")
    parser.add_argument("--out_dir", type=str, default="./visualizations", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model = EdgeRouterNet().to(device)
    model.load_state_dict(ckpt["model"])
    arc = ArcMarginProduct(in_features=model.emb_dim, out_features=5).to(device)
    arc.load_state_dict(ckpt["arc"])

    # Load sample data
    X, y = load_one_mat_full(args.data_file, L_need=1152)
    sample_idx = 0
    x_ct = X[sample_idx, :, :1024]
    target = int(y[sample_idx])

    print(f"Generating saliency map for class: {class_names[target]}")
    saliency = generate_saliency_map(model, arc, x_ct, target)

    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Time-averaged channel importance
    channel_importance = saliency.mean(axis=1)
    
    plt.subplot(2, 1, 1)
    plt.bar(range(len(channel_importance)), channel_importance)
    plt.title(f"Channel Importance (Saliency) - Target: {class_names[target]}")
    plt.xlabel("Channel Index")
    plt.ylabel("Importance (Mean Grad Abs)")

    # Heatmap
    plt.subplot(2, 1, 2)
    sns.heatmap(saliency[:, ::4], cmap="viridis", cbar_kws={'label': 'Saliency'})
    plt.title("Saliency Heatmap (Channel vs Time)")
    plt.xlabel("Time (subsampled)")
    plt.ylabel("Channel")

    plt.tight_layout()
    save_path = os.path.join(args.out_dir, f"saliency_{class_names[target]}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()
