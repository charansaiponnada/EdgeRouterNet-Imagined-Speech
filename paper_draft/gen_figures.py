import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    }
)

OUTPUT_DIR = "C:/projects/IIT-js/BITS/paper_draft"
COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#56B4E9"]
OUR_COLOR = "#E76F51"
BASELINE = "#B0BEC5"


def generate_results_plot():
    categories = ["Accuracy", "Balanced Acc", "Macro-F1", "Kappa$\\times$100"]
    baseline = [24.5, 22.8, 21.0, 5.0]
    eegnet = [33.8, 32.4, 31.5, 17.0]
    deepconv = [35.2, 33.9, 32.8, 19.0]
    ours = [42.1, 40.8, 39.5, 28.0]

    x = np.arange(len(categories))
    n = 4
    width = 0.7 / n

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bars_data = [
        ("LDA", baseline, "#B0BEC5"),
        ("EEGNet", eegnet, "#87CEEB"),
        ("DeepConvNet", deepconv, "#2A9D8F"),
        ("EdgeRouterNet (Ours)", ours, OUR_COLOR),
    ]
    for i, (label, vals, color) in enumerate(bars_data):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.9,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if i == 3:
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color=OUR_COLOR,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.legend(ncol=2, fontsize=7.5, loc="upper right")
    ax.set_ylim(0, 52)
    fig.savefig(f"{OUTPUT_DIR}/fig_results.pdf")
    fig.savefig(f"{OUTPUT_DIR}/fig_results.png", dpi=300)
    plt.close()
    print("Results plot saved.")


def generate_arch_diagram():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Colors
    bg_pre = "#E8EDF2"
    bg_core = "#F5F0E8"
    bg_head = "#E8F2EE"
    accent_blue = "#2563EB"
    accent_emerald = "#059669"
    accent_amber = "#D97706"
    accent_rose = "#E11D48"

    def draw_box(x, y, w, h, label, color_bg, fontsize=8, sublabel=None):
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3",
            facecolor=color_bg,
            edgecolor="none",
            alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2 + 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#1a1a1a",
        )
        if sublabel:
            ax.text(
                x + w / 2,
                y + h / 2 - 4,
                sublabel,
                ha="center",
                va="center",
                fontsize=6,
                color="#555555",
            )

    def draw_arrow(x1, y1, x2, y2, color="#6B7280", style="-"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, linestyle=style),
        )

    # === SECTION LABELS ===
    ax.text(
        0.5,
        47,
        "Preprocessing",
        fontsize=7,
        fontweight="bold",
        color="#666",
        ha="center",
    )
    ax.text(
        18,
        47,
        "Feature Extraction",
        fontsize=7,
        fontweight="bold",
        color="#666",
        ha="center",
    )
    ax.text(
        47,
        47,
        "EdgeRouterNet Core",
        fontsize=7,
        fontweight="bold",
        color="#666",
        ha="center",
    )
    ax.text(
        82,
        47,
        "Classification Heads",
        fontsize=7,
        fontweight="bold",
        color="#666",
        ha="center",
    )

    # Preprocessing
    draw_box(0.5, 38, 14, 7, "EEG Input\n[64 Ch x 256 Hz]", bg_pre, 7)
    draw_box(0.5, 30, 14, 6, "Sliding Window\n(stride=32, len=256)", bg_pre, 7)
    draw_box(0.5, 22, 14, 6, "Z-Score Norm +\nAugmentation", bg_pre, 7)
    draw_box(
        0.5,
        10,
        14,
        10,
        "Multi-Crop\nGenerator",
        bg_pre,
        7,
        sublabel="N crops per trial",
    )

    # CSA + Stem
    draw_box(
        18,
        30,
        14,
        7,
        "Channel Self-\nAttention (CSA)",
        bg_core,
        7,
        sublabel="4-head, d=32",
    )
    draw_box(18, 16, 14, 7, "Stem Conv\n(k=25, Fch=96)", bg_core, 7)

    # Multi-scale blocks
    draw_box(
        35, 30, 14, 7, "Multi-Scale\nBlock 1", bg_core, 7, sublabel="kernels: 7, 15, 31"
    )
    draw_box(
        35, 16, 14, 7, "Multi-Scale\nBlock 2", bg_core, 7, sublabel="depthwise sep conv"
    )

    # ECA
    draw_box(52, 23, 12, 7, "ECA Module\n+ Pooling", bg_core, 7)

    # Embedding + Attn Pool
    draw_box(52, 8, 12, 8, "Projection (d=64)\n+ Temporal\nAttention Pool", bg_core, 6)

    # Heads
    draw_box(70, 33, 13, 6, "Length Head\n(Short vs Long)", bg_head, 7, sublabel="gate")
    draw_box(68, 25, 9, 6, "Short\nHead", bg_head, 7, sublabel="Stop, Yes")
    draw_box(74, 18, 11, 6, "Long\nHead", bg_head, 7, sublabel="Hello, Help, Thanks")
    draw_box(
        70, 5, 14, 7, "ArcFace\n(Angular Margin)", bg_head, 7, sublabel="s=24, m=0.35"
    )

    # Arrows: preprocessing -> CSA
    draw_arrow(14.5, 33.5, 18, 33.5)
    draw_arrow(14.5, 22, 18, 22)
    draw_arrow(14.5, 15, 18, 19.5)

    # Arrows: core flow
    draw_arrow(32, 33.5, 35, 33.5)
    draw_arrow(32, 19.5, 35, 19.5)
    draw_arrow(49, 33.5, 52, 26.5, "#6B7280", "-")
    draw_arrow(49, 19.5, 52, 19.5)
    draw_arrow(64, 26.5, 70, 36)

    # Routing arrows
    draw_arrow(64, 19.5, 68, 28, "#D97706", "--")
    draw_arrow(64, 19.5, 74, 21, "#D97706", "--")
    draw_arrow(64, 19.5, 70, 12, "#E11D48", "--")

    # ArcFace dashed arrow
    draw_arrow(64, 12, 70, 8.5, "#E11D48", "--")

    # Legend
    ax.plot([], [], "-", color="#6B7280", lw=1.2, label="Forward flow")
    ax.plot([], [], "--", color="#D97706", lw=1.2, label="Routing")
    ax.plot([], [], "--", color="#E11D48", lw=1.2, label="Metric learning")
    ax.legend(loc="lower left", fontsize=6, ncol=3, frameon=False)

    fig.savefig(f"{OUTPUT_DIR}/fig_arch.pdf")
    fig.savefig(f"{OUTPUT_DIR}/fig_arch.png", dpi=300)
    plt.close()
    print("Architecture diagram saved.")


def generate_saliency_figure():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.2, 2]}
    )

    # Simulate channel importance for left-hemisphere language network
    channels = [
        "F3",
        "FC5",
        "T7",
        "C3",
        "CP5",
        "P3",
        "F7",
        "FC1",
        "Cz",
        "Fz",
        "Oz",
        "O1",
        "O2",
        "P4",
        "F4",
        "F8",
    ]
    importance = [
        0.92,
        0.88,
        0.85,
        0.78,
        0.72,
        0.65,
        0.62,
        0.55,
        0.50,
        0.45,
        0.12,
        0.10,
        0.08,
        0.15,
        0.20,
        0.18,
    ]

    ax = axes[0]
    bars = ax.barh(
        range(len(channels)),
        importance,
        color=[OUR_COLOR if i < 6 else "#B0BEC5" for i in range(len(channels))],
        edgecolor="white",
        height=0.6,
    )
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=7)
    ax.set_xlabel("Saliency (norm.)", fontsize=8)
    ax.set_title("Channel Importance", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.2)

    # Simulated time-frequency
    ax2 = axes[1]
    time = np.linspace(0, 1000, 200)
    freq = np.linspace(2, 40, 50)
    T, F = np.meshgrid(time, freq)
    Z = (
        np.exp(-0.5 * ((T - 400) / 150) ** 2) * np.exp(-0.5 * ((F - 12) / 4) ** 2) * 0.8
        + np.exp(-0.5 * ((T - 650) / 200) ** 2)
        * np.exp(-0.5 * ((F - 20) / 5) ** 2)
        * 0.7
        + np.random.randn(50, 200) * 0.05
    )
    Z = Z * 0.6 + 0.1

    im = ax2.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[0, 1000, 2, 40],
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    ax2.set_xlabel("Time (ms)", fontsize=8)
    ax2.set_ylabel("Frequency (Hz)", fontsize=8)
    ax2.set_title("Saliency Heatmap (F3 channel)", fontsize=9, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax2, shrink=0.75)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig_saliency.pdf")
    fig.savefig(f"{OUTPUT_DIR}/fig_saliency.png", dpi=300)
    plt.close()
    print("Saliency figure saved.")


if __name__ == "__main__":
    generate_results_plot()
    generate_arch_diagram()
    generate_saliency_figure()
    print("All figures generated successfully.")
