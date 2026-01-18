import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(exp_dir, out_name="loss_curve.png"):
    losses_path = os.path.join(exp_dir, "losses.npy")
    if not os.path.exists(losses_path):
        raise FileNotFoundError(f"losses.npy not found in {exp_dir}")

    losses = np.load(losses_path)

    if losses.ndim != 1:
        raise ValueError(f"Expected 1D losses array, got shape {losses.shape}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Drag Optimization Loss Curve")
    plt.grid(True)

    out_path = os.path.join(exp_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Loss curve saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to a single experiment directory (containing losses.npy)",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="loss_curve.png",
        help="Output image file name",
    )

    args = parser.parse_args()
    plot_loss_curve(args.exp_dir, args.out_name)
