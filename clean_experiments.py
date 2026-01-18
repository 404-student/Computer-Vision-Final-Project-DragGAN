import os
import json
import numpy as np
exp_root = "experiments"
md_list = []
fid_list = []

for name in sorted(os.listdir(exp_root)):
    exp_dir = os.path.join(exp_root, name)
    if not os.path.isdir(exp_dir):
        continue

    metrics_path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    md_list.append(metrics["mean_tracking_error"])
    fid_list.append(metrics["pseudo_fid"])

md = np.array(md_list)
fid = np.array(fid_list)

print(f"MD mean:  {md.mean():.4f} ± {md.std():.4f}")
print(f"FID mean: {fid.mean():.4f} ± {fid.std():.4f}")