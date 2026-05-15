from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

root = Path.home()
ready = root / "Documents/Datasets/MRART_jdac_ready"
outroot = root / "Documents/Datasets/MRART_jdac_ready_outputs_all6"

subjects = [
    "sub-000103",
    "sub-000148",
    "sub-000149",
    "sub-000159",
    "sub-000175",
    "sub-862915",
]

conditions = ["headmotion1", "headmotion2"]

def load(p):
    if not Path(p).exists():
        raise FileNotFoundError(p)
    return nib.load(str(p)).get_fdata().astype(np.float32)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def gradmag(x):
    gx, gy, gz = np.gradient(x)
    return np.sqrt(gx**2 + gy**2 + gz**2)

rows = []

for sub in subjects:
    standard_path = ready / sub / "anat" / f"{sub}_acq-standard_T1w_brain_norm01.nii.gz"
    standard = load(standard_path)
    standard_g = gradmag(standard)

    for cond in conditions:
        motion_path = ready / sub / "anat" / f"{sub}_acq-{cond}_T1w_brain_norm01.nii.gz"
        jdac_path = outroot / f"{sub}_{cond}" / f"{sub}_{cond}_T1w_jdac.nii.gz"

        motion = load(motion_path)
        jdac = load(jdac_path)

        motion_g = gradmag(motion)
        jdac_g = gradmag(jdac)

        row = {
            "subject": sub,
            "condition": cond,
            "image_rmse_before": rmse(motion, standard),
            "image_rmse_after": rmse(jdac, standard),
            "image_mae_before": mae(motion, standard),
            "image_mae_after": mae(jdac, standard),
            "grad_rmse_before": rmse(motion_g, standard_g),
            "grad_rmse_after": rmse(jdac_g, standard_g),
            "grad_mae_before": mae(motion_g, standard_g),
            "grad_mae_after": mae(jdac_g, standard_g),
        }

        for metric in ["image_rmse", "image_mae", "grad_rmse", "grad_mae"]:
            before = row[f"{metric}_before"]
            after = row[f"{metric}_after"]
            row[f"{metric}_change_pct"] = 100 * (after - before) / before

        rows.append(row)

df = pd.DataFrame(rows)

results_dir = root / "Documents/Datasets/MRART"
detail_csv = results_dir / "mrart_jdac_all6_detailed_metrics.csv"
summary_csv = results_dir / "mrart_jdac_all6_summary_metrics.csv"

df.to_csv(detail_csv, index=False)

summary = (
    df.groupby("condition")
    .agg({
        "image_rmse_before": "mean",
        "image_rmse_after": "mean",
        "image_rmse_change_pct": "mean",
        "image_mae_before": "mean",
        "image_mae_after": "mean",
        "image_mae_change_pct": "mean",
        "grad_rmse_before": "mean",
        "grad_rmse_after": "mean",
        "grad_rmse_change_pct": "mean",
        "grad_mae_before": "mean",
        "grad_mae_after": "mean",
        "grad_mae_change_pct": "mean",
    })
    .reset_index()
)

summary.to_csv(summary_csv, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

print("\n=== Detailed per-subject metrics ===")
print(df.round(6).to_string(index=False))

print("\n=== Mean summary by condition ===")
print(summary.round(6).to_string(index=False))

print(f"\nSaved detailed metrics: {detail_csv}")
print(f"Saved summary metrics:  {summary_csv}")
