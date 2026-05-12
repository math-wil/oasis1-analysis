import pandas as pd
import statsmodels.formula.api as smf
import os
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/home/av62870@ens.ad.etsmtl.ca/Documents/motion-analysis/datasets/oasis1")
MOTION_CSV   = BASE_DIR / "results_raw" / "oasis1_scores_raw.csv"
RESULTS_DIR  = BASE_DIR / "results_raw"
FREESURFER_DIR   = "/project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1"
PARTICIPANTS_TSV = "/project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids/participants.tsv"

# ── Scores de mouvement ───────────────────────────────────────────────────────
motion_df = pd.read_csv(MOTION_CSV, index_col=0)[["sub", "motion"]]

# ── Épaisseur corticale ───────────────────────────────────────────────────────
def get_mean_thickness(stats_file):
    with open(stats_file, "r") as f:
        for line in f:
            if "MeanThickness" in line and "Measure" in line:
                parts = line.strip().split(",")
                return float(parts[-2].strip())
    return None

thickness_data = []
for sub in os.listdir(FREESURFER_DIR):
    if not sub.startswith("sub-"):
        continue
    lh_stats = os.path.join(FREESURFER_DIR, sub, "stats", "lh.aparc.stats")
    if not os.path.exists(lh_stats):
        continue
    thickness = get_mean_thickness(lh_stats)
    if thickness is not None:
        thickness_data.append({"sub": sub, "lh_mean_thickness": thickness})

thickness_df = pd.DataFrame(thickness_data)

# ── Données démographiques ────────────────────────────────────────────────────
participants_df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
participants_df["sub"] = participants_df["participant_id"].str.extract(r"sub-OASIS1(\d+)").apply(
    lambda x: f"sub-{x[0]}", axis=1
)
participants_df = participants_df[["sub", "sex", "age_bl"]].rename(columns={"age_bl": "age"})

# ── Fusion ────────────────────────────────────────────────────────────────────
df = motion_df.merge(thickness_df, on="sub", how="inner")
df = df.merge(participants_df, on="sub", how="inner")
print(f"Sujets retenus : {len(df)}")

# ── GLM ───────────────────────────────────────────────────────────────────────
est = smf.glm("lh_mean_thickness ~ age + sex + motion", data=df)
result = est.fit()
print(result.summary())

out_file = RESULTS_DIR / "oasis1_glm_mean.txt"
with open(out_file, "w") as f:
    f.write(result.summary().as_text())
print(f"Résultats sauvegardés dans {out_file}")