"""
ds001907_fix_scores.py
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--demo",   required=True)
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, index_col=0)
    scores = scores[["sub", "motion"]].drop_duplicates(subset="sub").reset_index(drop=True)
    print(f"Scores : {len(scores)} sujets")
    print(f"  min={scores['motion'].min():.3f}  max={scores['motion'].max():.3f}  mean={scores['motion'].mean():.3f}")

    demo = pd.read_csv(args.demo)
    demo = demo.sort_values("agevisit").drop_duplicates(subset="subject_id", keep="first")
    demo["sub"] = "sub-" + demo["subject_id"].astype(str)
    demo["age"] = pd.to_numeric(demo["agevisit"], errors="coerce")
    demo["sex"] = demo["gender"].astype(str)
    demo["sex_num"] = (demo["gender"].astype(str).str.lower() == "male").astype(int)
    demo["group"] = demo["Group"].astype(str)
    demo["group_num"] = (demo["Group"].astype(str).str.lower().str.contains("patient")).astype(int)

    merged = scores.merge(demo[["sub","age","sex","sex_num","group","group_num"]], on="sub", how="left")
    print(f"\nGroupes :")
    print(merged.groupby("group")["motion"].agg(["count","mean","std"]).round(3))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nSauvegardé : {args.out}")

if __name__ == "__main__":
    main()
