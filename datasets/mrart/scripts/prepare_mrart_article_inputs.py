from pathlib import Path
import shutil
import subprocess
import numpy as np
import nibabel as nib

RAW = Path.home() / "Documents/Datasets/MRART"
SKULL = Path.home() / "Documents/Datasets/MRART_skullstrip"
READY = Path.home() / "Documents/Datasets/MRART_jdac_ready"

subjects = [
    "sub-000103",
    "sub-000148",
    "sub-000149",
    "sub-000159",
    "sub-000175",
    "sub-862915",
]

conditions = ["standard", "headmotion1", "headmotion2"]

synthstrip = shutil.which("mri_synthstrip")
bet = shutil.which("bet")

if synthstrip:
    skull_tool = "synthstrip"
elif bet:
    skull_tool = "bet"
else:
    raise SystemExit(
        "Aucun outil de skull stripping trouvé. Il faut soit mri_synthstrip, soit FSL bet."
    )

print(f"Skull stripping tool: {skull_tool}")

def skullstrip(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if skull_tool == "synthstrip":
        cmd = [synthstrip, "-i", str(src), "-o", str(dst)]
    else:
        cmd = [bet, str(src), str(dst), "-R", "-f", "0.25", "-g", "0"]

    subprocess.run(cmd, check=True)

def normalize_01(src: Path, dst: Path):
    img = nib.load(str(src))
    data = img.get_fdata().astype(np.float32)

    brain = data[data > 0]
    if brain.size == 0:
        raise RuntimeError(f"No brain voxels found after skull stripping: {src}")

    lo = np.percentile(brain, 0)
    hi = np.percentile(brain, 98)

    if hi <= lo:
        raise RuntimeError(f"Bad intensity range for {src}: lo={lo}, hi={hi}")

    out = (data - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[data <= 0] = 0.0
    out = out.astype(np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out, img.affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(dst))

    print(f"{src.name} -> {dst.name} | shape={out.shape} | min={out.min():.3f}, max={out.max():.3f}")

for sub in subjects:
    for cond in conditions:
        src = RAW / sub / "anat" / f"{sub}_acq-{cond}_T1w.nii.gz"
        if not src.exists():
            print(f"Missing: {src}")
            continue

        skull = SKULL / sub / "anat" / f"{sub}_acq-{cond}_T1w_brain.nii.gz"
        ready = READY / sub / "anat" / f"{sub}_acq-{cond}_T1w_brain_norm01.nii.gz"

        skullstrip(src, skull)
        normalize_01(skull, ready)

print("\nDone.")
print("Skull-stripped:", SKULL)
print("JDAC-ready:", READY)
