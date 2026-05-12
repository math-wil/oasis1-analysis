# motion-analysis — Contexte projet

## Qui
Mathilde Wilfart, M.Sc. recherche, labo Neuro-iX, ÉTS Montréal.
Directeur : Sylvain Bouix.
Sujet : correction des artefacts de mouvement en IRM cérébrale structurelle T1w.

## Objectif
Qualifier le mouvement (Agitation), corriger (JDAC), valider sur mesures morphométriques
(épaisseur corticale par région APARC, volume GM). Comparer sur plusieurs datasets.

## Environnements

### Laptop personnel (Windows) — écriture de code uniquement
```
Repo local : C:\Users\Mathilde\Documents\GitHub\motion-analysis
Pas d'accès aux données IRM — scripts seulement
```

### PC du labo — exécution des analyses
```
User      : av62870@ens.ad.etsmtl.ca@ETS053232L
Conda env : cortical-motion
Repo      : ~/Documents/motion-analysis

Données sur Hippocampus :
  OASIS1 BIDS raw    : /project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids
  OASIS1 preprocMNI  : /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni
  OASIS1 FreeSurfer  : /project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1
  ds001907           : à télécharger (pas encore sur Hippocampus)

Narval : username mathw, auth Duo Mobile
```

## Structure du repo
```
motion-analysis/
├── pipelines/
│   ├── pipeline_agitation.py    Wrapper agitation CLI, gère BIDS et non-BIDS
│   ├── pipeline_glm_aparc.py    GLM par région APARC, ΔAIC, sélection JDAC
│   └── pipeline_jdac.py         (à créer — fichier source sur PC du labo)
├── datasets/
│   ├── oasis1/
│   │   ├── scripts/
│   │   │   ├── oasis1_agitation.py
│   │   │   ├── oasis1_glm_mean.py
│   │   │   └── oasis1_glm_aparc.py
│   │   ├── results_raw/
│   │   │   ├── oasis1_scores_raw.csv
│   │   │   ├── oasis1_glm_aparc.csv
│   │   │   └── oasis1_glm_mean.txt
│   │   ├── results_mni/
│   │   │   ├── oasis1_scores_mni.csv       (existe en local labo, pas encore pushé)
│   │   │   ├── oasis1_jdac_subjects.csv    25 sujets stratifiés low/mid/high
│   │   │   └── oasis1_glm_aparc_mni.csv    (à générer)
│   │   ├── figures/
│   │   │   ├── oasis1_aic_delta_raw.png
│   │   │   └── oasis1_motion_thickness_raw.png
│   │   └── metadata/
│   │       ├── oasis1_demographics.xlsx
│   │       └── participants.tsv
│   └── ds001907/
│       ├── scripts/
│       ├── results/
│       ├── figures/
│       └── metadata/
└── CLAUDE.md
```

## Datasets

### OASIS1
416 sujets (18-96 ans), 100 avec Alzheimer. Scores Agitation calculés sur
preprocessed_mni (plus fiables que raw). JDAC déjà lancé sur 25 sujets
stratifiés. FreeSurfer 7.4.1 dispo sur Hippocampus.

### ds001907
Attention Network Test — Parkinson + sujets âgés sains. T1w + fMRI.
Pas encore sur Hippocampus. Pertinent pour comparer profils de mouvement
Parkinson (tremblements) vs OASIS1 (population générale/Alzheimer).

## Commandes types (PC du labo)

```bash
# Agitation sur preprocessed_mni
python pipelines/pipeline_agitation.py \
    --bids_root /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni \
    --out       datasets/oasis1/results_mni/oasis1_scores_mni.csv \
    --pattern   "{sub}/{sub}_t1w_mni.nii.gz"

# GLM APARC OASIS1 (scores MNI)
python pipelines/pipeline_glm_aparc.py \
    --dataset      oasis1 \
    --scores       datasets/oasis1/results_mni/oasis1_scores_mni.csv \
    --fs_dir       /project/hippocampus/common/datasets/OASIS1_BIDS/processed_freesurfer7.4.1 \
    --participants /project/hippocampus/common/datasets/OASIS1_BIDS/raw_data_bids/participants.tsv \
    --bids_root    /project/hippocampus/common/datasets/OASIS1_BIDS/preprocessed_mni \
    --out_dir      datasets/oasis1/results_mni \
    --id_format    oasis1

# GLM APARC ds001907 (une fois téléchargé)
python pipelines/pipeline_glm_aparc.py \
    --dataset      ds001907 \
    --scores       datasets/ds001907/results/ds001907_scores.csv \
    --fs_dir       /project/hippocampus/common/datasets/ds001907/derivatives/freesurfer \
    --participants /project/hippocampus/common/datasets/ds001907/participants.tsv \
    --bids_root    /project/hippocampus/common/datasets/ds001907 \
    --out_dir      datasets/ds001907/results

# Télécharger ds001907
cd /project/hippocampus/common/datasets
datalad install https://github.com/OpenNeuroDatasets/ds001907
# ou : aws s3 sync s3://openneuro.org/ds001907 ds001907 --no-sign-request
```

## Conventions de nommage
`{dataset}_{contenu}_{variante}.ext`
Exemples : oasis1_scores_mni.csv, oasis1_aic_delta_raw.png, ds001907_glm_aparc.csv

## Prochaines étapes
1. Pusher oasis1_scores_mni.csv depuis le labo
2. Relancer pipeline_glm_aparc.py sur scores MNI (pas encore fait avec la nouvelle structure)
3. Télécharger ds001907 + lancer Agitation + GLM APARC
4. Créer pipeline_jdac.py à partir du fichier jdac_infer.py existant sur le labo
5. Comparer ΔAIC OASIS1 vs ds001907
