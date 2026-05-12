#!/bin/bash
# ============================================================
# À lancer sur le PC du labo demain matin
# ============================================================
set -e

REPO=~/Documents/motion-analysis
OASIS_LOCAL=~/Documents/oasis1-analysis   # ancien dossier si toujours là

echo "=== 1. Pull du repo ==="
cd $REPO
git pull origin main

echo ""
echo "=== 2. Copie des fichiers locaux manquants vers la nouvelle structure ==="

# Scores MNI (générés sur le labo, jamais pushés)
MNI_SRC=~/Documents/oasis1-analysis/oasis1_motion_scores_mni.csv
MNI_DST=$REPO/datasets/oasis1/results_mni/oasis1_scores_mni.csv
if [ -f "$MNI_SRC" ] && [ ! -f "$MNI_DST" ]; then
    cp "$MNI_SRC" "$MNI_DST"
    echo "  oasis1_scores_mni.csv copié"
fi

# GLM aparc MNI (si généré)
GLM_MNI_SRC=~/Documents/oasis1-analysis/results_mni/glm_aparc_results.csv
GLM_MNI_DST=$REPO/datasets/oasis1/results_mni/oasis1_glm_aparc_mni.csv
if [ -f "$GLM_MNI_SRC" ] && [ ! -f "$GLM_MNI_DST" ]; then
    cp "$GLM_MNI_SRC" "$GLM_MNI_DST"
    echo "  oasis1_glm_aparc_mni.csv copié"
fi

# Figures MNI si elles existent
for f in ~/Documents/oasis1-analysis/results_mni/*.png; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    dst=$REPO/datasets/oasis1/figures/${fname%.png}_mni.png
    [ -f "$dst" ] || cp "$f" "$dst" && echo "  figure MNI copiée : $fname"
done

# Outputs JDAC (images corrigées — NE PAS committer, juste vérifier où ils sont)
echo ""
echo "  Vérification outputs JDAC :"
find ~/Documents -name "*jdac*" -o -name "*corrected*" 2>/dev/null | grep -v ".git" | head -20

echo ""
echo "=== 3. Ajout des nouveaux scripts ds001907 ==="
mkdir -p $REPO/datasets/ds001907/scripts
mkdir -p $REPO/datasets/ds001907/results
mkdir -p $REPO/datasets/ds001907/figures
mkdir -p $REPO/datasets/ds001907/metadata

# Script GLM pour ds001907
cp ~/Documents/motion-analysis/datasets/oasis1/scripts/oasis1_glm_aparc.py \
   $REPO/datasets/ds001907/scripts/ds001907_glm_aparc.py 2>/dev/null || true

echo ""
echo "=== 4. Push ==="
cd $REPO
git add -A
git status
git commit -m "add MNI results + ds001907 structure" || echo "  Rien à committer"
git push

echo ""
echo "=== 5. Téléchargement ds001907 si pas déjà fait ==="
DS_DIR=/project/hippocampus/common/datasets/ds001907
if [ ! -d "$DS_DIR" ]; then
    echo "  Dataset absent de Hippocampus — à télécharger via :"
    echo "  cd /project/hippocampus/common/datasets"
    echo "  aws s3 sync s3://openneuro.org/ds001907 ds001907 --no-sign-request"
    echo "  ou : datalad install https://github.com/OpenNeuroDatasets/ds001907"
else
    echo "  ds001907 déjà présent dans $DS_DIR"
    ls $DS_DIR
fi

echo ""
echo "=== 6. Lancer Agitation sur ds001907 ==="
echo "  Commande à lancer manuellement une fois le dataset vérifié :"
echo ""
echo "  agitation dataset \\"
echo "    -d $DS_DIR \\"
echo "    -o $REPO/datasets/ds001907/results/ds001907_scores.csv"