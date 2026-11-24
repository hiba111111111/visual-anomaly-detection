import nbformat as nbf, sys
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = [
md("""# 03 — Extraction de caractéristiques
Module `src/features.py`, appliqué aux images **prétraitées** (configuration du notebook 02).

| Famille | Contenu | Cours |
|---|---|---|
| **histogramme** (par zone) | histogramme normalisé 16 intervalles | Ch. 2 |
| **statistiques** (par zone) | moyenne, écart-type, contraste (max−min)/(max+min) | Ch. 2 |
| **sobel** (par zone) | énergie du gradient (passe-haut) | Ch. 3 |
| **résidu** (par zone, objets) | \\|I − I_ref\\| moyen et maximal | Ch. 2 (soustraction d'images) |
| **ORB / SIFT sac de mots** (global) | histogramme de 64 mots visuels (K-Means) | Ch. 1 |
| **ORB / SIFT quantification** (global) | distance des descripteurs à leur mot le plus proche | Ch. 1 |
| **ORB / SIFT appariement** (global) | bons appariements avec la référence (test de ratio) | Ch. 1 |

Zones : 32×32 pixels, pas de 16 → **225 zones** qui se chevauchent (hazelnut : 16×16, pas de 8 → **961 zones**, choisi à l'étape 6).  
Prétraitement final (étape 8, choisi par le critère sans étiquettes) : bottle et hazelnut sans normalisation photométrique ; carpet en `localnorm` et sans filtre médian."""),
code("""import sys, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.data_loader import load_config
from src.features import compute_features

cfg = load_config(ROOT / "config.yaml"); CATS = cfg["data"]["categories"]
OUT = ROOT / "results" / "03_caracteristiques"; OUT.mkdir(parents=True, exist_ok=True)
F, T = {}, {}
for c in CATS:
    t0 = time.time(); F[c] = compute_features(c, cfg); T[c] = time.time() - t0
print({c: f"{t:.0f} s" for c, t in T.items()})"""),
code("""rows = []
for c in CATS:
    d, fx = F[c], F[c]["fx"]
    rows.append({"catégorie": c, "zones / image": d["train"]["zones"].shape[1],
                 "caract. / zone": d["train"]["zones"].shape[2], "caract. globales": d["train"]["global"].shape[1],
                 "familles par zone": ", ".join(fx.zone_families()), "temps total (s)": round(T[c])})
pd.DataFrame(rows).set_index("catégorie")"""),

md("## 1. Visualisation des caractéristiques par zone sur des images défectueuses"),
code("""def zmap(v):
    n = int(round(np.sqrt(len(v)))); return v.reshape(n, n)   # 15x15 (zones 32 px) ou 31x31 (16 px)
for c in CATS:
    d = F[c]; te = d["test"]; fx = d["fx"]; names = fx.zone_names()
    i = int(np.argmax(te["zone_truth"].max(1) * (te["labels"] == 1)))   # défaut le plus visible
    Z = te["zones"][i]; Ztr = d["train"]["zones"]
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-3
    zsc = ((Z - mu) / sd) ** 2
    panels = [("image prétraitée", te["images"][i], "gray"), ("référence", d["pre"].reference, "gray"),
              ("vérité terrain (zones)", zmap(te["zone_truth"][i]), "Reds"),
              ("écart-type zone (z²)", zmap(zsc[:, names.index("ecart_type")]), "magma"),
              ("Sobel (z²)", zmap(zsc[:, names.index("sobel")]), "magma"),
              ("histogramme (z² moyen)", zmap(zsc[:, :fx.hist_bins].mean(1)), "magma")]
    if "residu_moy" in names:
        panels.append(("|I − I_ref| moyen (z²)", zmap(zsc[:, names.index("residu_moy")]), "magma"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3))
    for ax, (t, im, cm) in zip(axes, panels):
        ax.imshow(im, cmap=cm); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle(f"{c} — {te['defects'][i]}", fontweight="bold"); plt.tight_layout()
    plt.savefig(OUT / f"zones_{c}.png", dpi=110); plt.show()"""),

md("""## 2. Points ORB et SIFT, colorés par erreur de quantification
Rouge = descripteur **loin de tous les mots** du vocabulaire appris sur les images saines (motif inhabituel)."""),
code("""from src.features import _as_float
for c in CATS:
    d = F[c]; te = d["test"]; fx = d["fx"]
    i = int(np.argmax(te["zone_truth"].max(1) * (te["labels"] == 1)))
    img, mask = te["images"][i], te["masks"][i]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, kind in zip(axes, fx.descriptors):
        bv = fx.bovw[kind]; kp, des = bv.describe(img)
        X = _as_float(des, kind); q = np.linalg.norm(X - bv.kmeans.cluster_centers_[bv.kmeans.predict(X)], axis=1)
        pts = np.array([k.pt for k in kp])
        ax.imshow(img, cmap="gray"); ax.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=q, cmap="coolwarm", s=12, vmin=np.percentile(q, 5), vmax=np.percentile(q, 95))
        ax.set_title(f"{kind.upper()} : {len(kp)} points (contour vert = défaut)"); ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="erreur de quantification")
    fig.suptitle(f"{c} — {te['defects'][i]}", fontweight="bold"); plt.tight_layout()
    plt.savefig(OUT / f"orb_sift_{c}.png", dpi=110); plt.show()"""),

md("""## 3. Pouvoir discriminant de chaque famille, **sans modèle** (référence pour l'étape 6)
Test volontairement simple : on standardise chaque caractéristique avec la moyenne et l'écart-type des images saines d'**apprentissage**, puis :
- **par zone** : score de zone = moyenne des z² de la famille ; score d'image = **maximum** sur les 225 zones ;
  - *par position* : moyenne et écart-type calculés **pour chaque position** de zone (suppose un alignement parfait) ;
  - *partagé* : statistiques **communes à toutes les zones** (insensible à la position) ;
- **global** : score d'image = moyenne des z².

On mesure l'AUROC sur le test (0,5 = hasard, 1 = séparation parfaite). Les modèles OC-SVM et Isolation Forest de l'étape 6 devront faire **mieux** que cette base."""),
code("""def auroc_zone(d, cols, mode):
    Ztr, Zte = d["train"]["zones"][:, :, cols], d["test"]["zones"][:, :, cols]
    if mode == "par position":
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-3
    else:
        flat = Ztr.reshape(-1, len(cols)); mu, sd = flat.mean(0), flat.std(0) + 1e-3
    zone_score = (((Zte - mu) / sd) ** 2).mean(-1)
    img = roc_auc_score(d["test"]["labels"], zone_score.max(1))
    zt = d["test"]["zone_truth"]; keep = (zt == 0) | (zt > 0.1)       # zones nettement saines / défectueuses
    loc = roc_auc_score((zt[keep] > 0.1).astype(int), zone_score[keep])
    return img, loc

def auroc_global(d, cols):
    Gtr, Gte = d["train"]["global"][:, cols], d["test"]["global"][:, cols]
    mu, sd = Gtr.mean(0), Gtr.std(0) + 1e-6
    return roc_auc_score(d["test"]["labels"], (((Gte - mu) / sd) ** 2).mean(1))

rows = []
for c in CATS:
    d, fx = F[c], F[c]["fx"]
    fams = dict(fx.zone_families()); fams["toutes (zones)"] = list(range(d["train"]["zones"].shape[2]))
    for fam, cols in fams.items():
        for mode in ("par position", "partagé"):
            a_img, a_loc = auroc_zone(d, cols, mode)
            rows.append({"catégorie": c, "famille": fam, "type": f"zone, {mode}", "AUROC image": a_img, "AUROC zone": a_loc})
    gf = dict(fx.global_families()); gf["toutes (globales)"] = list(range(d["train"]["global"].shape[1]))
    for fam, cols in gf.items():
        rows.append({"catégorie": c, "famille": fam, "type": "global", "AUROC image": auroc_global(d, cols), "AUROC zone": np.nan})
df_auc = pd.DataFrame(rows)
tab = df_auc.pivot_table(index=["type", "famille"], columns="catégorie", values="AUROC image").round(3)[CATS]
tab.to_csv(OUT / "auroc_sans_modele.csv"); tab"""),
code("""tab_loc = df_auc.dropna().pivot_table(index=["type", "famille"], columns="catégorie", values="AUROC zone").round(3)[CATS]
tab_loc.to_csv(OUT / "auroc_localisation_sans_modele.csv"); tab_loc"""),
code("""fig, axes = plt.subplots(1, len(CATS), figsize=(6 * len(CATS), 5), sharex=True)
for ax, c in zip(axes, CATS):
    sub = df_auc[df_auc["catégorie"] == c].copy(); sub["nom"] = sub["famille"] + " (" + sub["type"] + ")"
    sub = sub.sort_values("AUROC image")
    colors = ["tab:blue" if "global" in t else ("tab:orange" if "partagé" in t else "tab:green") for t in sub["type"]]
    ax.barh(sub["nom"], sub["AUROC image"], color=colors); ax.axvline(0.5, color="k", ls="--")
    ax.set_xlim(0.3, 1); ax.set_title(c); ax.tick_params(axis="y", labelsize=8)
axes[0].set_xlabel("AUROC image (sans modèle)")
plt.tight_layout(); plt.savefig(OUT / "auroc_sans_modele.png", dpi=110); plt.show()"""),
md("## Conclusions (configuration finale de l'étape 6)\n\n**AUROC image sans modèle : meilleure famille par catégorie** (choisie *a posteriori* sur le test, donc à titre indicatif)\n\n| | meilleure famille | AUROC | remarque |\n|---|---|---|---|\n| **bottle** | résidu \\|I−I_ref\\|, par position | **0,988** | toutes les familles par zone dépassent 0,95 ; ORB/SIFT global entre 0,78 et 0,94 |\n| **hazelnut** (zones 16 px) | Sobel, **partagé** | 0,880 | le résidu par position ne vaut que 0,53, contre 0,73 en partagé |\n| **carpet** (`localnorm`) | histogramme, par position | 0,868 | 0,681 avec `minmax` : la correction d'éclairage aide nettement ; ORB/SIFT global toujours **inversé** (0,32 à 0,61) |\n\n**Constats :**\n1. **bottle** est bien aligné : comparer chaque zone à la même zone des images saines (par position) fonctionne presque parfaitement. Le résidu (soustraction d'images, Ch. 2) est la meilleure caractéristique.\n2. **hazelnut** : comme l'alignement des noix est imparfait, les statistiques par position sont faussées. Le modèle **partagé** entre les zones, insensible à la position, est meilleur pour le Sobel et le résidu. Les détecteurs de l'étape 6 retiennent bien la stratégie partagée.\n3. **carpet** : la **normalisation locale de l'éclairage** (étape 6) fait passer la meilleure famille de 0,68 à 0,87. Les caractéristiques globales ORB/SIFT restent mauvaises sur cette texture ; elles ne sont pas retenues par les détecteurs.\n4. La **localisation** fonctionne : l'AUROC par zone (zone défectueuse ou saine) va de 0,70 à 0,89.\n5. Ces valeurs choisissent la meilleure famille **en regardant le test**. Elles servent de repère et ne sont pas comparables directement aux détecteurs de l'étape 6, qui sont choisis **sans** vrai défaut."),
]
# conclusions : fichier tools/conclusions/XX.md (les conclusions intégrées plus haut sont ignorées)
from pathlib import Path as _P
cells = [c for c in cells if not (c.cell_type == "markdown" and "Conclusions" in c.source.lstrip().splitlines()[0])]
_f = _P(__file__).with_name("conclusions") / (_P(__file__).stem[-2:] + ".md")
if _f.exists():
    cells.append(md(_f.read_text(encoding="utf-8")))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}})
nbf.write(nb, sys.argv[1])
