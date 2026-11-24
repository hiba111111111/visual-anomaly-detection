import nbformat as nbf, sys
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = [
md("""# 06 — Évaluation finale et étude d'ablation
**Sujet 16 : Visual Anomaly Detection** — MVTec AD (`bottle`, `hazelnut`, `carpet`).

Rappel du pipeline et lien avec le cours :

| Étape du sujet | Réalisation | Cours |
|---|---|---|
| 1. Prétraitement | gris, 256×256 `INTER_AREA` ; **filtre médian 3×3** (objets) ; **normalisation locale** de l'éclairage (texture) ; **alignement** (centroïde Otsu, puis recherche de rotation). Variantes choisies par catégorie avec le critère sans étiquettes (section 4) | Ch. 2 (échantillonnage, histogrammes, transformations géométriques, interpolation) ; Ch. 3 (médian, passe-bas, régions) |
| 2. Caractéristiques | **histogrammes d'intensité par zone** (32 ou 16 px), moyenne, écart-type, contraste, Sobel, résidu \\|I−I_ref\\| ; **ORB / SIFT** en sac de mots visuels | Ch. 2 (histogrammes, soustraction d'images) ; Ch. 3 (Sobel) ; Ch. 1 (ORB, SIFT, Hamming) |
| 3. Apprentissage | **One-Class SVM** et **Isolation Forest** sur les seules images saines ; ACP ; validation croisée ; sélection par anomalies synthétiques | Ch. 1 (SVM, ACP) |
| 4. Reconnaissance | score > seuil → anomalie ; carte de chaleur ; contours, boîte, aire, centroïde | Ch. 3 (représentation des formes) |

Métriques : **AUROC** (mesure principale, indépendante du seuil), AP, précision, rappel, F1 au seuil du détecteur, rappel à taux de fausses alarmes fixé ; **AUROC pixel** et **IoU** pour la localisation."""),
code("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from src.data_loader import load_config
from src.evaluation import evaluate_detector

cfg = load_config(ROOT / "config.yaml"); CATS = cfg["data"]["categories"]
OUT = ROOT / "results" / "06_evaluation"; OUT.mkdir(parents=True, exist_ok=True)
E = {c: evaluate_detector(c, cfg) for c in CATS}
for c in CATS: print(f"{c:9s} {E[c]['info']['zone']} | agrégation {E[c]['info']['agrégation']} | seuil {E[c]['threshold']:.2f}")"""),

md("## 1. Niveau image : détection"),
code("""cols = ["AUROC", "AP", "précision", "rappel", "F1", "exactitude", "taux fausses alarmes", "rappel @5% FA", "rappel @10% FA", "rappel @20% FA"]
img = pd.DataFrame({c: E[c]["image"] for c in CATS}).T
tab = img[cols].astype(float).round(3)
tab.insert(0, "défauts détectés", [f"{int(img.loc[c, 'VP'])}/{int(img.loc[c, 'VP'] + img.loc[c, 'FN'])}" for c in CATS])
tab.insert(1, "sains rejetés", [f"{int(img.loc[c, 'FP'])}/{int(img.loc[c, 'FP'] + img.loc[c, 'VN'])}" for c in CATS])
tab.to_csv(OUT / "tableau_image.csv"); tab"""),
code("""fig, axes = plt.subplots(1, 2 + len(CATS), figsize=(5 * (2 + len(CATS)), 4.3), gridspec_kw={"width_ratios": [1.3, 1.3] + [1] * len(CATS)})
for c in CATS:
    fpr, tpr, thr = roc_curve(E[c]["labels"], E[c]["scores"])
    axes[0].plot(fpr, tpr, label=f"{c} (AUROC {E[c]['image']['AUROC']:.3f})")
    m = E[c]["image"]; axes[0].scatter(m["taux fausses alarmes"], m["rappel"], s=60, zorder=3)
axes[0].plot([0, 1], [0, 1], "k:"); axes[0].set_xlabel("taux de fausses alarmes"); axes[0].set_ylabel("rappel (défauts détectés)")
axes[0].set_title("Courbes ROC (point = seuil du détecteur)"); axes[0].legend(fontsize=8)
for c in CATS:
    s, y = E[c]["scores"], E[c]["labels"]
    axes[1].scatter(np.full((y == 0).sum(), CATS.index(c) - 0.12) + np.random.default_rng(0).normal(0, 0.03, (y == 0).sum()), s[y == 0], s=8, c="tab:green")
    axes[1].scatter(np.full((y == 1).sum(), CATS.index(c) + 0.12) + np.random.default_rng(1).normal(0, 0.03, (y == 1).sum()), s[y == 1], s=8, c="tab:red")
    axes[1].hlines(E[c]["threshold"], CATS.index(c) - 0.35, CATS.index(c) + 0.35, colors="k")
axes[1].set_xticks(range(len(CATS))); axes[1].set_xticklabels(CATS); axes[1].set_ylim(-3, 12)
axes[1].set_title("Scores du test (vert = sain, rouge = défaut, trait = seuil)")
for ax, c in zip(axes[2:], CATS):
    cm = confusion_matrix(E[c]["labels"], (E[c]["scores"] > E[c]["threshold"]).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14, color="k" if v < cm.max() / 2 else "w")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["prédit sain", "prédit défaut"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["sain", "défaut"])
    ax.set_title(f"{c} — matrice de confusion")
plt.tight_layout(); plt.savefig(OUT / "roc_scores_confusion.png", dpi=110); plt.show()"""),

md("## 2. Par type de défaut"),
code("""dft = pd.DataFrame([{"catégorie": c, "défaut": d, "AUROC": v, "surface médiane %": 100 * np.median([m.mean() for m, x in zip(E[c]["masks"], E[c]["defects"]) if x == d])}
                    for c in CATS for d, v in E[c]["per_defect"].items()])
fig, ax = plt.subplots(figsize=(12, 4))
colors = {"bottle": "tab:blue", "hazelnut": "tab:orange", "carpet": "tab:green"}
ax.bar(dft["catégorie"] + "\\n" + dft["défaut"], dft["AUROC"], color=[colors[c] for c in dft["catégorie"]])
ax.axhline(0.5, color="k", ls=":"); ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUROC"); ax.tick_params(axis="x", labelsize=8)
for i, (a, s) in enumerate(zip(dft["AUROC"], dft["surface médiane %"])): ax.text(i, a + 0.01, f"{s:.1f}%", ha="center", fontsize=7)
ax.set_title("AUROC par type de défaut (au-dessus : surface médiane du défaut, % de l'image)")
plt.tight_layout(); plt.savefig(OUT / "auroc_par_defaut.png", dpi=110); plt.show()
dft.round(3).to_csv(OUT / "auroc_par_defaut.csv", index=False); dft.round(3)"""),

md("""## 3. Niveau pixel : localisation
Cartes d'anomalie dans l'espace aligné (256×256), comparées aux masques vérité terrain recalés par la même transformation."""),
code("""pix = pd.DataFrame({c: E[c]["pixel"] for c in CATS}).T.astype(float).round(3); pix.to_csv(OUT / "tableau_pixel.csv"); pix"""),
code("""fig, axes = plt.subplots(len(CATS), 6, figsize=(18, 3.2 * len(CATS)))
for r, c in enumerate(CATS):
    e = E[c]; idx = [i for i in np.argsort(-e["scores"]) if e["labels"][i] == 1]
    picks = idx[:2] + idx[len(idx) // 2: len(idx) // 2 + 1]           # 2 plus forts + 1 médian
    for k, i in enumerate(picks):
        axes[r, 2 * k].imshow(e["images"][i], cmap="gray"); axes[r, 2 * k].contour(e["masks"][i], levels=[0.5], colors="lime")
        axes[r, 2 * k].set_title(f"{c} — {e['defects'][i]}", fontsize=9)
        axes[r, 2 * k + 1].imshow(np.clip(e["maps"][i], 0, 8), cmap="jet", vmin=0, vmax=8)
        axes[r, 2 * k + 1].contour(e["masks"][i], levels=[0.5], colors="white", linewidths=1); axes[r, 2 * k + 1].set_title("carte (z-score)", fontsize=9)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "localisation_exemples.png", dpi=100); plt.show()"""),

md("""## 4. Étude d'ablation : que rapporte chaque élément du pipeline ?
On retire ou remplace **un seul élément à la fois** par rapport au pipeline de l'étape 6 (la **référence**). La configuration du modèle reste fixe. Chaque variante est réentraînée avec le même protocole (validation croisée, anomalies synthétiques, seuil au 95e centile hors-pli) : **aucun vrai défaut** n'est utilisé pour l'entraîner. Δ = AUROC de la variante − AUROC de référence.

Cette étude sert **aussi à choisir le prétraitement final** : pour chaque catégorie, on retient la variante de prétraitement qui a le **meilleur critère synthétique** (section 4 bis). Les colonnes de test ne servent qu'au compte rendu."""),
code("""ab = pd.concat([pd.read_csv(OUT / f"ablation_{c}.csv") for c in CATS])
ref = ab[ab["variante"] == "complet (référence)"].set_index("catégorie")["AUROC test"]
ab["Δ AUROC"] = ab["AUROC test"] - ab["catégorie"].map(ref)
show = ab.pivot_table(index="variante", columns="catégorie", values="AUROC test").round(3)[CATS]
order = ["complet (référence)", "sans filtre médian", "sans normalisation photométrique", "normalisation globale minmax (au lieu de locale)",
         "sans alignement", "zones 32 px (au lieu de 16)"] + sorted(v for v in show.index if v.startswith("famille"))
show = show.reindex([o for o in order if o in show.index]); show.to_csv(OUT / "ablation_auroc.csv"); show"""),
code("""fig, axes = plt.subplots(1, len(CATS), figsize=(6 * len(CATS), 4.8), sharex=True)
for ax, c in zip(axes, CATS):
    d = ab[(ab["catégorie"] == c) & (ab["variante"] != "complet (référence)")].sort_values("Δ AUROC")
    ax.barh(d["variante"], d["Δ AUROC"], color=["tab:red" if v < 0 else "tab:green" for v in d["Δ AUROC"]])
    ax.axvline(0, color="k"); ax.set_title(f"{c} (référence AUROC {ref[c]:.3f})"); ax.tick_params(axis="y", labelsize=8)
    for y_, v in enumerate(d["Δ AUROC"]): ax.text(v, y_, f" {v:+.3f}", va="center", fontsize=8)
axes[0].set_xlabel("Δ AUROC test (variante − référence)")
plt.tight_layout(); plt.savefig(OUT / "ablation.png", dpi=110); plt.show()
ab.round(3).to_csv(OUT / "ablation_complete.csv", index=False)"""),
code("""det_rows = ab.pivot_table(index="variante", columns="catégorie", values=["F1", "taux fausses alarmes"]).round(3)
det_rows.reindex([o for o in order if o in det_rows.index])"""),

md("""## 4 bis. Choix du prétraitement final par le critère sans étiquettes
Variantes de **prétraitement** uniquement : les variantes « famille seule » portent sur les caractéristiques et sont déjà couvertes par la grille de sélection de l'étape 6. Le choix se fait sur la colonne *critère synthétique* ; l'AUROC de test est donnée pour information."""),
code("""pre_var = ["complet (référence)", "sans filtre médian", "sans normalisation photométrique",
           "normalisation globale minmax (au lieu de locale)", "sans alignement", "zones 32 px (au lieu de 16)"]
rows = []
for c in CATS:
    d = ab[(ab["catégorie"] == c) & ab["variante"].isin(pre_var)]
    best = d.loc[d["AUROC synthétique"].idxmax(), "variante"]
    for _, r in d.iterrows():
        rows.append({"catégorie": c, "variante": r["variante"], "critère synthétique": round(r["AUROC synthétique"], 3),
                     "AUROC test (rapport)": round(r["AUROC test"], 3), "retenue": "✔" if r["variante"] == best else ""})
choix = pd.DataFrame(rows); choix.to_csv(OUT / "choix_pretraitement.csv", index=False); choix"""),
md("""## 5. Comparaisons de modèles (rappel de l'étape 6)
Dans chaque groupe, la configuration choisie **par le critère synthétique**, et son AUROC de test."""),
code("""cmp_ = pd.read_csv(ROOT / "results" / "04_apprentissage" / "comparaisons.csv")
cmp_.pivot_table(index=["comparaison", "valeur"], columns="catégorie", values="AUROC test").round(3)[CATS]"""),
]
# conclusions : fichier tools/conclusions/XX.md (les conclusions intégrées plus haut sont ignorées)
from pathlib import Path as _P
cells = [c for c in cells if not (c.cell_type == "markdown" and "Conclusions" in c.source.lstrip().splitlines()[0])]
_f = _P(__file__).with_name("conclusions") / (_P(__file__).stem[-2:] + ".md")
if _f.exists():
    cells.append(md(_f.read_text(encoding="utf-8")))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}})
nbf.write(nb, sys.argv[1])
