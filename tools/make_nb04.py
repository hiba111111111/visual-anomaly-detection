import nbformat as nbf, sys
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = [
md("""# 04 — Apprentissage : One-Class SVM et Isolation Forest (données saines uniquement)
Modules `src/models.py`, `src/training.py`, `src/synthetic.py`, script `train.py` (`python train.py --all`).

**Chaîne de chaque modèle :** `VarianceThreshold` → `StandardScaler` → `ACP (95 % de variance)` → `OneClassSVM (RBF)` ou `IsolationForest`

**Deux niveaux :**
- **zones** (225 par image) : un modèle **par position** ou un modèle **partagé** ; score d'image = **max** ou **moyenne des 3 zones** les plus anormales ;
- **global** (sac de mots ORB/SIFT) : ajouté seulement s'il améliore le critère de sélection.

### Protocole (version 2, après correction de deux problèmes de la version 1)
1. **Toutes les images saines** (train + val, par exemple 209 bouteilles) sont découpées en **5 plis**. Chaque image saine est notée par un modèle entraîné sur les 4 autres plis : c'est un score **hors-pli**, donc non biaisé.
2. **Anomalies synthétiques réalistes** (tache, rayure, fissure, copier-coller, de **1 à 5 %** de l'image) créées sur chaque image saine. Le critère de sélection est l'AUROC hors-pli des images saines contre les images dégradées. **Aucun vrai défaut n'est utilisé.**
3. Détecteur final = **moyenne des 5 modèles**. Calibration et **seuil = 95e centile** des scores hors-pli des images saines, ce qui vise **5 % de fausses alarmes**.
4. Le **test** (vrais défauts) ne sert qu'au compte rendu."""),
code("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from src.data_loader import load_config
from src.features import compute_features
from src.models import AnomalyDetector
from src.training import healthy_and_synthetic

cfg = load_config(ROOT / "config.yaml"); CATS = cfg["data"]["categories"]
RES = ROOT / "results" / "04_apprentissage"; RES.mkdir(parents=True, exist_ok=True)
F = {c: compute_features(c, cfg, verbose=False) for c in CATS}
DET = {c: AnomalyDetector.load(ROOT / "models" / f"{c}_detector.joblib") for c in CATS}
Z = {c: pd.read_csv(RES / f"selection_zones_{c}.csv") for c in CATS}
GL = {c: pd.read_csv(RES / f"selection_global_{c}.csv") for c in CATS}
SYN = {c: healthy_and_synthetic(F[c], c, cfg)["S"] for c in CATS}
for c in CATS:
    i = DET[c].info
    print(f"{c:9s} → {i['zone']} | {i['agrégation']} | global : {i['global']} | {i['images saines (calibration)']} images saines, {i['plis']} plis")"""),

md("## 1. Anomalies synthétiques (1 à 5 % de l'image) utilisées pour la sélection"),
code("""fig, axes = plt.subplots(len(CATS), 8, figsize=(20, 2.8 * len(CATS)))
for r, c in enumerate(CATS):
    s = SYN[c]; area = s["masks"].reshape(len(s["masks"]), -1).mean(1) * 100
    for j in range(8):
        ax = axes[r, j]; ax.imshow(s["images"][j], cmap="gray", vmin=0, vmax=255)
        ax.contour(s["masks"][j], levels=[0.5], colors="red", linewidths=1)
        ax.set_title(f"{c} — {s['kinds'][j]} ({area[j]:.1f} %)", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(RES / "anomalies_synthetiques.png", dpi=110); plt.show()
pd.DataFrame({c: pd.Series(SYN[c]["masks"].reshape(len(SYN[c]["masks"]), -1).mean(1) * 100).describe()[["count", "min", "50%", "max"]]
              for c in CATS}).round(2).rename(index={"count": "nb défauts", "min": "surface min %", "50%": "surface médiane %", "max": "surface max %"})"""),

md("""## 2. Problème 1 résolu ? Le critère de sélection distingue-t-il les configurations ?
Version 1 (défauts de 0,4 %, validation de 42 à 78 images) : sur hazelnut et carpet, **toutes** les configurations obtenaient un critère entre 0,50 et 0,59, et la zone la plus anormale ne tombait sur le défaut synthétique que **20 %** du temps. Le choix se faisait donc au hasard.

Pour être fiable, le critère doit (a) **s'étaler** entre les configurations, (b) **localiser** le défaut synthétique et (c) **classer** les configurations comme le ferait le test. Le point (c) est vérifié a posteriori, pour information."""),
code("""rows = []
v1 = {"bottle": (0.966, 0.94), "hazelnut": (0.545, 0.20), "carpet": (0.589, 0.21)}   # mesures de la v1 (notebook 04 v1, diagnostic)
for c in CATS:
    d = Z[c]; s = d["AUROC synthétique (sélection)"]
    rows.append({"catégorie": c,
                 "v1 : meilleur critère": v1[c][0], "v1 : zone max sur défaut": f"{v1[c][1]:.0%}",
                 "v2 : critère min-max": f"{s.min():.2f} – {s.max():.2f}",
                 "v2 : zone max sur défaut (config retenue)": f"{d.loc[s.idxmax(), 'zone max sur le défaut synth.']:.0%}",
                 "v2 : corrélation critère / test (Spearman)": round(d[["AUROC synthétique (sélection)", "AUROC test (rapport)"]].corr("spearman").iloc[0, 1], 2)})
crit = pd.DataFrame(rows).set_index("catégorie"); crit.to_csv(RES / "fiabilite_critere.csv"); crit"""),
code("""fig, axes = plt.subplots(1, len(CATS), figsize=(5.5 * len(CATS), 4.5))
for ax, c in zip(axes, CATS):
    d = Z[c]
    for (strat, mod), g in d.groupby(["stratégie", "modèle"]):
        ax.scatter(g["AUROC synthétique (sélection)"], g["AUROC test (rapport)"], label=f"{strat} / {mod}", alpha=0.7, s=25)
    b = d.loc[d["AUROC synthétique (sélection)"].idxmax()]
    ax.scatter(b["AUROC synthétique (sélection)"], b["AUROC test (rapport)"], marker="*", s=300, c="k", label="retenu")
    ax.set_title(c); ax.set_xlabel("critère : AUROC synthétique hors-pli"); ax.set_ylabel("AUROC test (rapport)")
    ax.plot([0.4, 1], [0.4, 1], "k:", lw=0.8); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(RES / "synthetique_vs_test.png", dpi=110); plt.show()"""),
code("""for c in CATS:
    t = Z[c].sort_values("AUROC synthétique (sélection)", ascending=False).head(8)
    cols = ["caractéristiques", "stratégie", "modèle", "paramètres", "agrégation", "AUROC synthétique (sélection)",
            "zone max sur le défaut synth.", "AUROC test (rapport)", "temps (s)"]
    print(f"=== {c} : 8 meilleures configurations selon le critère ==="); display(t[cols].round(3).reset_index(drop=True))"""),

md("""## 3. Comparaisons demandées par le sujet
Dans chaque groupe, on prend la configuration choisie **par le critère synthétique**, puis on affiche son AUROC de **test**."""),
code("""def pick(d):
    return d.loc[d["AUROC synthétique (sélection)"].idxmax()]
rows = []
for c in CATS:
    d = Z[c]
    for col in ["modèle", "stratégie", "caractéristiques"]:
        for v, g in d.groupby(col):
            b = pick(g)
            rows.append({"catégorie": c, "comparaison": col, "valeur": v, "AUROC test": b["AUROC test (rapport)"]})
    for v, g in GL[c].groupby("modèle"):
        rows.append({"catégorie": c, "comparaison": "global ORB/SIFT seul", "valeur": v, "AUROC test": pick(g)["AUROC test (rapport)"]})
    for k, v in DET[c].info["combinaisons"].items():
        rows.append({"catégorie": c, "comparaison": "combinaison", "valeur": k, "AUROC test": v["test"]})
cmp_ = pd.DataFrame(rows); cmp_.to_csv(RES / "comparaisons.csv", index=False)
cmp_.pivot_table(index=["comparaison", "valeur"], columns="catégorie", values="AUROC test").round(3)[CATS]"""),

md("""## 4. Problème 2 résolu ? Le seuil tient-il sur le test ?
Seuil = 95e centile des scores **hors-pli** de toutes les images saines. On vise donc environ **5 %** d'images saines rejetées. Sur le test, le taux observé doit en être proche ; en version 1, il atteignait 15 à 50 %."""),
code("""rows, v1 = [], {"bottle": ("4/20", "61/63", 0.959), "hazelnut": ("6/40", "32/70", 0.684), "carpet": ("14/28", "67/89", 0.699)}
fig, axes = plt.subplots(1, len(CATS), figsize=(5.5 * len(CATS), 4))
for ax, c in zip(axes, CATS):
    det, f = DET[c], F[c]
    s_te = det.score_features(f["test"]["zones"], f["test"]["global"])[0]; y = f["test"]["labels"]
    bins = np.linspace(np.percentile(s_te, 1), np.percentile(s_te, 99), 35)
    ax.hist(s_te[y == 0], bins, alpha=0.55, color="tab:green", label="test sain", density=True)
    ax.hist(s_te[y == 1], bins, alpha=0.55, color="tab:red", label="test défaut", density=True)
    ax.axvline(det.threshold_, color="k", lw=2, label="seuil (95e centile hors-pli sain)"); ax.set_title(c); ax.legend(fontsize=8)
    pred = (s_te > det.threshold_).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    rows.append({"catégorie": c, "v1 : AUROC": v1[c][2], "v1 : sains rejetés": v1[c][0], "v1 : défauts détectés": v1[c][1],
                 "v2 : AUROC": round(roc_auc_score(y, s_te), 3), "v2 : sains rejetés": f"{fp}/{fp + tn} ({fp / (fp + tn):.0%})",
                 "v2 : défauts détectés": f"{tp}/{tp + fn} ({tp / (tp + fn):.0%})", "v2 : précision": round(p, 3),
                 "v2 : rappel": round(r, 3), "v2 : F1": round(f1, 3)})
plt.tight_layout(); plt.savefig(RES / "distribution_scores.png", dpi=110); plt.show()
res = pd.DataFrame(rows).set_index("catégorie"); res.to_csv(RES / "detecteur_retenu.csv"); res.T"""),
code("""base = pd.read_csv(ROOT / "results" / "03_caracteristiques" / "auroc_sans_modele.csv", index_col=[0, 1])
summary = pd.DataFrame({
    "sans modèle (meilleure famille, choisie a posteriori sur le test)": base.max(),
    "détecteur retenu v2 (choisi sans vrai défaut)": pd.Series({c: DET[c].info["AUROC test"] for c in CATS}),
    "meilleure config. de la grille (a posteriori, pour information)": pd.Series({c: Z[c]["AUROC test (rapport)"].max() for c in CATS}),
}).T[CATS].round(3)
summary.to_csv(RES / "resume_auroc.csv"); summary"""),
md("## Conclusions : les deux problèmes de la version 1 sont-ils résolus ?\n\n| | bottle | hazelnut | carpet |\n|---|---|---|---|\n| AUROC test v1 → **v2** | 0,959 → **0,989** | 0,684 → **0,815** | 0,699 → **0,810** |\n| Sains rejetés à tort v1 → **v2** (cible 5 %) | 20 % → **10 %** | 15 % → **5 %** | 50 % → **0 %** |\n| Défauts détectés v2 | 97 % | 24 % | 57 % |\n| F1 v2 | 0,968 | 0,38 | 0,729 |\n\n**Problème 1 (critère de sélection aveugle) : résolu.** Avec des défauts synthétiques de 1 à 5 % de l'image et une notation hors-pli sur toutes les images saines :\n- le critère **départage** maintenant les configurations (de 0,50 à 0,90, contre un plateau entre 0,54 et 0,59) ;\n- il **localise** le défaut synthétique dans 60 à 95 % des cas, contre 20 % ;\n- il **prédit bien le classement du test** pour bottle (Spearman 0,72) et carpet (0,83).\n\n*Limite restante pour hazelnut :* le critère n'est pas corrélé au test (−0,02). Les vrais défauts des noisettes (impressions, coupures fines) ne ressemblent pas assez aux défauts synthétiques, et l'alignement imparfait des noix produit beaucoup de zones « anormales » naturelles.\n\n**Problème 2 (seuil trop bas pour le test) : résolu.** Deux causes, deux corrections, sans aucune étiquette de test :\n1. *La calibration reposait sur 42 à 78 images de validation*, trop peu pour estimer un centile de score maximal. → **Validation croisée en 5 plis** sur toutes les images saines (de 209 à 391), avec un seuil au 95e centile des scores hors-pli.\n2. *Carpet : l'éclairage est non uniforme dans le test.* Un diagnostic **sans étiquettes** montre un dégradé gauche-droite de +13,8 niveaux contre +5,2 pour les images saines. → **Normalisation locale de l'éclairage** (soustraction d'un fond estimé par filtre passe-bas, puis division par le contraste local, Ch. 2 et 3). Elle s'applique aux textures seulement et ramène le dégradé à +2,8.\n\nLes taux de rejet observés (10 %, 5 % et 0 %, mesurés sur seulement 20 à 40 images saines de test) sont cohérents avec la cible de 5 %.\n\n**Amélioration de hazelnut : zones de 16 px.** Les défauts de hazelnut sont fins (une coupure couvre 0,8 % de l'image en médiane). Un essai avec des zones de 16×16 pixels (961 zones) a été décidé **par une règle fixée à l'avance** : on l'adopte si le critère synthétique hors-pli s'améliore. Il passe de 0,724 à **0,795**, donc l'essai est adopté. Au test, l'AUROC passe de 0,765 à **0,815**.\n\n| hazelnut, rappel à taux de fausses alarmes fixé (test) | 5 % | 10 % | 20 % |\n|---|---|---|---|\n| zones de 32 px | 36 % | 43 % | 56 % |\n| zones de 16 px | 31 % | 46 % | **74 %** |\n\n**Ce qui reste à améliorer : le rappel de hazelnut au point de fonctionnement de 5 %** (24 % des défauts détectés). L'AUROC progresse et le rappel augmente nettement dès 10 à 20 % de fausses alarmes. En revanche, au seuil très strict de 5 %, quelques noisettes saines ont des scores élevés (formes atypiques, alignement imparfait) et masquent les défauts. C'est la limite d'une approche classique (caractéristiques manuelles et alignement rigide) sur des produits naturels de forme variable. Le compromis entre fausses alarmes et rappel est un **choix de cahier des charges** : avec 20 % de fausses alarmes tolérées, on détecterait 74 % des défauts."),
]
# conclusions : fichier tools/conclusions/XX.md (les conclusions intégrées plus haut sont ignorées)
from pathlib import Path as _P
cells = [c for c in cells if not (c.cell_type == "markdown" and "Conclusions" in c.source.lstrip().splitlines()[0])]
_f = _P(__file__).with_name("conclusions") / (_P(__file__).stem[-2:] + ".md")
if _f.exists():
    cells.append(md(_f.read_text(encoding="utf-8")))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}})
nbf.write(nb, sys.argv[1])
