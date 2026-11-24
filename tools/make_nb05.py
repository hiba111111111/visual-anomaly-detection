import nbformat as nbf, sys
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = [
md("""# 05 — Reconnaissance / inférence sur des images brutes
Module `src/inference.py`, script `predict.py`.

```
image brute → prétraitement (gris, médian, normalisation, alignement) → caractéristiques des zones
            → scores des zones (ensemble de 5 modèles) → score d'image
            → DÉCISION : score > seuil → ANOMALIE, sinon NORMALE
            → LOCALISATION : carte de chaleur des zones, reprojetée sur l'image d'origine
              (inverse de la transformation d'alignement, Ch. 2)
            → DESCRIPTION (Ch. 3) : carte binarisée (z > 3) → contours → boîte, aire, centroïde
```
Ce notebook part des **fichiers image** du test, et non du cache de caractéristiques. Il vérifie donc la chaîne complète, telle qu'elle serait utilisée en production."""),
code("""import sys, time, subprocess
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from src.data_loader import load_config, get_splits, load_mask
from src.inference import Inspector, render

cfg = load_config(ROOT / "config.yaml"); CATS = cfg["data"]["categories"]
OUT = ROOT / "results" / "05_inference"; OUT.mkdir(parents=True, exist_ok=True)
INSP = {c: Inspector.for_category(c, cfg) for c in CATS}
TEST = {c: get_splits(c, cfg)["test"] for c in CATS}
print({c: f"seuil {INSP[c].det.threshold_:.2f}" for c in CATS})"""),

md("## 1. Inférence sur tout le jeu de test (à partir des images brutes)"),
code("""PRED = {}
for c in CATS:
    PRED[c] = [INSP[c].predict(x.path) for x in tqdm(TEST[c], desc=c)]
rows = []
for c in CATS:
    y = np.array([x.label for x in TEST[c]]); p = np.array([pr.is_anomaly for pr in PRED[c]]).astype(int)
    t = np.array([pr.time_ms for pr in PRED[c]])
    rows.append({"catégorie": c, "images": len(y), "sains rejetés": f"{((p == 1) & (y == 0)).sum()}/{(y == 0).sum()}",
                 "défauts détectés": f"{((p == 1) & (y == 1)).sum()}/{(y == 1).sum()}",
                 "temps moyen (ms/image)": round(t.mean()), "temps max (ms)": round(t.max())})
df_inf = pd.DataFrame(rows).set_index("catégorie"); df_inf.to_csv(OUT / "inference_test.csv"); df_inf"""),
md("""Ces décisions doivent être **identiques** à celles du notebook 04, qui partait des caractéristiques en cache : c'est la vérification que la chaîne d'inférence reproduit exactement celle de l'apprentissage."""),
code("""from src.features import compute_features
for c in CATS:
    f = compute_features(c, cfg, verbose=False); det = INSP[c].det
    s_cache = det.score_features(f["test"]["zones"], f["test"]["global"])[0]
    s_raw = np.array([pr.score for pr in PRED[c]])
    print(f"{c:9s} écart max des scores (image brute vs cache) : {np.abs(s_cache - s_raw).max():.1e} | "
          f"décisions identiques : {np.array_equal(s_cache > det.threshold_, s_raw > det.threshold_)}")"""),

md("""## 2. Exemples : une image saine et un exemple par type de défaut
Contour **vert** = vérité terrain ; boîtes **rouges** = régions signalées par le détecteur (seulement si l'image est jugée anormale)."""),
code("""def gt(x):
    return load_mask(x) if x.label else None
rows = []
for c in CATS:
    test = TEST[c]
    picks = [next(i for i, x in enumerate(test) if x.label == 0)]
    picks += [next(i for i, x in enumerate(test) if x.defect == d) for d in sorted({x.defect for x in test if x.label})]
    for i in picks:
        x, pr = test[i], PRED[c][i]
        render(pr, gt_mask=gt(x), path=OUT / f"{c}_{x.defect}_{x.path.stem}.png", show=True)
        m = gt(x)
        hit = None
        if m is not None and pr.regions:
            bx, by, bw, bh = pr.regions[0].bbox
            hit = bool(m[by:by + bh, bx:bx + bw].any())
        rows.append({"catégorie": c, "vérité": x.defect, **pr.summary(),
                     "région 1 sur le défaut": "—" if hit is None else ("oui" if hit else "non")})
pd.DataFrame(rows).to_csv(OUT / "exemples.csv", index=False); pd.DataFrame(rows)"""),

md("""## 3. Localisation : la région n° 1 tombe-t-elle sur le vrai défaut ?
Pour chaque défaut **détecté**, on vérifie si la boîte de la région la plus anormale recouvre le masque vérité terrain. Les mesures pixel par pixel (AUROC pixel, IoU) viendront à l'étape 8."""),
code("""rows = []
for c in CATS:
    hits, n_det = [], 0
    for x, pr in zip(TEST[c], PRED[c]):
        if x.label and pr.is_anomaly and pr.regions:
            n_det += 1; m = load_mask(x); bx, by, bw, bh = pr.regions[0].bbox
            hits.append(m[by:by + bh, bx:bx + bw].any())
    rows.append({"catégorie": c, "défauts détectés avec région": n_det,
                 "région n° 1 sur le défaut": f"{np.mean(hits):.0%}" if hits else "—",
                 "régions par image (moyenne)": round(np.mean([len(pr.regions) for x, pr in zip(TEST[c], PRED[c]) if x.label and pr.is_anomaly]), 1)})
df_loc = pd.DataFrame(rows).set_index("catégorie"); df_loc.to_csv(OUT / "localisation_region1.csv"); df_loc"""),

md("""## 4. Erreurs : une fausse alarme et un défaut manqué par catégorie
Montrer les erreurs aide à comprendre les limites du détecteur."""),
code("""for c in CATS:
    test, preds = TEST[c], PRED[c]
    fa = [i for i, (x, p) in enumerate(zip(test, preds)) if x.label == 0 and p.is_anomaly]
    miss = [i for i, (x, p) in enumerate(zip(test, preds)) if x.label == 1 and not p.is_anomaly]
    if fa:
        i = max(fa, key=lambda i: preds[i].score); print(f"{c} — fausse alarme la plus forte : {test[i].path.name}")
        render(preds[i], path=OUT / f"erreur_{c}_fausse_alarme.png", show=True)
    else:
        print(f"{c} — aucune fausse alarme sur le test")
    if miss:
        i = min(miss, key=lambda i: preds[i].score); print(f"{c} — défaut le plus manqué : {test[i].defect} {test[i].path.name}")
        render(preds[i], gt_mask=load_mask(test[i]), path=OUT / f"erreur_{c}_defaut_manque.png", show=True)"""),

md("## 5. Utilisation en ligne de commande (`predict.py`)"),
code("""cmd = [sys.executable, "-W", "ignore", str(ROOT / "predict.py"), "--category", "bottle",
       "--image", str(ROOT / "data/mvtec/bottle/test/broken_large/000.png"), str(ROOT / "data/mvtec/bottle/test/good/000.png")]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                   env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
print(r.stdout or r.stderr)"""),
md("## Conclusions de l'étape 7\n\n**1. La chaîne d'inférence reproduit exactement l'apprentissage.** Partir des fichiers image donne les mêmes scores que le cache (écart inférieur à 10⁻⁶) et **les mêmes décisions** pour les 310 images de test.\n\n**2. Décisions sur le test** (seuil = 95e centile des scores hors-pli des images saines) :\n\n| | sains rejetés | défauts détectés | temps moyen |\n|---|---|---|---|\n| bottle | 2/20 | 61/63 | ~0,3 s |\n| hazelnut | 2/40 | 17/70 | ~0,8 s (961 zones) |\n| carpet | 0/28 | 51/89 | ~0,35 s |\n\n**3. Localisation.** Quand un défaut est détecté, la région la plus anormale (n° 1) tombe sur le vrai défaut dans **100 %** (bottle), **94 %** (hazelnut) et **92 %** (carpet) des cas.\n\n**4. Limites observées :**\n- **Les boîtes sont plus grandes que les défauts.** La carte de chaleur a la résolution des zones (32 ou 16 pixels sur une image de 256) et elle est lissée. Par exemple, la tache de couleur sur carpet est encadrée par une boîte de 16 % de l'image pour un défaut d'environ 3 %. L'**aire** annoncée est donc une **surestimation**. La précision pixel par pixel (AUROC pixel, IoU) sera mesurée à l'étape 8.\n- Pour 3 tapis détectés comme anormaux, aucune zone ne dépasse z = 3 : l'image est jugée anormale par la moyenne des 3 zones les plus suspectes, sans région isolée.\n- La fausse alarme la plus forte sur bottle est une bouteille **en partie hors du cadre** (coupée en haut de l'image). C'est une acquisition atypique, et la signaler comme anomalie se défend.\n- **Défauts manqués :** les plus difficiles sont les coupures fines de hazelnut (`cut`), la contamination métallique de carpet (0,4 % de l'image en médiane) et la contamination légère de bottle."),
]
# conclusions : fichier tools/conclusions/XX.md (les conclusions intégrées plus haut sont ignorées)
from pathlib import Path as _P
cells = [c for c in cells if not (c.cell_type == "markdown" and "Conclusions" in c.source.lstrip().splitlines()[0])]
_f = _P(__file__).with_name("conclusions") / (_P(__file__).stem[-2:] + ".md")
if _f.exists():
    cells.append(md(_f.read_text(encoding="utf-8")))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}})
nbf.write(nb, sys.argv[1])
