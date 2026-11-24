"""Inférence : python predict.py --category bottle --image chemin.png [chemin2.png | dossier ...]

Affiche la décision (NORMALE / ANOMALIE), le score, le seuil et les régions défectueuses
(boîte englobante, aire, centroïde), et enregistre une image annotée dans results/predictions/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : les figures sont enregistrées

from src.data_loader import IMG_EXT, PROJECT_ROOT, load_config
from src.inference import Inspector, render


def main():
    ap = argparse.ArgumentParser(description="Détection d'anomalies visuelles sur des images brutes")
    ap.add_argument("--category", required=True, help="catégorie du détecteur (ex. bottle)")
    ap.add_argument("--image", required=True, nargs="+", help="image(s) ou dossier(s)")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : results/predictions/<catégorie>)")
    ap.add_argument("--no-figure", action="store_true", help="ne pas enregistrer d'image annotée")
    a = ap.parse_args()

    cfg = load_config()
    insp = Inspector.for_category(a.category, cfg)
    out = Path(a.out) if a.out else PROJECT_ROOT / cfg["paths"]["results"] / "predictions" / a.category
    paths = []
    for p in map(Path, a.image):
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]

    for p in paths:
        pred = insp.predict(p)
        print(f"{p.name:30s} {pred.label:9s} score {pred.score:6.2f} (seuil {pred.threshold:.2f})  {pred.time_ms:5.0f} ms")
        for k, r in enumerate(pred.regions, 1):
            x, y, w, h = r.bbox
            print(f"    région {k}: boîte x={x} y={y} {w}x{h} px | aire {r.area_pct:.2f} % | "
                  f"centroïde ({r.centroid[0]:.0f}, {r.centroid[1]:.0f}) | z max {r.peak:.1f}")
        if not a.no_figure:
            f = out / f"{p.parent.name}_{p.stem}.png"
            render(pred, path=f)
            print(f"    -> {f}")


if __name__ == "__main__":
    main()
