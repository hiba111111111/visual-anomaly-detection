"""Évaluation : python evaluate.py --all            (métriques des détecteurs finaux)
                python evaluate.py --all --ablation  (+ étude d'ablation, long : réentraîne des variantes)

Écrit les tableaux dans results/06_evaluation/.
"""
import argparse

import pandas as pd

from src.data_loader import PROJECT_ROOT, load_config
from src.evaluation import evaluate_detector, run_ablation


def main():
    ap = argparse.ArgumentParser(description="Évaluation des détecteurs d'anomalies")
    ap.add_argument("--category"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="étude d'ablation (réentraînement des variantes)")
    ap.add_argument("--skip-metrics", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cats = cfg["data"]["categories"] if a.all or not a.category else [a.category]
    out = PROJECT_ROOT / cfg["paths"]["results"] / "06_evaluation"; out.mkdir(parents=True, exist_ok=True)

    if not a.skip_metrics:
        img, pix, dft = [], [], []
        for c in cats:
            r = evaluate_detector(c, cfg)
            img.append({"catégorie": c, **r["image"]}); pix.append({"catégorie": c, **r["pixel"]})
            dft += [{"catégorie": c, "défaut": d, "AUROC": v} for d, v in r["per_defect"].items()]
            m = r["image"]
            print(f"{c:9s} AUROC {m['AUROC']:.3f} | AP {m['AP']:.3f} | F1 {m['F1']:.3f} | FA {m['FP']}/{m['FP'] + m['VN']} "
                  f"| détectés {m['VP']}/{m['VP'] + m['FN']} | AUROC pixel {r['pixel']['AUROC pixel']:.3f}", flush=True)
        pd.DataFrame(img).to_csv(out / "metriques_image.csv", index=False)
        pd.DataFrame(pix).to_csv(out / "metriques_pixel.csv", index=False)
        pd.DataFrame(dft).to_csv(out / "auroc_par_defaut.csv", index=False)

    if a.ablation:
        for c in cats:
            df = run_ablation(c, cfg)
            df.to_csv(out / f"ablation_{c}.csv", index=False)


if __name__ == "__main__":
    main()
