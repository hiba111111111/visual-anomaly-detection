"""Entraînement : python train.py --category bottle   (ou --all)

Sélectionne la configuration (anomalies synthétiques sur la validation saine),
entraîne le détecteur et l'enregistre dans models/<catégorie>_detector.joblib.
"""
import argparse

from src.data_loader import load_config
from src.training import train_category


def main():
    ap = argparse.ArgumentParser(description="Entraînement du détecteur d'anomalies (images saines uniquement)")
    ap.add_argument("--category", help="catégorie MVTec (ex. bottle)")
    ap.add_argument("--all", action="store_true", help="toutes les catégories de config.yaml")
    ap.add_argument("--config", default=None, help="chemin de config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    cats = cfg["data"]["categories"] if a.all or not a.category else [a.category]
    for c in cats:
        train_category(c, cfg)


if __name__ == "__main__":
    main()
