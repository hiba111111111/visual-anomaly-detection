"""Chargement du jeu de données MVTec AD.

Structure attendue (une catégorie) :
    <root>/<categorie>/train/good/*.png              -> images saines (apprentissage)
    <root>/<categorie>/test/good/*.png               -> images saines (évaluation)
    <root>/<categorie>/test/<type_defaut>/*.png      -> anomalies (évaluation uniquement)
    <root>/<categorie>/ground_truth/<type>/*_mask.png -> masques des défauts

Principe semi-supervisé : l'apprentissage n'utilise QUE train/good.
Une partie de train/good (val_ratio) est mise de côté pour fixer le seuil
de décision, sans jamais utiliser d'image défectueuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Sample:
    path: Path              # chemin de l'image
    label: int              # 0 = normale, 1 = anomalie
    defect: str             # "good" ou type de défaut (ex. "broken_large")
    mask_path: Path | None  # masque vérité terrain (None pour les images saines)
    category: str


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def _category_dir(root: str | Path, category: str) -> Path:
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    d = root / category
    if not d.is_dir():
        raise FileNotFoundError(f"Catégorie introuvable : {d}")
    return d


def list_train(root: str | Path, category: str) -> list[Sample]:
    """Images saines d'apprentissage (train/good)."""
    d = _category_dir(root, category) / "train" / "good"
    return [Sample(p, 0, "good", None, category) for p in _list_images(d)]


def list_test(root: str | Path, category: str) -> list[Sample]:
    """Images de test (saines + défauts) avec leur masque éventuel."""
    cdir = _category_dir(root, category)
    samples: list[Sample] = []
    for defect_dir in sorted(p for p in (cdir / "test").iterdir() if p.is_dir()):
        defect = defect_dir.name
        for img in _list_images(defect_dir):
            if defect == "good":
                samples.append(Sample(img, 0, "good", None, category))
            else:
                mask = cdir / "ground_truth" / defect / f"{img.stem}_mask.png"
                samples.append(Sample(img, 1, defect, mask if mask.exists() else None, category))
    return samples


def split_train_val(samples: list[Sample], val_ratio: float = 0.2, seed: int = 42
                    ) -> tuple[list[Sample], list[Sample]]:
    """Découpe train/good en apprentissage / validation (reproductible)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_val = int(round(len(samples) * val_ratio))
    val = [samples[i] for i in sorted(idx[:n_val])]
    fit = [samples[i] for i in sorted(idx[n_val:])]
    return fit, val


def get_splits(category: str, cfg: dict | None = None) -> dict[str, list[Sample]]:
    """Renvoie {'train', 'val', 'test'} pour une catégorie."""
    cfg = cfg or load_config()
    d = cfg["data"]
    fit, val = split_train_val(list_train(d["root"], category), d["val_ratio"], d["seed"])
    return {"train": fit, "val": val, "test": list_test(d["root"], category)}


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Lit une image (BGR ou niveaux de gris) ; compatible chemins avec accents/espaces."""
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise IOError(f"Lecture impossible : {path}")
    return img


def load_mask(sample: Sample, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Masque binaire (0/1). Image saine -> masque vide de la taille de l'image."""
    if sample.mask_path is None:
        if shape is None:
            shape = load_image(sample.path, grayscale=True).shape[:2]
        return np.zeros(shape, dtype=np.uint8)
    m = load_image(sample.mask_path, grayscale=True)
    return (m > 127).astype(np.uint8)


def summary(category: str, cfg: dict | None = None) -> dict:
    """Comptage des images par sous-ensemble et par type de défaut."""
    s = get_splits(category, cfg)
    test = s["test"]
    per_defect: dict[str, int] = {}
    for x in test:
        per_defect[x.defect] = per_defect.get(x.defect, 0) + 1
    return {
        "categorie": category,
        "train": len(s["train"]),
        "val": len(s["val"]),
        "test_normales": sum(x.label == 0 for x in test),
        "test_anomalies": sum(x.label == 1 for x in test),
        "masques_manquants": sum(x.label == 1 and x.mask_path is None for x in test),
        "defauts": per_defect,
    }


if __name__ == "__main__":
    cfg = load_config()
    for cat in cfg["data"]["categories"]:
        print(summary(cat, cfg))
