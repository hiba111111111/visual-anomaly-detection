"""Anomalies synthétiques pour CHOISIR les hyperparamètres sans utiliser de vrai défaut.

Principe semi-supervisé : on ne dispose que d'images saines. Pour comparer des réglages,
on dégrade artificiellement des images saines (notées « hors-pli » : par un modèle qui ne les a
pas vues), avec des défauts simulés inspirés des dégradations du Ch. 3, puis on mesure l'AUROC
« saine » contre « dégradée ». Le TEST (vrais défauts) n'intervient jamais dans le choix.

Taille des défauts simulés : 1 % à 5 % de la surface de l'image (tirage log-uniforme).
C'est la « taille minimale de défaut à détecter » du cahier des charges. Une première version
avec des défauts de 0,4 % rendait le critère aveugle sur hazelnut et carpet (notebook 04).

Défauts simulés (appliqués à l'image déjà prétraitée) :
    - tache    : ellipse plus claire ou plus sombre, bords adoucis   (contamination, trou, couleur)
    - rayure   : segment épais clair ou sombre                      (rayure, coupure)
    - fissure  : ligne brisée sombre et irrégulière                 (fissure, fil)
    - cutpaste : morceau de l'image recopié ailleurs et tourné      (structure incohérente)
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing import otsu_mask

KINDS = ("tache", "rayure", "fissure", "cutpaste")
AREA_RANGE = (0.01, 0.05)          # fraction de l'image couverte par le défaut


def _random_point(rng, region: np.ndarray, margin: int = 20) -> tuple[int, int]:
    h, w = region.shape
    ys, xs = np.nonzero(region[margin:h - margin, margin:w - margin])
    if len(ys) == 0:
        return int(rng.integers(margin, w - margin)), int(rng.integers(margin, h - margin))
    i = rng.integers(len(ys))
    return int(xs[i] + margin), int(ys[i] + margin)


def _target_area(rng, shape) -> float:
    lo, hi = np.log(AREA_RANGE[0]), np.log(AREA_RANGE[1])
    return float(np.exp(rng.uniform(lo, hi))) * shape[0] * shape[1]


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator, on_object: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (image dégradée, masque binaire du défaut simulé)."""
    h, w = img.shape
    region = otsu_mask(img) if on_object else np.ones_like(img)
    out = img.astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    area = _target_area(rng, img.shape)
    cx, cy = _random_point(rng, region, margin=24)
    sign = rng.choice([-1.0, 1.0])

    if kind == "tache":
        ratio = rng.uniform(0.4, 1.0)                       # ellipse : aire = pi*a*b
        a = np.sqrt(area / (np.pi * ratio)); b = a * ratio
        cv2.ellipse(mask, (cx, cy), (int(a), max(int(b), 2)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
        out = out + sign * rng.uniform(35, 80) * soft
    elif kind == "rayure":
        t = int(rng.integers(3, 8)); L = min(area / t, 200)
        ang = rng.uniform(0, np.pi)
        p1 = (int(cx - L / 2 * np.cos(ang)), int(cy - L / 2 * np.sin(ang)))
        p2 = (int(cx + L / 2 * np.cos(ang)), int(cy + L / 2 * np.sin(ang)))
        cv2.line(mask, p1, p2, 1, t)
        val = 0.0 if out[mask > 0].mean() > 127 else 255.0
        out[mask > 0] = 0.35 * out[mask > 0] + 0.65 * val
    elif kind == "fissure":
        t = int(rng.integers(2, 5)); L = min(area / t, 260)
        pts, ang, x, y = [], rng.uniform(0, 2 * np.pi), float(cx), float(cy)
        for _ in range(int(L // 8)):                          # marche aléatoire : segments de 8 px
            pts.append((int(x), int(y)))
            ang += rng.normal(0, 0.5); x += 8 * np.cos(ang); y += 8 * np.sin(ang)
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, t)
        out[mask > 0] = out[mask > 0] * rng.uniform(0.15, 0.4)
    elif kind == "cutpaste":
        s = int(np.clip(np.sqrt(area), 16, min(h, w) // 3))
        sx, sy = _random_point(rng, region, margin=s // 2 + 2)
        patch = img[sy - s // 2: sy - s // 2 + s, sx - s // 2: sx - s // 2 + s]
        M = cv2.getRotationMatrix2D((s / 2, s / 2), float(rng.uniform(45, 315)), 1.0)
        patch = cv2.warpAffine(patch, M, (s, s), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        patch = patch * rng.uniform(0.8, 1.2)                # léger changement de réflectance
        x0, y0 = min(max(cx - s // 2, 0), w - s), min(max(cy - s // 2, 0), h - s)
        out[y0:y0 + s, x0:x0 + s] = patch
        mask[y0:y0 + s, x0:x0 + s] = 1
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8), mask


def make_synthetic_set(images: np.ndarray, seed: int = 0, on_object: bool = True, per_image: int = 1
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`per_image` défauts par image (types en rotation).
    -> (images, masques, types, indice de l'image saine d'origine)."""
    rng = np.random.default_rng(seed)
    imgs, masks, kinds, base = [], [], [], []
    for i, im in enumerate(images):
        for r in range(per_image):
            k = KINDS[(i * per_image + r) % len(KINDS)]
            a, m = add_defect(im, k, rng, on_object)
            imgs.append(a); masks.append(m); kinds.append(k); base.append(i)
    return np.stack(imgs), np.stack(masks), np.array(kinds), np.array(base)
