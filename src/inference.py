"""Étape 4 du pipeline : reconnaissance / inférence sur une image BRUTE.

image brute ─► prétraitement (gris, médian, normalisation, alignement) ─► caractéristiques
            ─► scores des zones (ensemble OC-SVM / IForest) ─► score d'image
            ─► DÉCISION : score > seuil  =>  ANOMALIE, sinon NORMALE
            ─► LOCALISATION : carte de chaleur des zones, reprojetée sur l'image d'origine
            ─► DESCRIPTION du défaut (Ch. 3, représentation des formes) :
               binarisation de la carte -> contours -> boîte englobante, aire, centroïde, % de surface
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.data_loader import load_image
from src.models import AnomalyDetector


@dataclass
class Region:
    bbox: tuple            # (x, y, largeur, hauteur) en pixels de l'image d'origine
    area_px: float         # aire du contour (pixels)
    area_pct: float        # % de la surface de l'image
    centroid: tuple        # (cx, cy)
    peak: float            # score de zone maximal dans la région (z-score)


@dataclass
class Prediction:
    category: str
    score: float
    threshold: float
    is_anomaly: bool
    zone_scores: np.ndarray            # grille n x n des z-scores de zone
    heatmap: np.ndarray                # carte (taille d'origine), z-scores interpolés
    regions: list = field(default_factory=list)
    original: np.ndarray | None = None  # image BGR d'origine
    aligned: np.ndarray | None = None   # image prétraitée (256 x 256)
    align_info: object = None
    time_ms: float = 0.0

    @property
    def label(self) -> str:
        return "ANOMALIE" if self.is_anomaly else "NORMALE"

    def summary(self) -> dict:
        return {"catégorie": self.category, "décision": self.label, "score": round(self.score, 3),
                "seuil": round(self.threshold, 3), "nb régions": len(self.regions),
                "surface défaut %": round(sum(r.area_pct for r in self.regions), 2),
                "temps (ms)": round(self.time_ms, 1)}


# --------------------------------------------------------------------------- #

def zones_to_pixels(zone_scores: np.ndarray, size: int, zone: int, stride: int) -> np.ndarray:
    """Reprojette les scores de zone sur les pixels : chaque pixel reçoit la MOYENNE des scores
    des zones (qui se chevauchent) qui le recouvrent."""
    n = int(round(np.sqrt(zone_scores.size)))
    grid = zone_scores.reshape(n, n)
    acc = np.zeros((size, size), np.float32)
    cnt = np.zeros((size, size), np.float32)
    for i in range(n):
        for j in range(n):
            y, x = i * stride, j * stride
            acc[y:y + zone, x:x + zone] += grid[i, j]
            cnt[y:y + zone, x:x + zone] += 1
    return acc / np.maximum(cnt, 1)


def to_original(heat: np.ndarray, matrix: np.ndarray, shape: tuple) -> np.ndarray:
    """Ramène une carte de l'espace aligné (256 x 256) vers l'image d'origine :
    inverse de la transformation d'alignement (Ch. 2), puis redimensionnement bilinéaire."""
    s = heat.shape[0]
    inv = cv2.invertAffineTransform(matrix.astype(np.float32))
    back = cv2.warpAffine(heat, inv, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(back, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def extract_regions(heat: np.ndarray, z_threshold: float, min_area_pct: float) -> tuple[list, np.ndarray]:
    """Binarisation de la carte (seuil sur le z-score) -> contours -> mesures géométriques (Ch. 3)."""
    mask = (heat > z_threshold).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = heat.shape[0] * heat.shape[1]
    regions = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 100 * area / total < min_area_pct:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
        peak = float(heat[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0].max())
        regions.append(Region((x, y, w, h), float(area), 100 * area / total, (cx, cy), peak))
    regions.sort(key=lambda r: -r.peak)
    return regions, mask


class Inspector:
    """Charge un détecteur entraîné et analyse des images brutes."""

    def __init__(self, detector: AnomalyDetector | str | Path, cfg: dict | None = None):
        self.det = AnomalyDetector.load(detector) if isinstance(detector, (str, Path)) else detector
        inf = (cfg or {}).get("inference", {})
        self.z_threshold = inf.get("zone_z_threshold", 3.0)
        self.min_area_pct = inf.get("min_region_pct", 0.05)
        self.sigma = inf.get("heatmap_sigma", 4)

    @classmethod
    def for_category(cls, category: str, cfg: dict) -> "Inspector":
        from src.data_loader import PROJECT_ROOT
        return cls(PROJECT_ROOT / cfg["paths"]["models"] / f"{category}_detector.joblib", cfg)

    def predict(self, image: np.ndarray | str | Path) -> Prediction:
        t0 = time.perf_counter()
        raw = load_image(image) if isinstance(image, (str, Path)) else image
        det, fx = self.det, self.det.fx
        aligned, info = det.pre.transform(raw)
        Z = fx.zones(aligned)[None]
        G = fx.global_(aligned)[None] if det.global_model is not None else np.zeros((1, 1), np.float32)
        score, zs = det.score_features(Z, G)
        zs = zs[0]
        heat = zones_to_pixels(zs, aligned.shape[0], fx.zone, fx.stride)
        if self.sigma:
            heat = cv2.GaussianBlur(heat, (0, 0), self.sigma)
        heat_orig = to_original(heat, info.matrix, raw.shape[:2])
        is_anom = bool(score[0] > det.threshold_)
        regions = extract_regions(heat_orig, self.z_threshold, self.min_area_pct)[0] if is_anom else []
        n = int(round(np.sqrt(zs.size)))
        return Prediction(det.category, float(score[0]), float(det.threshold_), is_anom, zs.reshape(n, n),
                          heat_orig, regions, raw, aligned, info, 1000 * (time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def render(pred: Prediction, gt_mask: np.ndarray | None = None, path: str | Path | None = None,
           vmax: float = 8.0, show: bool = False):
    """Figure : image annotée (boîtes, contour vérité terrain en vert), carte de chaleur, image prétraitée."""
    import matplotlib.pyplot as plt
    rgb = cv2.cvtColor(pred.original, cv2.COLOR_BGR2RGB).copy()
    thick = max(2, rgb.shape[0] // 250)
    if gt_mask is not None and gt_mask.any():
        cnts, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (0, 255, 0), thick)
    for k, r in enumerate(pred.regions):
        x, y, w, h = r.bbox
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), thick)
        cv2.putText(rgb, f"{k + 1}", (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, rgb.shape[0] / 700, (255, 0, 0), thick)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    color = "red" if pred.is_anomaly else "green"
    ax[0].imshow(rgb); ax[0].set_title(f"{pred.label}  (score {pred.score:.2f} / seuil {pred.threshold:.2f})",
                                       color=color, fontweight="bold")
    ax[1].imshow(cv2.cvtColor(pred.original, cv2.COLOR_BGR2RGB))
    im = ax[1].imshow(np.clip(pred.heatmap, 0, vmax), cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    ax[1].set_title("carte de chaleur (z-score des zones)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(pred.aligned, cmap="gray"); ax[2].set_title(
        f"prétraitée + alignée ({pred.align_info.method}, {pred.align_info.angle:.0f}°)")
    txt = "\n".join(f"{k + 1}: aire {r.area_pct:.2f} %, centre ({r.centroid[0]:.0f}, {r.centroid[1]:.0f}), "
                    f"boîte {r.bbox[2]}x{r.bbox[3]}" for k, r in enumerate(pred.regions[:4]))
    fig.text(0.01, 0.01, txt or "aucune région signalée", fontsize=8, family="monospace")
    for a in ax:
        a.axis("off")
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
