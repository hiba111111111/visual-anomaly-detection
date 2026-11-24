"""Évaluation finale et étude d'ablation.

Niveau IMAGE  : AUROC, précision moyenne (AP), précision / rappel / F1 au seuil du détecteur,
                matrice de confusion, rappel à taux de fausses alarmes fixé (5, 10, 20 %).
Niveau DÉFAUT : AUROC de chaque type de défaut contre les images saines.
Niveau PIXEL  : carte d'anomalie (scores de zone reprojetés sur les pixels, espace aligné 256x256)
                comparée au masque vérité terrain recalé : AUROC pixel, IoU au seuil de zone.

Ablation : on retire / remplace UN élément du pipeline à la fois, la configuration du modèle
restant celle du détecteur final (« toutes choses égales par ailleurs »). Chaque variante est
réentraînée avec le même protocole (validation croisée, anomalies synthétiques, seuil au 95e
centile hors-pli) : aucun vrai défaut n'est utilisé pour l'entraîner.
"""
from __future__ import annotations

import ast
import copy
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from src.features import compute_features
from src.inference import zones_to_pixels
from src.models import AnomalyDetector


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #

def recall_at_fpr(y, s, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def image_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = (s > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"AUROC": roc_auc_score(y, s), "AP": average_precision_score(y, s),
            "précision": p, "rappel": r, "F1": f1, "exactitude": (tp + tn) / len(y),
            "VP": int(tp), "FP": int(fp), "VN": int(tn), "FN": int(fn),
            "taux fausses alarmes": fp / max(fp + tn, 1),
            "rappel @5% FA": recall_at_fpr(y, s, 0.05), "rappel @10% FA": recall_at_fpr(y, s, 0.10),
            "rappel @20% FA": recall_at_fpr(y, s, 0.20)}


def per_defect_auroc(y: np.ndarray, s: np.ndarray, defects: np.ndarray) -> dict:
    good = defects == "good"
    return {d: roc_auc_score(np.r_[np.zeros(good.sum()), np.ones((defects == d).sum())],
                             np.r_[s[good], s[defects == d]])
            for d in sorted(set(defects) - {"good"})}


def pixel_maps(det: AnomalyDetector, zone_scores: np.ndarray, sigma: float = 4) -> np.ndarray:
    """(n, n_zones) z-scores -> (n, 256, 256) cartes d'anomalie dans l'espace aligné."""
    fx, size = det.fx, det.pre.size
    out = []
    for z in zone_scores:
        m = zones_to_pixels(z, size, fx.zone, fx.stride)
        out.append(cv2.GaussianBlur(m, (0, 0), sigma) if sigma else m)
    return np.stack(out)


def pixel_metrics(maps: np.ndarray, masks: np.ndarray, labels: np.ndarray, z_threshold: float = 3.0,
                  max_pixels: int = 3_000_000, seed: int = 0) -> dict:
    """AUROC pixel (toutes les images de test) + IoU moyen sur les images défectueuses."""
    flat_s, flat_m = maps.ravel(), masks.ravel().astype(int)
    if flat_s.size > max_pixels:                      # sous-échantillonnage uniforme (mémoire)
        idx = np.random.default_rng(seed).choice(flat_s.size, max_pixels, replace=False)
        flat_s, flat_m = flat_s[idx], flat_m[idx]
    ious = []
    for mp, mk, y in zip(maps, masks, labels):
        if y == 1 and mk.any():
            pred = mp > z_threshold
            inter, union = np.logical_and(pred, mk > 0).sum(), np.logical_or(pred, mk > 0).sum()
            ious.append(inter / union if union else 0.0)
    return {"AUROC pixel": roc_auc_score(flat_m, flat_s), "IoU moyen (z>%g)" % z_threshold: float(np.mean(ious)),
            "IoU médian": float(np.median(ious))}


def evaluate_detector(category: str, cfg: dict, det: AnomalyDetector | None = None) -> dict:
    """Toutes les métriques du détecteur final d'une catégorie (à partir du cache)."""
    from src.data_loader import PROJECT_ROOT
    det = det or AnomalyDetector.load(PROJECT_ROOT / cfg["paths"]["models"] / f"{category}_detector.joblib")
    f = compute_features(category, cfg, verbose=False)
    te = f["test"]
    s, zs = det.score_features(te["zones"], te["global"])
    y = te["labels"]
    maps = pixel_maps(det, zs, cfg["inference"].get("heatmap_sigma", 4))
    return {"category": category, "scores": s, "labels": y, "defects": te["defects"], "maps": maps,
            "masks": te["masks"], "images": te["images"], "threshold": det.threshold_,
            "image": image_metrics(y, s, det.threshold_), "per_defect": per_defect_auroc(y, s, te["defects"]),
            "pixel": pixel_metrics(maps, te["masks"], y, cfg["inference"].get("zone_z_threshold", 3.0)),
            "info": det.info}


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #

def _fixed_model_cfg(cfg: dict, det: AnomalyDetector) -> dict:
    """Grille réduite à la configuration du détecteur final (modèle, stratégie, famille)."""
    c = copy.deepcopy(cfg)
    zone_desc = det.info["zone"]                       # ex. "toutes | shared | ocsvm(nu=0.05, gamma=0.5)"
    fs, strategy, model = [x.strip() for x in zone_desc.split("|")]
    kind, params = model.split("(", 1)
    params = dict(kv.split("=") for kv in params.rstrip(")").split(", "))
    c["model"]["zone_strategies"] = [strategy]
    c["model"]["zone_feature_sets"] = {fs: cfg["model"]["zone_feature_sets"][fs]}
    if kind == "ocsvm":
        g = params["gamma"]
        c["model"]["ocsvm"]["nu"] = [float(params["nu"])]
        c["model"]["ocsvm"]["gamma"] = [g if g == "scale" else float(g)]
        c["model"]["iforest"]["n_estimators"] = []
    else:
        c["model"]["ocsvm"]["nu"] = []
        c["model"]["iforest"]["n_estimators"] = [int(params["n_estimators"])]
    return c


def ablation_variants(cfg: dict, category: str) -> dict[str, callable]:
    """Variantes : chacune modifie UNE chose dans une copie de la configuration."""
    is_texture = category in cfg["preprocessing"].get("no_align_categories", [])
    V = {"complet (référence)": lambda c: c}

    def no_median(c):
        c["preprocessing"]["denoise"] = "none"; return c

    def no_norm(c):
        c["preprocessing"]["normalize"] = "none"
        c["preprocessing"].get("normalize_per_category", {}).pop(category, None); return c

    V["sans filtre médian"] = no_median
    V["sans normalisation photométrique"] = no_norm
    if category in cfg["preprocessing"].get("normalize_per_category", {}):
        def global_norm(c):
            c["preprocessing"]["normalize_per_category"].pop(category, None); return c
        V["normalisation globale minmax (au lieu de locale)"] = global_norm
    if not is_texture:
        def no_align(c):
            c["preprocessing"].setdefault("align_method_per_category", {})[category] = "none"; return c
        V["sans alignement"] = no_align
    if category in cfg.get("features_per_category", {}):
        def default_zones(c):
            c["features_per_category"].pop(category, None); return c
        V["zones 32 px (au lieu de 16)"] = default_zones
    return V


def family_variants(cfg: dict, fx) -> dict[str, list]:
    fam = fx.zone_families()
    return {f"famille seule : {k}": [k] for k in fam}


def run_ablation(category: str, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    from src.data_loader import PROJECT_ROOT
    from src.training import run_selection
    det = AnomalyDetector.load(PROJECT_ROOT / cfg["paths"]["models"] / f"{category}_detector.joblib")
    base = _fixed_model_cfg(cfg, det)
    fs_name = list(base["model"]["zone_feature_sets"])[0]
    jobs = [(name, fn(copy.deepcopy(base))) for name, fn in ablation_variants(cfg, category).items()]
    fx = compute_features(category, cfg, verbose=False)["fx"]
    for name, fams in family_variants(cfg, fx).items():
        c = copy.deepcopy(base); c["model"]["zone_feature_sets"] = {name: fams}; jobs.append((name, c))
    rows = []
    for name, c in jobs:
        t0 = time.time()
        r = run_selection(category, c, verbose=False)
        d = r["detector"]; te = r["feats"]["test"]
        s = d.score_features(te["zones"], te["global"])[0]
        m = image_metrics(te["labels"], s, d.threshold_)
        rows.append({"catégorie": category, "variante": name, "AUROC synthétique": d.info["AUROC synthétique"],
                     "AUROC test": m["AUROC"], "F1": m["F1"], "taux fausses alarmes": m["taux fausses alarmes"],
                     "rappel": m["rappel"], "rappel @10% FA": m["rappel @10% FA"],
                     "configuration": d.info["zone"], "temps (s)": time.time() - t0})
        if verbose:
            print(f"[{category}] {name:52s} AUROC test {m['AUROC']:.3f} | synth {d.info['AUROC synthétique']:.3f} "
                  f"| F1 {m['F1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)
