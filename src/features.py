"""Étape 2 du pipeline : extraction de caractéristiques.

Deux familles, calculées sur l'image PRÉTRAITÉE (gris 256x256, médian, normalisée, alignée) :

A. Caractéristiques PAR ZONE  (« histogrammes d'intensité sur des zones spécifiques »)
   Fenêtres de `zone` x `zone` pixels, pas `stride` (chevauchement), pour chaque zone :
     - histogramme d'intensité normalisé H(k)/N sur `hist_bins` intervalles      (Ch. 2)
     - moyenne (luminance), écart-type, contraste (max-min)/(max+min)            (Ch. 2)
     - énergie moyenne du gradient de Sobel (filtre passe-haut)                  (Ch. 3)
     - écart à la référence |I - I_ref| : moyenne et maximum                     (Ch. 2, soustraction
                                                                                   d'images = détection de défauts)
B. Caractéristiques GLOBALES à partir de descripteurs locaux ORB / SIFT          (Ch. 1)
     - sac de mots visuels : vocabulaire K-Means appris sur les images saines,
       histogramme normalisé des mots de l'image
     - erreur de quantification : distance de chaque descripteur à son mot le plus
       proche (moyenne, 95e centile) -> un motif jamais vu chez les images saines est loin
       de tous les mots du vocabulaire
     - appariement avec l'image de référence : proportion de bons appariements
       (test de ratio de Lowe), distance moyenne des appariements
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import MiniBatchKMeans

# --------------------------------------------------------------------------- #
# A. Caractéristiques par zone
# --------------------------------------------------------------------------- #

def zone_grid(size: int = 256, zone: int = 32, stride: int = 16) -> np.ndarray:
    """Coordonnées (y, x) du coin haut-gauche de chaque zone."""
    pos = np.arange(0, size - zone + 1, stride)
    return np.array([(y, x) for y in pos for x in pos])


def zone_feature_names(hist_bins: int = 16, residual: bool = True) -> list[str]:
    names = [f"hist_{k}" for k in range(hist_bins)] + ["moyenne", "ecart_type", "contraste", "sobel"]
    if residual:
        names += ["residu_moy", "residu_max"]
    return names


def zone_families(hist_bins: int = 16, residual: bool = True) -> dict[str, list[int]]:
    """Indices des colonnes de chaque famille (pour l'ablation)."""
    fam = {"histogramme": list(range(hist_bins)),
           "statistiques": [hist_bins, hist_bins + 1, hist_bins + 2],
           "sobel": [hist_bins + 3]}
    if residual:
        fam["residu"] = [hist_bins + 4, hist_bins + 5]
    return fam


def zone_features(img: np.ndarray, reference: np.ndarray | None = None, zone: int = 32,
                  stride: int = 16, hist_bins: int = 16) -> np.ndarray:
    """Matrice (n_zones, n_caractéristiques) pour une image prétraitée uint8."""
    def win(a):  # (ny, nx, zone, zone)
        return sliding_window_view(a, (zone, zone))[::stride, ::stride]

    w = win(img)
    ny, nx = w.shape[:2]
    flat = w.reshape(ny * nx, -1)
    # histogramme normalisé par zone
    bins = (flat.astype(np.int32) * hist_bins) >> 8
    hist = np.stack([(bins == k).mean(axis=1) for k in range(hist_bins)], axis=1)
    # statistiques d'intensité
    f = flat.astype(np.float32)
    mean, std = f.mean(1), f.std(1)
    mx, mn = f.max(1), f.min(1)
    contrast = (mx - mn) / (mx + mn + 1e-6)
    # gradient de Sobel (norme), moyenné par zone
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobel = win(np.sqrt(gx ** 2 + gy ** 2)).reshape(ny * nx, -1).mean(1)
    cols = [hist, mean[:, None] / 255, std[:, None] / 255, contrast[:, None], sobel[:, None] / 255]
    # écart à l'image de référence
    if reference is not None:
        r = win(np.abs(img.astype(np.float32) - reference.astype(np.float32))).reshape(ny * nx, -1)
        cols += [r.mean(1)[:, None] / 255, r.max(1)[:, None] / 255]
    return np.hstack(cols).astype(np.float32)


def zone_truth(mask: np.ndarray, zone: int = 32, stride: int = 16) -> np.ndarray:
    """Fraction de pixels défectueux dans chaque zone (évaluation de la localisation)."""
    return sliding_window_view(mask.astype(np.float32), (zone, zone))[::stride, ::stride].mean(axis=(-1, -2)).ravel()


# --------------------------------------------------------------------------- #
# B. Descripteurs locaux ORB / SIFT -> caractéristiques globales
# --------------------------------------------------------------------------- #

def make_detector(kind: str, nfeatures: int = 500):
    if kind == "orb":
        # seuil FAST abaissé : images sombres/peu contrastées (cf. notebook 02)
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7, edgeThreshold=15, patchSize=15)
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    raise ValueError(kind)


def _as_float(des: np.ndarray, kind: str) -> np.ndarray:
    """ORB : descripteur binaire 256 bits -> vecteur 0/1 (distance euclidienne² = distance de Hamming)."""
    if kind == "orb":
        return np.unpackbits(des, axis=1).astype(np.float32)
    return des.astype(np.float32)


@dataclass
class BoVW:
    """Sac de mots visuels sur descripteurs ORB ou SIFT."""
    kind: str = "orb"
    k: int = 64
    nfeatures: int = 500
    lowe_ratio: float = 0.75
    max_train_desc: int = 60000
    seed: int = 42
    kmeans: MiniBatchKMeans | None = field(default=None, repr=False)
    ref_des: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self._det = make_detector(self.kind, self.nfeatures)
        self._norm = cv2.NORM_HAMMING if self.kind == "orb" else cv2.NORM_L2

    def __getstate__(self):          # les objets OpenCV ne se sérialisent pas
        d = self.__dict__.copy(); d.pop("_det", None); return d

    def __setstate__(self, d):
        self.__dict__.update(d); self._det = make_detector(self.kind, self.nfeatures)
        self._norm = cv2.NORM_HAMMING if self.kind == "orb" else cv2.NORM_L2

    def describe(self, img: np.ndarray):
        kp, des = self._det.detectAndCompute(img, None)
        return kp, des

    def fit(self, images: list[np.ndarray], reference: np.ndarray | None = None) -> "BoVW":
        all_des = [d for d in (self.describe(im)[1] for im in images) if d is not None]
        X = _as_float(np.vstack(all_des), self.kind)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train_desc:
            X = X[rng.choice(len(X), self.max_train_desc, replace=False)]
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=self.seed, n_init=3,
                                      batch_size=4096).fit(X)
        if reference is not None:
            self.ref_des = self.describe(reference)[1]
        return self

    def names(self) -> list[str]:
        p = self.kind
        return ([f"{p}_mot_{i}" for i in range(self.k)] +
                [f"{p}_nb_points", f"{p}_quantif_moy", f"{p}_quantif_p95", f"{p}_app_ratio", f"{p}_app_dist"])

    def families(self, offset: int = 0) -> dict[str, list[int]]:
        p, k = self.kind, self.k
        return {f"{p}_sac_de_mots": list(range(offset, offset + k)),
                f"{p}_quantification": [offset + k + 1, offset + k + 2],
                f"{p}_appariement": [offset + k + 3, offset + k + 4]}

    def transform(self, img: np.ndarray) -> np.ndarray:
        kp, des = self.describe(img)
        n = 0 if des is None else len(des)
        hist = np.zeros(self.k, np.float32)
        q_mean = q_p95 = 0.0
        ratio, mdist = 0.0, 1.0
        if n:
            X = _as_float(des, self.kind)
            words = self.kmeans.predict(X)
            hist = np.bincount(words, minlength=self.k).astype(np.float32) / n
            d = np.linalg.norm(X - self.kmeans.cluster_centers_[words], axis=1)
            q_mean, q_p95 = float(d.mean()), float(np.percentile(d, 95))
            if self.ref_des is not None and len(self.ref_des) >= 2 and n >= 2:
                pairs = cv2.BFMatcher(self._norm).knnMatch(des, self.ref_des, k=2)
                pairs = [p for p in pairs if len(p) == 2]
                good = [p[0] for p in pairs if p[0].distance < self.lowe_ratio * p[1].distance]
                ratio = len(good) / n
                scale = 256.0 if self.kind == "orb" else 512.0   # distance max indicative
                mdist = float(np.mean([p[0].distance for p in pairs])) / scale if pairs else 1.0
        return np.concatenate([hist, [n / self.nfeatures, q_mean, q_p95, ratio, mdist]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Extracteur complet + cache disque
# --------------------------------------------------------------------------- #

@dataclass
class FeatureExtractor:
    zone: int = 32
    stride: int = 16
    hist_bins: int = 16
    use_residual: bool = True
    descriptors: tuple = ("orb", "sift")
    bovw_k: int = 64
    nfeatures: int = 500
    lowe_ratio: float = 0.75
    reference: np.ndarray | None = field(default=None, repr=False)
    bovw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, cfg: dict, category: str) -> "FeatureExtractor":
        f = resolved_features(cfg, category)
        residual = f.get("use_residual", True) and category not in cfg["preprocessing"].get("no_align_categories", [])
        return cls(zone=f.get("zone", 32), stride=f.get("stride", 16), hist_bins=f["hist_bins"],
                   use_residual=residual, descriptors=tuple(f.get("descriptors", ["orb", "sift"])),
                   bovw_k=f["bovw_k"], nfeatures=f.get("nfeatures", 500), lowe_ratio=f["lowe_ratio"])

    def fit(self, train_images: list[np.ndarray], reference: np.ndarray) -> "FeatureExtractor":
        """Apprend les vocabulaires visuels sur les images saines prétraitées."""
        self.reference = reference
        self.bovw = {d: BoVW(d, self.bovw_k, self.nfeatures, self.lowe_ratio).fit(train_images, reference)
                     for d in self.descriptors}
        return self

    def zones(self, img: np.ndarray) -> np.ndarray:
        return zone_features(img, self.reference if self.use_residual else None,
                             self.zone, self.stride, self.hist_bins)

    def global_(self, img: np.ndarray) -> np.ndarray:
        return np.concatenate([self.bovw[d].transform(img) for d in self.descriptors])

    def zone_names(self):
        return zone_feature_names(self.hist_bins, self.use_residual)

    def zone_families(self):
        return zone_families(self.hist_bins, self.use_residual)

    def global_names(self):
        return [n for d in self.descriptors for n in self.bovw[d].names()]

    def global_families(self):
        fam, off = {}, 0
        for d in self.descriptors:
            fam.update(self.bovw[d].families(off)); off += len(self.bovw[d].names())
        return fam


def resolved_features(cfg: dict, category: str) -> dict:
    """Section `features` + réglages propres à la catégorie (`features_per_category`)."""
    f = {k: v for k, v in cfg["features"].items()}
    f.update(cfg.get("features_per_category", {}).get(category, {}))
    return f


def _config_hash(cfg: dict, category: str) -> str:
    # réglages RÉSOLUS pour la catégorie : un réglage propre à une autre catégorie n'invalide pas ce cache
    P = cfg["preprocessing"]
    pre = {k: v for k, v in P.items() if k not in ("normalize_per_category", "denoise_per_category")}
    if category in P.get("normalize_per_category", {}):
        pre["normalize"] = P["normalize_per_category"][category]
    if category in P.get("denoise_per_category", {}):
        pre["denoise"] = P["denoise_per_category"][category]
    key = json.dumps({"cat": category, "pre": pre, "feat": resolved_features(cfg, category),
                      "data": cfg["data"], "v": 1}, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def compute_features(category: str, cfg: dict, cache_dir: str | Path | None = None,
                     verbose: bool = True) -> dict:
    """Prétraite + extrait les caractéristiques de train / val / test d'une catégorie.

    Résultat (mis en cache sur disque, clé = hash de la configuration) :
      {"pre": Preprocessor ajusté, "fx": FeatureExtractor ajusté,
       "<split>": {"zones": (n, n_zones, d), "global": (n, g), "labels", "defects", "paths",
                   "align_ncc", "zone_truth": (n, n_zones), "masks": (n, 256, 256) uint8}}
    """
    from tqdm.auto import tqdm
    from src.data_loader import PROJECT_ROOT, get_splits, load_image, load_mask
    from src.preprocessing import Preprocessor

    cache_dir = Path(cache_dir or PROJECT_ROOT / cfg["paths"]["models"] / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"features_{category}_{_config_hash(cfg, category)}.joblib"
    if path.exists():
        if verbose:
            print(f"[cache] {path.name}")
        return joblib.load(path)

    splits = get_splits(category, cfg)
    pre = Preprocessor.from_config(cfg, category)
    pre.fit([load_image(x.path) for x in splits["train"]])

    processed = {}
    for name, samples in splits.items():
        imgs, masks, ncc = [], [], []
        for x in tqdm(samples, desc=f"{category}/{name} prétraitement", disable=not verbose):
            raw = load_image(x.path)
            out, m, info = pre.transform_with_mask(raw, load_mask(x) if x.label else None)
            imgs.append(out); ncc.append(info.ncc)
            masks.append(m if m is not None else np.zeros_like(out))
        processed[name] = (imgs, masks, ncc)

    fx = FeatureExtractor.from_config(cfg, category).fit(processed["train"][0], pre.reference)
    result = {"pre": pre, "fx": fx, "category": category}
    for name, samples in splits.items():
        imgs, masks, ncc = processed[name]
        Z = np.stack([fx.zones(im) for im in tqdm(imgs, desc=f"{category}/{name} zones", disable=not verbose)])
        G = np.stack([fx.global_(im) for im in tqdm(imgs, desc=f"{category}/{name} ORB/SIFT", disable=not verbose)])
        result[name] = {
            "zones": Z, "global": G,
            "labels": np.array([x.label for x in samples]),
            "defects": np.array([x.defect for x in samples]),
            "paths": [str(x.path) for x in samples],
            "align_ncc": np.array(ncc, np.float32),
            "zone_truth": np.stack([zone_truth(m, fx.zone, fx.stride) for m in masks]).astype(np.float32),
            "masks": np.stack(masks).astype(np.uint8),
            "images": np.stack(imgs).astype(np.uint8),
        }
    tmp = path.with_suffix(".tmp")
    joblib.dump(result, tmp, compress=3)   # écriture atomique : pas de cache partiel en cas d'erreur
    tmp.replace(path)
    if verbose:
        print(f"[cache] écrit : {path.name}")
    return result
