"""Étape 1 du pipeline : prétraitement.

Chaîne appliquée à chaque image (même traitement en apprentissage et en inférence) :
    1. niveaux de gris + redimensionnement INTER_AREA        (Ch. 2 : anti-repliement)
    2. débruitage (filtre médian 3x3 par défaut)              (Ch. 3 : bruit impulsionnel)
    3. normalisation photométrique                            (Ch. 2 : histogrammes, CLAHE)
    4. alignement spatial sur une image de référence          (Ch. 1 : ORB/Hamming ;
                                                               Ch. 2 : translation, rotation, coord. homogènes,
                                                               interpolation bilinéaire ;
                                                               Ch. 3 : région, centroïde)
L'image de référence est la médiane pixel à pixel des images saines alignées.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 1-2. Conversion, redimensionnement, débruitage
# --------------------------------------------------------------------------- #

def to_gray_resized(img: np.ndarray, size: int = 256) -> np.ndarray:
    """BGR/gris -> gris uint8 size x size (INTER_AREA : moyenne locale = filtre anti-repliement)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    """Filtres de lissage du Ch. 3."""
    if method in (None, "none") or ksize <= 1:
        return img
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "mean":
        return cv2.blur(img, (ksize, ksize))
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=ksize + 2, sigmaColor=25, sigmaSpace=3)
    raise ValueError(f"Filtre inconnu : {method}")


# --------------------------------------------------------------------------- #
# 3. Normalisation photométrique
# --------------------------------------------------------------------------- #

def normalize_photometric(img: np.ndarray, method: str = "zscore", clahe_clip: float = 2.0,
                          target_mean: float = 128.0, target_std: float = 48.0) -> np.ndarray:
    """Compense les variations d'éclairage entre acquisitions.

    - none     : aucune correction
    - equalize : égalisation d'histogramme globale  T(k) = (L-1)*CDF(k)       (Ch. 2)
    - clahe    : égalisation adaptative à contraste limité                    (Ch. 2)
    - zscore   : transformation affine g = a*f + b ramenant moyenne/écart-type
                 à des valeurs cibles (transformation ponctuelle linéaire, Ch. 2)
    - minmax   : étirement linéaire des centiles 1-99 sur [0, 255]
    - localnorm: normalisation LOCALE de l'éclairage (textures) : on soustrait l'éclairage estimé par
                 un flou gaussien large (filtre passe-bas, Ch. 3), puis on divise par le contraste
                 local. Corrige un éclairage non uniforme (dégradé, vignettage), ce que ne fait pas
                 une transformation globale (cf. Ch. 2 : Otsu ne corrige pas un éclairage variable).
    """
    if method in (None, "none"):
        return img
    if method == "equalize":
        return cv2.equalizeHist(img)
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img)
    f = img.astype(np.float32)
    if method == "zscore":
        g = (f - f.mean()) / (f.std() + 1e-6) * target_std + target_mean
    elif method == "localnorm":
        sigma = img.shape[0] / 8.0
        background = cv2.GaussianBlur(f, (0, 0), sigma)
        d = f - background
        local_std = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma)) + 1.0
        g = d / local_std * target_std + target_mean
    elif method == "minmax":
        lo, hi = np.percentile(f, (1, 99))
        g = (f - lo) / max(hi - lo, 1e-6) * 255.0
    else:
        raise ValueError(f"Normalisation inconnue : {method}")
    return np.clip(g, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Outils : masque objet (Otsu), moments, corrélation
# --------------------------------------------------------------------------- #

def otsu_mask(img: np.ndarray) -> np.ndarray:
    """Masque binaire (0/1) de la plus grande région, seuil d'Otsu (Ch. 2 / Ch. 3)."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # l'objet est la région qui ne touche pas majoritairement le bord
    if m[[0, -1], :].mean() + m[:, [0, -1]].mean() > 255:   # bord surtout blanc -> inverser
        m = 255 - m
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return (out > 0).astype(np.uint8)


def centroid(mask: np.ndarray) -> tuple[float, float]:
    """Centroïde de la région (moments d'ordre 0 et 1, Ch. 3)."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        h, w = mask.shape
        return w / 2, h / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Corrélation croisée normalisée entre deux images (1 = identiques)."""
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def _h(M: np.ndarray) -> np.ndarray:
    """Matrice affine 2x3 -> 3x3 (coordonnées homogènes, Ch. 2)."""
    return np.vstack([M, [0, 0, 1]])


def warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Applique une transformation affine, interpolation bilinéaire, bords répliqués (Ch. 2 / Ch. 3)."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# 4. Alignement spatial
# --------------------------------------------------------------------------- #

@dataclass
class AlignInfo:
    method: str                 # méthode réellement utilisée
    ok: bool                    # alignement jugé réussi
    ncc: float                  # corrélation avec la référence après alignement
    angle: float = 0.0          # rotation estimée (degrés)
    shift: tuple = (0.0, 0.0)   # translation estimée (pixels)
    n_matches: int = 0          # appariements ORB retenus (inliers RANSAC)
    matrix: np.ndarray | None = field(default=None, repr=False)  # transformation 2x3 appliquée


@dataclass
class Aligner:
    """Recalage d'une image sur une référence.

    method :
      - "none"      : pas d'alignement (textures)
      - "orb"       : ORB -> appariement Hamming + test de ratio de Lowe -> RANSAC
                      (estimateAffinePartial2D : rotation + translation + échelle)
      - "rotsearch" : centroïde de la région Otsu ramené au centre (translation),
                      puis recherche de l'angle maximisant la NCC avec la référence
      - "orb+rotsearch" : ORB, et repli sur rotsearch si ORB échoue
    ecc_refine : affinage final par maximisation de corrélation (cv2.findTransformECC).
    """
    method: str = "orb+rotsearch"
    min_matches: int = 10
    lowe_ratio: float = 0.75
    angle_step: int = 5
    ecc_refine: bool = True
    ok_ncc: float = 0.5
    reference: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # seuil FAST abaissé (7 au lieu de 20) : sinon 0 point clé sur les images sombres (hazelnut)
        self._detector = cv2.ORB_create(nfeatures=1000, fastThreshold=7, edgeThreshold=15, patchSize=15)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_kp = self._ref_des = None

    def __getstate__(self):          # les objets OpenCV (ORB, BFMatcher) ne se sérialisent pas
        d = self.__dict__.copy()
        for k in ("_detector", "_bf", "_ref_kp", "_ref_des"):
            d.pop(k, None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__post_init__()
        if self.reference is not None:
            self.set_reference(self.reference)

    # ---------------- référence ---------------- #
    def set_reference(self, ref: np.ndarray) -> None:
        self.reference = ref
        self._ref_kp, self._ref_des = self._detector.detectAndCompute(ref, None)

    def fit(self, images: list[np.ndarray], n_iter: int = 2) -> np.ndarray:
        """Construit la référence : médiane des images alignées, raffinée sur n_iter itérations."""
        if self.method == "none":
            self.reference = np.median(np.stack(images), axis=0).astype(np.uint8)
            return self.reference
        # initialisation : image la plus « typique » (la plus corrélée à la médiane brute centrée)
        centered = [self._center(im)[0] for im in images]
        med = np.median(np.stack(centered), axis=0).astype(np.uint8)
        start = int(np.argmax([ncc(im, med) for im in centered]))
        self.set_reference(centered[start])
        for _ in range(n_iter):
            aligned = [self.transform(im)[0] for im in images]
            self.set_reference(np.median(np.stack(aligned), axis=0).astype(np.uint8))
        return self.reference

    # ---------------- méthodes ---------------- #
    def _center(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translation amenant le centroïde de l'objet au centre de l'image."""
        cx, cy = centroid(otsu_mask(img))
        h, w = img.shape
        M = np.float32([[1, 0, w / 2 - cx], [0, 1, h / 2 - cy]])
        return warp(img, M), M

    def _orb_align(self, img: np.ndarray):
        kp, des = self._detector.detectAndCompute(img, None)
        if des is None or self._ref_des is None or len(kp) < self.min_matches:
            return None, 0
        pairs = self._bf.knnMatch(des, self._ref_des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.lowe_ratio * p[1].distance]
        if len(good) < self.min_matches:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self._ref_kp[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        n_in = int(inl.sum()) if inl is not None else 0
        if M is None or n_in < self.min_matches:
            return None, n_in
        scale = np.hypot(M[0, 0], M[1, 0])
        if not 0.8 < scale < 1.25:          # transformation aberrante
            return None, n_in
        return M.astype(np.float32), n_in

    def _rotsearch(self, img: np.ndarray) -> np.ndarray:
        """Centrage puis recherche exhaustive de l'angle (grossier puis fin) par NCC."""
        _, T = self._center(img)
        h, w = img.shape
        small = 64 / w
        ref_s = cv2.resize(self.reference, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
        img_c = cv2.resize(warp(img, T), None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
        c = (img_c.shape[1] / 2, img_c.shape[0] / 2)

        def score(a):
            return ncc(warp(img_c, cv2.getRotationMatrix2D(c, a, 1.0)), ref_s)

        best = max(range(0, 360, self.angle_step), key=score)
        fine = np.arange(best - self.angle_step, best + self.angle_step + 0.5, 1.0)
        best = float(max(fine, key=score))
        R = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1.0)
        return (_h(R) @ _h(T))[:2].astype(np.float32)

    def _ecc(self, img: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Affinage euclidien (rotation + translation) par maximisation de corrélation.
        ECC estime W tel que img(W(x)) ~ ref(x) : c'est l'inverse de M (img -> ref)."""
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            W0 = cv2.invertAffineTransform(M).astype(np.float32)
            _, W = cv2.findTransformECC(self.reference.astype(np.float32), img.astype(np.float32),
                                        W0, cv2.MOTION_EUCLIDEAN, crit, None, 5)
            M2 = cv2.invertAffineTransform(W).astype(np.float32)
            # garde-fou : ne garder l'affinage que s'il améliore la corrélation
            return M2 if ncc(warp(img, M2), self.reference) >= ncc(warp(img, M), self.reference) else M
        except cv2.error:
            return M

    # ---------------- application ---------------- #
    def transform(self, img: np.ndarray) -> tuple[np.ndarray, AlignInfo]:
        if self.method == "none":
            I = np.float32([[1, 0, 0], [0, 1, 0]])
            return img, AlignInfo("none", True, ncc(img, self.reference) if self.reference is not None else 1.0, matrix=I)
        if self.reference is None:
            raise RuntimeError("Référence absente : appeler fit() d'abord.")
        M, used, n = None, self.method, 0
        if "orb" in self.method:
            M, n = self._orb_align(img)
            used = "orb"
        if M is None and "rotsearch" in self.method:
            M, used = self._rotsearch(img), "rotsearch"
        if M is None:                                   # ORB seul et échec
            I = np.float32([[1, 0, 0], [0, 1, 0]])
            return img, AlignInfo("orb-échec", False, ncc(img, self.reference), n_matches=n, matrix=I)
        if self.ecc_refine:
            M = self._ecc(img, M)
        out = warp(img, M)
        s = ncc(out, self.reference)
        angle = -float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))   # convention OpenCV (antihoraire > 0)
        return out, AlignInfo(used, s >= self.ok_ncc, s, angle, (float(M[0, 2]), float(M[1, 2])), n, M)


# --------------------------------------------------------------------------- #
# Chaîne complète
# --------------------------------------------------------------------------- #

@dataclass
class Preprocessor:
    """Chaîne complète paramétrée par config.yaml (section preprocessing)."""
    size: int = 256
    denoise_method: str = "median"
    ksize: int = 3
    normalize: str = "zscore"
    clahe_clip: float = 2.0
    align_method: str = "orb+rotsearch"
    aligner: Aligner = field(default=None)

    def __post_init__(self):
        if self.aligner is None:
            self.aligner = Aligner(method=self.align_method)

    @classmethod
    def from_config(cls, cfg: dict, category: str) -> "Preprocessor":
        p = cfg["preprocessing"]
        method = p.get("align_method", "orb+rotsearch") if p.get("align", True) else "none"
        if category in p.get("no_align_categories", []):
            method = "none"
        method = p.get("align_method_per_category", {}).get(category, method)
        normalize = p.get("normalize_per_category", {}).get(category, p.get("normalize", "zscore"))
        denoise = p.get("denoise_per_category", {}).get(category, p.get("denoise", "median"))
        return cls(size=p["image_size"], denoise_method=denoise,
                   ksize=p["median_ksize"], normalize=normalize,
                   clahe_clip=p.get("clahe_clip", 2.0), align_method=method)

    def base(self, img: np.ndarray) -> np.ndarray:
        """Étapes 1 à 3 (sans alignement)."""
        g = to_gray_resized(img, self.size)
        g = denoise(g, self.denoise_method, self.ksize)
        return normalize_photometric(g, self.normalize, self.clahe_clip)

    def fit(self, images: list[np.ndarray]) -> "Preprocessor":
        """Construit l'image de référence à partir des images saines (brutes)."""
        self.aligner.fit([self.base(im) for im in images])
        return self

    @property
    def reference(self) -> np.ndarray:
        return self.aligner.reference

    def transform(self, img: np.ndarray) -> tuple[np.ndarray, AlignInfo]:
        return self.aligner.transform(self.base(img))

    def transform_with_mask(self, img: np.ndarray, mask: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray | None, AlignInfo]:
        """Prétraite l'image et applique la même géométrie au masque vérité terrain (évaluation pixel).
        Masque : INTER_NEAREST pour garder des valeurs binaires (Ch. 2)."""
        out, info = self.transform(img)
        if mask is None:
            return out, None, info
        m = cv2.resize(mask.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        m = cv2.warpAffine(m, info.matrix, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=0)
        return out, m, info
