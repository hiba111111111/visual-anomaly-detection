"""Étape 3 du pipeline : apprentissage non supervisé sur les images SAINES uniquement.

Chaîne de chaque modèle :  StandardScaler -> ACP (95 % de variance, Ch. 1) -> One-Class SVM ou Isolation Forest

- ZoneModel   : sur les caractéristiques par zone (225 zones / image)
    * strategy="position" : un modèle par position de zone (suppose un bon alignement)
    * strategy="shared"   : un seul modèle pour toutes les zones (insensible à la position)
- GlobalModel : sur les caractéristiques globales ORB / SIFT
- AnomalyDetector : combine les scores, calibre sur la validation saine, fixe le seuil.

Convention : score = -decision_function -> PLUS GRAND = PLUS ANORMAL.
Les scores sont ensuite standardisés (z-score) avec les images saines de validation,
ce qui rend comparables des positions / modèles différents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def make_estimator(kind: str, params: dict, seed: int = 42):
    if kind == "ocsvm":
        return OneClassSVM(kernel="rbf", nu=params.get("nu", 0.05), gamma=params.get("gamma", "scale"))
    if kind == "iforest":
        return IsolationForest(n_estimators=params.get("n_estimators", 200), max_samples="auto",
                               contamination="auto", random_state=seed, n_jobs=-1)
    raise ValueError(kind)


class ConstantModel:
    """Repli pour une zone dont toutes les caractéristiques sont constantes sur les images saines
    (ex. coin de fond uniformément blanc) : tout écart à cette valeur est anormal."""
    def fit(self, X):
        self.center_ = X.mean(0); return self

    def decision_function(self, X):
        return -np.linalg.norm(X - self.center_, axis=1) * 100.0


def make_pipeline_(kind: str, params: dict, pca_var: float = 0.95, seed: int = 42):
    # VarianceThreshold : retire les colonnes constantes (sinon division par zéro dans l'ACP)
    steps = [VarianceThreshold(1e-10), StandardScaler()]
    if pca_var:
        steps.append(PCA(n_components=pca_var, svd_solver="full", random_state=seed))
    steps.append(make_estimator(kind, params, seed))
    return make_pipeline(*steps)


def _raw_score(model, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def aggregate(zone_scores: np.ndarray, how: str = "max", topk: int = 3) -> np.ndarray:
    """Score d'image à partir des scores de zone (n_images, n_zones)."""
    if how == "max":
        return zone_scores.max(1)
    if how == "topk":
        return np.sort(zone_scores, axis=1)[:, -topk:].mean(1)
    if how == "mean":
        return zone_scores.mean(1)
    raise ValueError(how)


# --------------------------------------------------------------------------- #

@dataclass
class ZoneModel:
    strategy: str = "position"          # position | shared
    kind: str = "ocsvm"
    params: dict = field(default_factory=dict)
    cols: list | None = None            # colonnes (familles) utilisées ; None = toutes
    pca_var: float = 0.95
    max_shared: int = 15000             # sous-échantillonnage des zones pour le modèle partagé
    seed: int = 42

    def _X(self, Z):
        return Z if self.cols is None else Z[..., self.cols]

    def fit(self, Z: np.ndarray) -> "ZoneModel":
        """Z : (n_images, n_zones, d) caractéristiques des images saines d'apprentissage."""
        Z = self._X(Z)
        n, nz, d = Z.shape
        if self.strategy == "position":
            self.models_ = [self._fit_one(Z[:, j]) for j in range(nz)]
        else:
            X = Z.reshape(-1, d)
            if len(X) > self.max_shared:
                X = X[np.random.default_rng(self.seed).choice(len(X), self.max_shared, replace=False)]
            self.model_ = make_pipeline_(self.kind, self.params, self.pca_var, self.seed).fit(X)
        return self

    def _fit_one(self, X):
        if np.all(X.var(0) < 1e-10):
            return ConstantModel().fit(X)
        return make_pipeline_(self.kind, self.params, self.pca_var, self.seed).fit(X)

    def raw(self, Z: np.ndarray) -> np.ndarray:
        Z = self._X(Z)
        n, nz, d = Z.shape
        if self.strategy == "position":
            return np.stack([_raw_score(self.models_[j], Z[:, j]) for j in range(nz)], axis=1)
        return _raw_score(self.model_, Z.reshape(-1, d)).reshape(n, nz)

    def calibrate(self, Zval: np.ndarray) -> "ZoneModel":
        """Standardisation des scores de zone avec les images saines de validation."""
        r = self.raw(Zval)
        if self.strategy == "position":
            self.mu_, self.sd_ = r.mean(0), r.std(0) + 1e-6
        else:
            self.mu_, self.sd_ = r.mean(), r.std() + 1e-6
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        """Scores de zone standardisés (n_images, n_zones)."""
        return (self.raw(Z) - self.mu_) / self.sd_


@dataclass
class GlobalModel:
    kind: str = "ocsvm"
    params: dict = field(default_factory=dict)
    cols: list | None = None
    pca_var: float = 0.95
    seed: int = 42

    def _X(self, G):
        return G if self.cols is None else G[:, self.cols]

    def fit(self, G):
        self.model_ = make_pipeline_(self.kind, self.params, self.pca_var, self.seed).fit(self._X(G))
        return self

    def raw(self, G):
        return _raw_score(self.model_, self._X(G))


# --------------------------------------------------------------------------- #
# Ensembles issus de la validation croisée (un modèle par pli)
# --------------------------------------------------------------------------- #

@dataclass
class ZoneEnsemble:
    """Moyenne des scores bruts des modèles de zone des K plis.
    Calibration (mu_, sd_) estimée sur les scores HORS-PLI de toutes les images saines :
    chaque image saine est notée par un modèle qui ne l'a pas vue -> calibration non biaisée."""
    members: list
    strategy: str

    def raw(self, Z):
        return np.mean([m.raw(Z) for m in self.members], axis=0)

    def calibrate_from_oof(self, oof_raw: np.ndarray) -> "ZoneEnsemble":
        if self.strategy == "position":
            self.mu_, self.sd_ = oof_raw.mean(0), oof_raw.std(0) + 1e-6
        else:
            self.mu_, self.sd_ = oof_raw.mean(), oof_raw.std() + 1e-6
        return self

    def score(self, Z):
        return (self.raw(Z) - self.mu_) / self.sd_


@dataclass
class GlobalEnsemble:
    members: list

    def raw(self, G):
        return np.mean([m.raw(G) for m in self.members], axis=0)


# --------------------------------------------------------------------------- #

@dataclass
class AnomalyDetector:
    """Détecteur complet : scores de zone (+ score global), calibration et seuil.
    zone_model / global_model : ZoneModel ou ZoneEnsemble, GlobalModel ou GlobalEnsemble."""
    zone_model: object
    global_model: object = None
    agg: str = "max"
    topk: int = 3
    threshold_percentile: float = 99.0
    category: str = ""
    pre: object = None                  # Preprocessor ajusté (pour l'inférence sur image brute)
    fx: object = None                   # FeatureExtractor ajusté
    info: dict = field(default_factory=dict)

    def fit(self, Ztr, Gtr, Zval, Gval) -> "AnomalyDetector":
        self.zone_model.fit(Ztr).calibrate(Zval)
        if self.global_model is not None:
            self.global_model.fit(Gtr)
        # calibration des composantes au niveau image sur la validation saine
        comps = self._components(Zval, Gval)
        self.comp_mu_ = {k: v.mean() for k, v in comps.items()}
        self.comp_sd_ = {k: v.std() + 1e-6 for k, v in comps.items()}
        val_scores = self.score_features(Zval, Gval)[0]
        self.threshold_ = float(np.percentile(val_scores, self.threshold_percentile))
        return self

    def _components(self, Z, G) -> dict:
        comps = {"zones": aggregate(self.zone_model.score(Z), self.agg, self.topk)}
        if self.global_model is not None:
            comps["global"] = self.global_model.raw(G)
        return comps

    def score_features(self, Z, G) -> tuple[np.ndarray, np.ndarray]:
        """-> (score d'image, scores de zone standardisés)."""
        zs = self.zone_model.score(Z)
        comps = {"zones": aggregate(zs, self.agg, self.topk)}
        if self.global_model is not None:
            comps["global"] = self.global_model.raw(G)
        z = [(comps[k] - self.comp_mu_[k]) / self.comp_sd_[k] for k in comps]
        return np.mean(z, axis=0), zs

    def predict_features(self, Z, G) -> np.ndarray:
        return (self.score_features(Z, G)[0] > self.threshold_).astype(int)

    # ---------------- sauvegarde ---------------- #
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        return joblib.load(path)
