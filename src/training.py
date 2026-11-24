"""Sélection des hyperparamètres (sans vrai défaut) et entraînement final par validation croisée.

Protocole (aucune image de test n'intervient dans les choix) :
  1. Images SAINES = train + val (ex. 209 bouteilles). Découpage en K = 5 plis.
  2. Anomalies SYNTHÉTIQUES (1 à 5 % de l'image) créées sur chaque image saine.
  3. Pour chaque configuration candidate et chaque pli k : apprentissage sur les 4 autres plis,
     notation des images saines du pli k ET de leurs versions dégradées (scores « hors-pli »).
  4. Critère de sélection = AUROC hors-pli « saine » vs « synthétique ».
  5. Le détecteur retenu = ensemble des K modèles (moyenne des scores). Sa calibration
     (z-score par zone) et son SEUIL (centile `threshold_percentile` des scores hors-pli
     des images saines) utilisent toutes les images saines, et non 42 images de validation.
  Le TEST est calculé pour chaque configuration uniquement pour le rapport.
"""
from __future__ import annotations

import gc as _gc
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.data_loader import PROJECT_ROOT
from src.features import compute_features, zone_truth
from src.models import AnomalyDetector, GlobalEnsemble, GlobalModel, ZoneEnsemble, ZoneModel, aggregate
from src.synthetic import make_synthetic_set


# --------------------------------------------------------------------------- #
# Grille de configurations
# --------------------------------------------------------------------------- #

def feature_sets(cfg: dict, fx) -> dict[str, list[int]]:
    """Jeux de familles de caractéristiques de zone à comparer (ceux disponibles pour la catégorie)."""
    fam = fx.zone_families()
    out = {}
    for name, families in cfg["model"]["zone_feature_sets"].items():
        if families == "all":
            cols = sorted(i for v in fam.values() for i in v)
        else:
            cols = sorted(i for f in families if f in fam for i in fam[f])
        if cols and cols not in out.values():
            out[name] = cols
    return out


def _estimators(cfg: dict) -> list[tuple[str, dict]]:
    m = cfg["model"]
    out = [("ocsvm", {"nu": nu, "gamma": g}) for nu, g in itertools.product(m["ocsvm"]["nu"], m["ocsvm"]["gamma"])]
    out += [("iforest", {"n_estimators": n}) for n in m["iforest"]["n_estimators"]]
    return out


def zone_candidates(cfg: dict, fx) -> list[dict]:
    return [{"features": fs, "cols": cols, "strategy": st, "kind": k, "params": p}
            for fs, cols in feature_sets(cfg, fx).items()
            for st in cfg["model"]["zone_strategies"]
            for k, p in _estimators(cfg)]


def global_candidates(cfg: dict) -> list[dict]:
    return [{"kind": k, "params": p} for k, p in _estimators(cfg)]


def describe(c: dict) -> str:
    p = ", ".join(f"{k}={v}" for k, v in c["params"].items())
    fs = f"{c['features']} | " if "features" in c else ""
    return f"{fs}{c.get('strategy', 'global')} | {c['kind']}({p})"


def make_zone_model(c: dict, cfg: dict) -> ZoneModel:
    return ZoneModel(c["strategy"], c["kind"], c["params"], cols=c["cols"],
                     pca_var=cfg["features"]["pca_variance"], seed=cfg["data"]["seed"])


def make_global_model(c: dict, cfg: dict) -> GlobalModel:
    return GlobalModel(c["kind"], c["params"], pca_var=cfg["features"]["pca_variance"], seed=cfg["data"]["seed"])


# --------------------------------------------------------------------------- #
# Données saines + synthétiques
# --------------------------------------------------------------------------- #

def healthy_and_synthetic(feats: dict, category: str, cfg: dict) -> dict:
    """Regroupe train + val (images saines) et crée leurs versions dégradées."""
    fx = feats["fx"]
    H = {k: np.concatenate([feats["train"][k], feats["val"][k]]) for k in ("zones", "global", "images")}
    on_object = category not in cfg["preprocessing"].get("no_align_categories", [])
    imgs, masks, kinds, base = make_synthetic_set(H["images"], seed=cfg["data"]["seed"], on_object=on_object,
                                                  per_image=cfg["model"].get("synthetic_per_image", 1))
    S = {"images": imgs, "masks": masks, "kinds": kinds, "base": base,
         "zones": np.stack([fx.zones(im) for im in imgs]),
         "global": np.stack([fx.global_(im) for im in imgs]),
         "zone_truth": np.stack([zone_truth(m, fx.zone, fx.stride) for m in masks])}
    return {"H": H, "S": S}


def make_folds(n: int, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    k = cfg["model"].get("cv_folds", 5)
    return list(KFold(k, shuffle=True, random_state=cfg["data"]["seed"]).split(np.arange(n)))


def cross_val_scores(make, X_h, X_s, base, folds, X_test=None, keep_members=False):
    """Scores bruts hors-pli des images saines (X_h) et synthétiques (X_s), + moyenne des modèles sur le test.
    Une image synthétique est notée par le modèle du pli de son image saine d'origine.
    keep_members=False : chaque modèle est libéré après usage (mémoire ; ex. 961 forêts x 5 plis)."""
    oof_h = np.zeros(X_h.shape[:2] if X_h.ndim == 3 else X_h.shape[:1])
    oof_s = np.zeros(X_s.shape[:2] if X_s.ndim == 3 else X_s.shape[:1])
    members, te = [], []
    for tr, va in folds:
        m = make().fit(X_h[tr])
        oof_h[va] = m.raw(X_h[va])
        sel = np.isin(base, va)
        oof_s[sel] = m.raw(X_s[sel])
        if X_test is not None:
            te.append(m.raw(X_test))
        if keep_members:
            members.append(m)
        del m
    return oof_h, oof_s, (np.mean(te, axis=0) if te else None), members


def _calib(raw_h: np.ndarray, strategy: str):
    if strategy == "position":
        return raw_h.mean(0), raw_h.std(0) + 1e-6
    return raw_h.mean(), raw_h.std() + 1e-6


def _auc(neg, pos) -> float:
    return roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])


# --------------------------------------------------------------------------- #
# Sélection
# --------------------------------------------------------------------------- #

def run_selection(category: str, cfg: dict, verbose: bool = True) -> dict:
    feats = compute_features(category, cfg, verbose=verbose)
    data = healthy_and_synthetic(feats, category, cfg)
    H, S, te = data["H"], data["S"], feats["test"]
    y_te = te["labels"]
    folds = make_folds(len(H["zones"]), cfg)
    k_top = cfg["inference"]["topk_zones"]
    aggs = [("max", 1), (f"top{k_top}", k_top)]

    # ---------- 1. modèles de zone ----------
    rows, cands = [], zone_candidates(cfg, feats["fx"])
    for i, c in enumerate(cands):
        t0 = time.time()
        oh, os_, ot, _ = cross_val_scores(lambda: make_zone_model(c, cfg), H["zones"], S["zones"], S["base"],
                                          folds, te["zones"])
        mu, sd = _calib(oh, c["strategy"])
        zh, zs, zt = (oh - mu) / sd, (os_ - mu) / sd, (ot - mu) / sd
        hit = float(np.mean(S["zone_truth"][np.arange(len(zs)), zs.argmax(1)] > 0))
        for agg_name, k in aggs:
            how = "max" if agg_name == "max" else "topk"
            rows.append({"id": i, "caractéristiques": c["features"], "stratégie": c["strategy"],
                         "modèle": c["kind"], "paramètres": str(c["params"]), "agrégation": agg_name,
                         "AUROC synthétique (sélection)": _auc(aggregate(zh, how, k), aggregate(zs, how, k)),
                         "zone max sur le défaut synth.": hit,
                         "AUROC test (rapport)": roc_auc_score(y_te, aggregate(zt, how, k)),
                         "temps (s)": time.time() - t0})
        _gc.collect()
        if verbose:
            best = max(rows[-1]["AUROC synthétique (sélection)"], rows[-2]["AUROC synthétique (sélection)"])
            print(f"[{category}] zone {i + 1:2d}/{len(cands)} {describe(c):60s} synth={best:.3f} "
                  f"zone-max-sur-défaut={hit:.0%} ({time.time() - t0:.0f}s)", flush=True)
    df_zone = pd.DataFrame(rows)

    # ---------- 2. modèles globaux ORB / SIFT ----------
    rows, gcands, gcache = [], global_candidates(cfg), {}
    for i, c in enumerate(gcands):
        oh, os_, ot, _ = cross_val_scores(lambda: make_global_model(c, cfg), H["global"], S["global"], S["base"],
                                          folds, te["global"])
        gcache[i] = (oh, os_, ot)
        rows.append({"id": i, "modèle": c["kind"], "paramètres": str(c["params"]),
                     "AUROC synthétique (sélection)": _auc(oh, os_), "AUROC test (rapport)": roc_auc_score(y_te, ot)})
    df_glob = pd.DataFrame(rows)

    # ---------- 3. choix final ----------
    best_z = df_zone.sort_values(["AUROC synthétique (sélection)", "temps (s)"], ascending=[False, True]).iloc[0]
    best_g = df_glob.sort_values("AUROC synthétique (sélection)", ascending=False).iloc[0]
    zc, gc = cands[int(best_z["id"])], gcands[int(best_g["id"])]
    how = "max" if best_z["agrégation"] == "max" else "topk"

    oh, os_, ot, members = cross_val_scores(lambda: make_zone_model(zc, cfg), H["zones"], S["zones"], S["base"],
                                            folds, te["zones"], keep_members=True)
    zone_ens = ZoneEnsemble(members, zc["strategy"]).calibrate_from_oof(oh)
    mu, sd = zone_ens.mu_, zone_ens.sd_
    comp_h = {"zones": aggregate((oh - mu) / sd, how, k_top)}
    comp_s = {"zones": aggregate((os_ - mu) / sd, how, k_top)}
    comp_t = {"zones": aggregate((ot - mu) / sd, how, k_top)}
    gh, gs, gt = gcache[int(best_g["id"])]

    def combine(use_global):
        ch, cs, ct = dict(comp_h), dict(comp_s), dict(comp_t)
        keys = ["zones"]
        if use_global:
            ch["global"], cs["global"], ct["global"] = gh, gs, gt
            keys.append("global")
        m_ = {k: float(ch[k].mean()) for k in keys}
        s_ = {k: float(ch[k].std() + 1e-6) for k in keys}
        f = lambda d: np.mean([(d[k] - m_[k]) / s_[k] for k in keys], axis=0)
        return f(ch), f(cs), f(ct), m_, s_

    combos = {}
    for use_global in (False, True):
        sh, ss, st, m_, s_ = combine(use_global)
        combos[use_global] = {"synth": _auc(sh, ss), "test": roc_auc_score(y_te, st),
                              "scores_sains": sh, "scores_test": st, "mu": m_, "sd": s_}
    use_global = combos[True]["synth"] > combos[False]["synth"] + 0.005     # le global doit apporter un gain net
    best = combos[use_global]

    global_ens = None
    if use_global:
        global_ens = GlobalEnsemble(cross_val_scores(lambda: make_global_model(gc, cfg), H["global"], S["global"],
                                                     S["base"], folds, keep_members=True)[3])
    det = AnomalyDetector(zone_model=zone_ens, global_model=global_ens, agg=how, topk=k_top,
                          threshold_percentile=cfg["model"]["threshold_percentile"], category=category,
                          pre=feats["pre"], fx=feats["fx"])
    det.comp_mu_, det.comp_sd_ = best["mu"], best["sd"]
    det.threshold_ = float(np.percentile(best["scores_sains"], det.threshold_percentile))
    det.info = {"zone": describe(zc), "agrégation": best_z["agrégation"],
                "global": describe(gc) if use_global else None,
                "AUROC synthétique": float(best["synth"]), "AUROC test": float(best["test"]),
                "images saines (calibration)": int(len(H["zones"])), "plis": len(folds),
                "seuil": f"{det.threshold_percentile:g}e centile des scores hors-pli des images saines",
                "combinaisons": {("zones + global" if k else "zones seules"):
                                 {"synth": float(v["synth"]), "test": float(v["test"])} for k, v in combos.items()}}
    return {"category": category, "detector": det, "zone_table": df_zone, "global_table": df_glob,
            "combos": combos, "feats": feats, "synthetic": S, "healthy": H}


def train_category(category: str, cfg: dict, verbose: bool = True) -> dict:
    """Sélection + sauvegarde du détecteur et des tableaux de sélection."""
    res = run_selection(category, cfg, verbose)
    out_dir = PROJECT_ROOT / cfg["paths"]["results"] / "04_apprentissage"
    out_dir.mkdir(parents=True, exist_ok=True)
    res["zone_table"].to_csv(out_dir / f"selection_zones_{category}.csv", index=False)
    res["global_table"].to_csv(out_dir / f"selection_global_{category}.csv", index=False)
    path = PROJECT_ROOT / cfg["paths"]["models"] / f"{category}_detector.joblib"
    res["detector"].save(path)
    res["model_path"] = path
    if verbose:
        print(f"[{category}] retenu : {res['detector'].info}", flush=True)
        print(f"[{category}] sauvegardé : {path}", flush=True)
    return res
