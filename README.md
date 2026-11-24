# Sujet 16 — Visual Anomaly Detection (Image)

Détection d'anomalies visuelles **semi-supervisée** sur le jeu **MVTec AD** (catégories `bottle`, `hazelnut`, `carpet`) :
les modèles n'apprennent que sur des images **saines**, puis signalent tout écart majeur à la distribution normale comme anomalie,
et le localisent.

Cours de Vision par Ordinateur (S9). Pipeline imposé par le cahier des charges :

| Phase | Réalisation | Cours |
|---|---|---|
| 1. Prétraitement | niveaux de gris, 256×256 (`INTER_AREA`) ; **filtre médian 3×3** ; normalisation **locale** de l'éclairage (texture) ; **alignement spatial** (centroïde de la région Otsu, puis recherche de rotation par corrélation) | Ch. 2, Ch. 3 |
| 2. Caractéristiques | **histogrammes d'intensité par zone** (zones de 32 ou 16 px qui se chevauchent), moyenne, écart-type, contraste, Sobel, **résidu \|I − I_ref\|** ; **ORB / SIFT** en sac de mots visuels | Ch. 1, 2, 3 |
| 3. Apprentissage | **One-Class SVM** et **Isolation Forest** (scikit-learn) sur les seules images saines ; ACP ; validation croisée en 5 plis ; choix des réglages par **anomalies synthétiques** (aucun vrai défaut utilisé) | Ch. 1 |
| 4. Reconnaissance | score > seuil → **ANOMALIE** ; carte de chaleur ; contours, boîte englobante, aire, centroïde du défaut | Ch. 3 |

---

## Résultats (test MVTec AD)

Aucun vrai défaut n'est utilisé pour entraîner, choisir les réglages ou fixer le seuil
(seuil = 95e centile des scores hors-pli des images saines → 5 % de fausses alarmes visées).

| Catégorie | AUROC image | AP | F1 | Défauts détectés | Sains rejetés à tort | AUROC pixel |
|---|---|---|---|---|---|---|
| bottle | **0,990** | 0,997 | **0,961** | 61/63 | 3/20 | **0,954** |
| hazelnut | 0,846 | 0,914 | 0,803 | 51/70 | 6/40 | 0,886 |
| carpet | 0,779 | 0,923 | 0,560 | 35/89 | 1/28 | 0,771 |

Les défauts grands ou contrastés sont très bien détectés (AUROC de 0,92 à 1,00) ; ceux qui couvrent moins de 1 % de l'image
le sont mal (de 0,67 à 0,72). L'analyse complète, l'étude d'ablation et les limites se trouvent dans `notebooks/06_evaluation.ipynb`.

Configuration finale retenue (`config.yaml`) :

| | Débruitage | Normalisation | Alignement | Zones | Modèle |
|---|---|---|---|---|---|
| bottle | médian 3×3 | aucune | rotsearch | 32 px | OC-SVM par position |
| hazelnut | médian 3×3 | aucune | rotsearch | 16 px | OC-SVM par position |
| carpet | aucun | locale (`localnorm`) | aucun (texture) | 32 px | OC-SVM partagé |

---

## Installation (Windows)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> ⚠️ **Smart App Control (Windows 11)** bloque les DLL des versions très récentes de certaines bibliothèques
> (`ImportError: DLL load failed … Une stratégie de contrôle d'application a bloqué ce fichier`).
> Les versions figées dans `requirements.txt` (**opencv-python 4.10.0.84**, **scikit-learn 1.5.2**) fonctionnent :
> **ne pas les mettre à jour** sans vérifier l'import.

Données : télécharger MVTec AD (Kaggle ou site officiel de MVTec) et placer chaque catégorie dans `data/mvtec/<catégorie>/`
(`train/good`, `test/<type>`, `ground_truth/<type>`). Licence CC BY-NC-SA 4.0 : citer
*Bergmann et al., « MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection », CVPR 2019*.

---

## Utilisation

```
# Entraînement (images saines uniquement) -> models/<catégorie>_detector.joblib
.venv\Scripts\python.exe train.py --category bottle
.venv\Scripts\python.exe train.py --all

# Inférence sur des images brutes -> décision + image annotée dans results/predictions/
.venv\Scripts\python.exe predict.py --category bottle --image data\mvtec\bottle\test\broken_large\000.png
.venv\Scripts\python.exe predict.py --category carpet --image chemin\vers\un\dossier

# Évaluation (métriques) et étude d'ablation (longue : réentraîne des variantes)
.venv\Scripts\python.exe evaluate.py --all
.venv\Scripts\python.exe evaluate.py --all --ablation
```

Exemple de sortie de `predict.py` :

```
000.png   ANOMALIE  score   7.14 (seuil 2.04)   305 ms
    région 1: boîte x=279 y=281 560x619 px | aire 23.85 % | centroïde (576, 631) | z max 17.9
```

Durées indicatives (CPU) : environ 0,1 à 1,4 s par image en inférence ; de 5 à 40 minutes d'entraînement par catégorie
(hazelnut est la plus longue : 961 zones). Les caractéristiques sont mises en cache dans `models/cache/`.

---

## Organisation

```
config.yaml                 tous les paramètres (prétraitement, caractéristiques, grille de modèles, seuil)
train.py / predict.py / evaluate.py
src/
  data_loader.py            lecture de MVTec AD, découpage apprentissage / validation
  preprocessing.py          gris, débruitage, normalisation, alignement, image de référence
  features.py               caractéristiques par zone + ORB/SIFT (sac de mots), cache disque
  synthetic.py              anomalies synthétiques (tache, rayure, fissure, copier-coller)
  models.py                 OC-SVM / Isolation Forest, modèles par zone, ensembles, détecteur
  training.py               validation croisée, sélection sans vrai défaut, seuil
  inference.py              décision, carte de chaleur, régions (boîte, aire, centroïde)
  evaluation.py             métriques image / défaut / pixel, étude d'ablation
notebooks/
  00_synthese.ipynb         vue d'ensemble pour la soutenance
  01_exploration … 06_evaluation   une étape du projet par notebook (exécutés, avec conclusions)
tools/                      génération et exécution des notebooks (voir tools/README.md)
models/                     détecteurs entraînés (*.joblib) et cache des caractéristiques
results/                    figures et tableaux de chaque notebook, prédictions
```

---

## Démarche et choix (résumé)

1. **Exploration** (01) : un histogramme global ne suffit pas (AUROC ≈ 0,5) ; l'éclairage diffère entre images
   d'apprentissage et de test ; la moitié des défauts couvrent moins de 1/64 de l'image → zones petites et qui se chevauchent.
2. **Prétraitement** (02) : le médian 3×3 est le meilleur filtre contre le bruit impulsionnel (PSNR 34,8 dB, SSIM 0,91).
   ORB + RANSAC **n'aligne pas deux objets différents** (0 à 21 % de réussite) → alignement par centroïde et rotation.
   Pour la texture, la normalisation **locale** corrige un éclairage non uniforme du test.
3. **Caractéristiques** (03) : le résidu et les statistiques par zone sont les plus discriminants pour les objets,
   l'histogramme par zone pour la texture ; ORB/SIFT en global sont nettement moins bons.
4. **Apprentissage** (04) : sélection par **anomalies synthétiques** notées hors-pli. Deux problèmes de la première
   version (défauts simulés trop petits ; seuil calibré sur trop peu d'images) ont été corrigés.
5. **Inférence** (05) : la chaîne sur images brutes reproduit exactement l'apprentissage ; la région la plus anormale
   tombe sur le vrai défaut dans 80 à 100 % des cas.
6. **Évaluation** (06) : métriques image, défaut et pixel ; ablation ; choix final du prétraitement par le critère sans étiquettes.

**Limites** : critère de sélection peu fiable pour hazelnut et carpet ; 15 % de fausses alarmes sur bottle et hazelnut
(mesurés sur 20 et 40 images seulement) ; localisation grossière (résolution des zones) ; approche classique limitée
sur les défauts de moins de 1 % de l'image et sur les produits naturels de forme variable.
