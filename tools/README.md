# tools/ — génération et exécution des notebooks

- `make_nbXX.py` : écrit le notebook XX (cellules de code + texte).
- `conclusions/XX.md` : cellule de conclusion ajoutée à la fin du notebook XX (à modifier ici, pas dans le script).
- `run_notebook.py` : exécute un ou plusieurs notebooks et enregistre leurs sorties.

Depuis la racine du projet :

```
.venv\Scripts\python.exe tools\make_nb03.py notebooks\03_caracteristiques.ipynb
.venv\Scripts\python.exe tools\run_notebook.py notebooks\03_caracteristiques.ipynb
```
