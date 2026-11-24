"""Exécute un notebook et enregistre ses sorties : python tools/run_notebook.py notebooks/XX.ipynb
Le noyau est lancé depuis le dossier du notebook (comme dans VS Code / Jupyter)."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

for p in [Path(a).resolve() for a in sys.argv[1:]]:
    os.chdir(p.parent)
    nb = nbformat.read(p, 4)
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3").execute()
    finally:
        nbformat.write(nb, p)
    errors = [o.evalue for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
    print(f"{p.name}: {'OK' if not errors else 'ERREURS : ' + str(errors)}", flush=True)
