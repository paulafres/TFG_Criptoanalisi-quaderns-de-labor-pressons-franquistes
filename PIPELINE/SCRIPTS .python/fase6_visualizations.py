# -*- coding: utf-8 -*-
"""
FASE 6 — Orquestrador de visualitzacions
=========================================

Crida els dos sub-mòduls de visualització:

  fase6_classical_views.py
      → dashboards individuals per teixit, arbres morfològics, heatmaps
        globals, PCA, correlacions, matrius de bombolles de patrons
        (gràfics heretats de l'antic pipeline; tots a res_fase6/)

  fase6_distributional_views.py
      → visualitzacions del nou pipeline distribucional
        (classes de tokens, signatures, tipus de volta, gramàtiques
        micro/macro, anomalies; tots a res_fase6/01_…/ ... 06_…/)

Aquest script només encadena. Per executar només una capa, executa
directament el seu sub-mòdul.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PY = sys.executable

SCRIPTS = [
    ("Capa clàssica   (dashboards, patrons, arbres, PCA, correlacions)",
     THIS_DIR / "fase6.1_classical_views.py"),
    ("Capa distribucional (classes, signatures, gramàtiques, anomalies)",
     THIS_DIR / "fase6.2_distributional_views.py"),
]


def run(label: str, script: Path) -> int:
    print("\n" + "=" * 70)
    print(f">> {label}")
    print(f"   {script.name}")
    print("=" * 70)
    r = subprocess.run([PY, str(script)])
    return r.returncode


def main():
    rcs = []
    for label, script in SCRIPTS:
        if not script.exists():
            print(f"[ERROR] No trobat: {script}")
            rcs.append(-1)
            continue
        rcs.append(run(label, script))

    print("\n" + "=" * 70)
    print("FASE 6 (orquestrador) — RESUM")
    print("=" * 70)
    for (label, script), rc in zip(SCRIPTS, rcs):
        status = "OK" if rc == 0 else f"FALLA (rc={rc})"
        print(f"  [{status:<14}]  {script.name}")
    sys.exit(0 if all(r == 0 for r in rcs) else 1)


if __name__ == "__main__":
    main()
