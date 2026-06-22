# -*- coding: utf-8 -*-
"""
ORQUESTRADOR PRINCIPAL DEL PIPELINE
===================================
Aquest script executa de forma seqüencial totes les fases del TFG.
Només has de canviar CORPUS_ACTIU a config.py i executar aquest arxiu.
"""

import subprocess
import sys
from pathlib import Path

# Llegim quin corpus està actiu directament de la config
from config import CORPUS_ACTIU

THIS_DIR = Path(__file__).resolve().parent
PY = sys.executable

# Llista exacta amb l'ordre d'execució de totes les fases
FASES = [
    "fase1_corpus_normalizer.py",
    "fase2_token_inventory.py",
    "fase3.1_markov_model.py",
    "fase3.1b_transition_significance.py",
    "fase3.2_distributional_classes.py",
    "fase3.2b_clustering_stability.py",
    "fase3.3_volta_signatures.py",
    "fase3.4_volta_clustering.py",
    "fase4.1_macro_grammar.py",
    "fase4.2_micro_grammar.py",
    "fase4.3_null_model.py",
    "fase4.4_loo_cv.py",
    "fase5.1_anomaly_report.py",
    "fase5.1b_crypto_validation.py",
    "fase5.2_segment_anomaly_detection.py",
    "fase5.2b_crypto_segments.py",
    "fase5.3_hipotesis_referencies.py",
    "fase6.1_classical_views.py",
    "fase6.2_distributional_views.py",
    "resum_global.py"        
]

def main():
    print("=" * 70)
    print(f" INICIANT PIPELINE COMPLET --- CORPUS ACTIU: [{CORPUS_ACTIU}]")
    print("=" * 70)

    for script_name in FASES:
        script_path = THIS_DIR / script_name
        if not script_path.exists():
            print(f"\n[!] ALERTA: No es troba {script_name}. Saltant...")
            continue
            
        print(f"\n---> Executant {script_name} ...")
        result = subprocess.run([PY, str(script_path)])
        
        if result.returncode != 0:
            print(f"\n[ERROR CRÍTIC] L'script {script_name} ha fallat.")
            print("Aturant el pipeline perquè les fases següents depenen d'aquest.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETAT AMB ÈXIT PER AL CORPUS: [{CORPUS_ACTIU}]")
    print(f" Revisa la carpeta RESULTATS_{CORPUS_ACTIU} i l'arxiu resum_global.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()