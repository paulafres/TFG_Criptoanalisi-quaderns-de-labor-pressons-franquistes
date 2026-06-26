# El missatge entre punts: Criptoanàlisi dels quaderns de labor en presons franquistes

Aquest repositori conté el pipeline computacional complet desenvolupat per al TFG en Enginyeria de Dades (UAB). El projecte explora els quaderns de labor de preses polítiques antifranquistes com a possibles canals d'informació encoberta. 

Mitjançant tècniques de Processament de Llenguatge Natural (NLP), aprenentatge no supervisat i validació estadística, el sistema modela la gramàtica emergent dels teixits per detectar anomalies estructurals i lèxiques que puguin ser indici d'esteganografia.

## Arquitectura del Pipeline

El projecte està dissenyat com un pipeline seqüencial de 6 fases. Cada script encapsula els seus hiperparàmetres i garanteix la reproductibilitat dels resultats.

### Fase 1: Adquisició i Normalització
*   `fase1_corpus_normalizer.py`: Extreu les instruccions en text pla, estandarditza les grafies i extreu patrons sintàctics.

### Fase 2: Inventari de Tokens
*   `fase2_token_inventory.py`: Expandeix físicament les macros i operadors de repetició, generant el dataset atòmic base per al modelatge.

### Fase 3: Modelatge Distribucional
*   `fase3.1_markov_model.py` / `fase3.1b_transition_significance.py`: Construeix cadenes de Markov bidireccionals i avalua la significació de les transicions (FDR-BH).
*   `fase3.2_distributional_classes.py` / `fase3.2b_clustering_stability.py`: Agrupa els tokens en classes funcionals usant K-Means sobre vectors PPMI i en valida l'estabilitat mitjançant Bootstrap.
*   `fase3.3_volta_signatures.py` / `fase3.4_volta_clustering.py`: Genera signatures distribucionals per a cada "volta" del teixit i n'extreu tipologies estructurals emergents.
*   `fase3_lib_distributional.py`: Llibreria compartida per a mètriques distribucionals.

### Fase 4: Gramàtiques i Models Nuls
*   `fase4.1_macro_grammar.py` / `fase4.2_micro_grammar.py`: Modela la predictibilitat (sorpresa de Shannon) a nivell macro (entre voltes) i micro (dins de cada volta).
*   `fase4.3_null_model.py`: Compara el corpus contra un model nul per permutació per confirmar la intencionalitat de l'estructura.
*   `fase4.4_loo_cv.py`: Calcula la capacitat de generalització del model mitjançant validació creuada Leave-One-Out.
*   `fase4_lib_ngram_grammar.py`: Llibreria base de models n-grama.

### Fase 5: Detecció i Contrast Criptogràfic
*   `fase5.1_anomaly_report.py`: Redueix els canals d'entropia mitjançant PCA (PC1) per aïllar voltes amb desviació extrema.
*   `fase5.1b_crypto_validation.py`: Aplica tests clàssics (Índex de Coincidència, Kasiski, Wald-Wolfowitz) per identificar la viabilitat de codis alfabètics.
*   `fase5.2_segment_anomaly_detection.py` / `fase5.2b_crypto_segments.py`: Troba segments de voltes contigües i triangula anomalies (perfil lèxic vs. gramatical).
*   `fase5.3_hipotesis_referencies.py`: Contrasta automàticament el corpus amb hipòtesis històriques (Morse tèxtil, anomalies belgues, sistema clandestí del PCE).

### Fase 6: Visualitzacions
*   `fase6.1_classical_views.py` / `fase6.2_distributional_views.py` / `fase6_visualizations.py`: Genera tots els gràfics (dashboards, PCA, heatmaps, matrius de transició i sèries temporals).

## Ús i Execució

L'orquestrador principal executa totes les fases en l'ordre correcte i s'atura automàticament si alguna dependència falla.

### 1. Configuració (`config.py`)
A l'arxiu `config.py`, defineix quin quadern (o conjunt de quaderns) vols analitzar canviant la variable `CORPUS_ACTIU`:
```
# Opcions: "CM", "MS", "ND", "P", "TOTS"
CORPUS_ACTIU = "P"
```

### 2. Executar el Pipeline Complet (`0_run_pipeline.py)
Simplement executa l'script mestre des de l'arrel dels scripts:
```
python 0_run_pipeline.py
```
Aquest procés generarà una carpeta `RESULTATS_[CORPUS_ACTIU]` amb tots els datasets generats, els JSONs intermedis i una carpeta `res_fase6` amb els gràfics generats.

### 3. Generar el Resum Global
Un cop completat el pipeline, pots generar un arxiu de text amb la síntesi interpretada de tots els resultats executant:
```python resum_global.py```
Això et donarà un output per consola i crearà l'arxiu `resum_global.txt` a l'arrel del projecte.

## Requisits
Assegura't de comptar amb un entorn Python 3.12+ i les següents llibreries instal·lades:
```
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn
- networkx
```
