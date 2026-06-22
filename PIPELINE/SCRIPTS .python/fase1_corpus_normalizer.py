import os
import re
import json
import csv
from collections import Counter
import zlib
import math
from pathlib import Path

from config import (
    TEIXITS_DIR, RES_FASE_1, ensure_dir,
    F1_TEIXITS_DATASET, F1_PATRONS_COMPARTITS, F1_SIMBOLS_INDIVIDUALS,
)

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓ
# ─────────────────────────────────────────────

CARPETA = str(TEIXITS_DIR)

MIN_FREQ_DINS_TEIXIT = 2
MIN_LEN_TOKENS = 2
MAX_LEN_TOKENS = 1000
MIN_TEIXITS_COMPARTIT = 2

FORMES = {
    # ─── Quadern Carmen Machado (CM) ───
    "CM_Hojas y cactus":   "Redondo",
    "CM_Abouchete":        "Redondo/Ovalado",
    "CM_Pasión":           "Redondo",
    "CM_Angelines":        "Cuadrado",
    "CM_Eterna":           "Redondo",
    "CM_Pelargonia":       "Redondo",
    "CM_Gardenia":         "Redondo",
    "CM_Hojas de Roble":   "Redondo",
    "CM_Punto Angelines":  "",
    "CM_Pepita":           "Redondo",
    "CM_Copas de champán": "Cuadrado/Redondo",

    # ─── Quadern Manoli Segovia (MS) ───
    # ─── Quadern Manoli Segovia (MS) ───
    "MS_Estrellita": "Redondo",
    "MS_Jazmín": "Redondo",
    "MS_Hojas": "",
    "MS_Susana": "Redondo",
    "MS_Julito": "Redondo",
    "MS_Pensamiento": "Hexagonal",
    "MS_Penas": "Redondo",
    "MS_Pelargonia": "Redondo",
    "MS_Dalia": "Redondo",
    "MS_Rita": "Redondo",
    "MS_Mariposa": "Redondo",
    "MS_Salvador": "Redondo",
    "MS_Punto Montaña": "",
    "MS_Puntilla pequeña Salvador": "",
    "MS_Marianin o Katiuska": "Cuadrado/Redondo",
    "MS_Puntilla1": "",
    "MS_Araceli": "Redondo",
    "MS_Vaño Mimosa": "Redondo",
    "MS_Leonor": "Redondo",
    "MS_Puntilla2": "",
    "MS_Gafitas1": "",
    "MS_Puntilla nueva": "",
    "MS_Blumen Flor": "Ovalado",
    "MS_Grafico B - Puntilla": "",
    "MS_Adolbert": "Cuadrado",
    "MS_Margarita": "",
    "MS_Plumitas": "Cuadrado",
    "MS_Mary-Luz": "Cuadrado",
    "MS_Puntilla3": "",
    "MS_Gafitas2": "Cuadrado",
    "MS_Madreselva": "Ovalado",
    "MS_Grafico 2": "",
    "MS_Luis": "Redondo",
    "MS_La Puntilla": "",
    "MS_Morin": "Cuadrado",
    "MS_Celia 1": "Redondo",
    "MS_Celia 2": "Redondo",
    "MS_Cacus": "Cuadrado",
    "MS_Fajas": "Redondo",
    "MS_Pino": "Redondo",
    "MS_Campanilla": "Redondo",
    "MS_Claveles": "",
    "MS_Dalila": "",
    "MS_MLuisa": "Redondo",
    "MS_Raetine": "Redondo",
    "MS_Titi 1": "Redondo",
    "MS_Titi 2 - Grafico A": "Ovalado",
    "MS_Titi 2 - Grafico B": "",
    "MS_Titi - Puntilla": "Redondo",
    "MS_Punto Bellota": "",
    "MS_Otro punto bellota": "",
    "MS_Otro Punto Calado": "",
    "MS_Punto Calado": "",
    "MS_Punto conchas": "",
    "MS_Punto piñas": "",
    "MS_Punto dos caras": "",
    "MS_Hojas de Matilde": "",
    "MS_Punto Feito": "",
    "MS_Punto plisado para vestidito": "",
    "MS_Punto Esmeralda": "",
    "MS_Punto Hojitas vueltas": "",
    "MS_Punto plisado para chaquetita": "",
    "MS_Punto semilla": "",
    "MS_Palmito": "Ovalado",
    "MS_Ovalo1": "Ovalado",
    "MS_Puntilla4": "",
    "MS_Antoña o Canijo": "",
    "MS_Ovalo2": "Ovalado",
    "MS_Vueltas del reves": "",
    "MS_Hijo": "Ovalado",

    # ─── Quadern Negro Dimetri (ND) ───
    "ND_Demetri grande": "",
    "ND_Demetri pequeño": "",
    "ND_Coco liso": "Cuadrado",
    "ND_Vilno (Hijo)": "",
    "ND_Isabelet": "",
    "ND_Rosario": "Cuadrado",
    "ND_Anelise": "Redondo",
    "ND_Aurea": "Redondo",
    "ND_Conchita Romero": "",
    "ND_Modelo Ros Mari": "",
    "ND_Punto Angelines para blusa": "",
    "ND_Zapatitos de recién nacido": "",
    "ND_Ruth": "",
    "ND_Chistrofe Rosa de Navidad": "",
    "ND_Maria": "",
    "ND_Sin nombre": "",
    "ND_Tara": "",
    "ND_Renacimiento": "Cuadrado",
    "ND_Campesina": "",
    "ND_Merche": "",
    "ND_Dolores": "",
    "ND_Chechee": "",

    # ─── Quadern Penadas (P) ───
    "P_Gentio": "Redondo",
    "P_Jafisa - Grafico A": "Ovalado",
    "P_Jafisa - Grafico B": "",
    "P_Jafisa - Grafico C": "",
    "P_Conchita Romero": "",
    "P_Ruth": "Ovalado",
    "P_Primavera": "",
    "P_Pasión": "",
    "P_Celia": "",
    "P_Rosa de los Vientos": "",
    "P_Renacimiento": "Cuadrado",
    "P_Puntilla del Cacus": "",
    "P_Helga": "",
    "P_Cori": "",
}

# ─────────────────────────────────────────────
# 2. FUNCIONS
# ─────────────────────────────────────────────

def netejar_nom(fitxer_path) -> str:
    """Extreu les sigles del quadern i el nom net. Ex: CM. 4. Pasión.txt -> CM_Pasión"""
    nom_arxiu = os.path.basename(fitxer_path).replace(".txt", "")
    # Cerca un patró: Lletres (CM/MS/ND/P) + número (ignorat) + Nom del teixit
    m = re.match(r'^([A-Z]+)\.?\s*\d+(?:-\d+)?\.?\s*(.*)$', nom_arxiu)
    if m:
        prefix = m.group(1)
        nom_teixit = m.group(2).strip()
        return f"{prefix}_{nom_teixit}"
    return nom_arxiu.strip()

def tokenize(instr: str) -> list[str]:

    tokens = []
    s = instr.strip().lower()
    i = 0

    while i < len(s):

        c = s[i]

        # BLOCS ()
        if c == '(':
            depth = 1
            j = i + 1

            while j < len(s) and depth > 0:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                j += 1

            tokens.append(s[i:j])
            i = j
            continue

        # N>
        if c.isdigit():

            j = i
            while j < len(s) and s[j].isdigit():
                j += 1

            num = s[i:j]

            if j < len(s) and s[j] == '>':
                tokens.append(num + '>')
                i = j + 1
                continue

            tokens.append(num)
            i = j
            continue

        # <N
        if c == '<':

            j = i + 1

            while j < len(s) and s[j].isdigit():
                j += 1

            if j > i + 1:
                tokens.append(s[i:j])

            i = j
            continue

        # OPERACIONS INDIVIDUALS
        if c in "lpagxr":
            tokens.append(c)
            i += 1
            continue

        i += 1

    return tokens

def enrich_tokens(tokens: list[str]) -> list[dict]:
    """
    Afegeix metadades posicionals a cada token sense canviar-ne la seqüència.
    Cap suposició semàntica: només informació observable.

    Camps per token:
      - token         : la cadena del token tal com l'ha produït el tokenitzador
      - index         : posició 0-based dins la instrucció
      - pos_norm      : posició normalitzada [0.0, 1.0] dins la instrucció
                        (útil per detectar tokens "d'obertura" o "de tancament")
      - longitud      : nombre de caràcters del token
      - es_macro      : True si el token és un bloc entre parèntesis "(...)"
      - profunditat   : nivell màxim de niuament de parèntesis dins el token
    """
    n = len(tokens)
    enriched = []
    for idx, tok in enumerate(tokens):
        pos_norm = idx / (n - 1) if n > 1 else 0.0
        es_macro = tok.startswith("(") and tok.endswith(")")
        # Profunditat: nivell màxim de "(" oberts simultàniament dins el token
        depth = 0
        max_depth = 0
        for ch in tok:
            if ch == '(':
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif ch == ')':
                depth -= 1
        enriched.append({
            "token": tok,
            "index": idx,
            "pos_norm": round(pos_norm, 4),
            "longitud": len(tok),
            "es_macro": es_macro,
            "profunditat": max_depth,
        })
    return enriched

def extract_repeated_patterns(tokenized_lines, min_freq=2, min_len=2, max_len=1000):
    """
    Neteja absoluta per Geometria de Tokens.
    Si una subseqüència no apareix MAI de forma independent (és a dir, 
    SEMPRE forma part d'una seqüència més llarga acceptada), s'elimina.
    """
    # 1. Trobar posicions exactes de totes les seqüències
    seq_counts = Counter()
    seq_positions = {}

    for line_idx, toks in enumerate(tokenized_lines):
        n = len(toks)
        for length in range(min_len, min(n, max_len) + 1):
            for i in range(n - length + 1):
                seq = tuple(toks[i:i+length])
                seq_counts[seq] += 1
                if seq not in seq_positions:
                    seq_positions[seq] = []
                seq_positions[seq].append((line_idx, i))

    # 2. Filtrar per freqüència mínima i ordenar de MÉS LLARG a MÉS CURT
    valid_seqs = [seq for seq, count in seq_counts.items() if count >= min_freq]
    valid_seqs.sort(key=lambda x: -len(x))

    # 3. Lògica de Cobertura (Absorció Real)
    # covered_tokens guardarà els índexs dels tokens que ja pertanyen a un patró llarg
    covered_tokens = {i: set() for i in range(len(tokenized_lines))}
    accepted_seqs = {}
    patterns_info = {}

    for seq in valid_seqs:
        independent_count = 0
        
        for line_idx, start_idx in seq_positions[seq]:
            # Mirem si AQUESTA aparició específica ja està "dins" d'un patró més llarg
            is_covered = True
            for k in range(start_idx, start_idx + len(seq)):
                if k not in covered_tokens[line_idx]:
                    is_covered = False
                    break
            
            if not is_covered:
                independent_count += 1
        
        # Si apareix de forma "independent" almenys 1 vegada, és un patró vàlid.
        if independent_count > 0:
            accepted_seqs[seq] = seq_counts[seq]
            s_str = ''.join(seq)
            patterns_info[s_str] = {
                "freq_total": seq_counts[seq],
                "freq_independent": independent_count,
                "es_variant_de": [],
                "te_variants_mes_llargues": []
            }
            
            # Com que l'hem acceptat, marquem els seus tokens com a "ocupats"
            for line_idx, start_idx in seq_positions[seq]:
                for k in range(start_idx, start_idx + len(seq)):
                    covered_tokens[line_idx].add(k)

    # 4. Construir la Genealogia NOMÉS amb els patrons que han sobreviscut
    accepted_keys = list(accepted_seqs.keys())
    for i, seq_i in enumerate(accepted_keys):
        s_i = ''.join(seq_i)
        for j, seq_j in enumerate(accepted_keys):
            if i == j: continue
            s_j = ''.join(seq_j)
            
            # Comprovem si seq_i és subseqüència de seq_j buscant-ho dins la tupla
            is_sub = False
            len_i, len_j = len(seq_i), len(seq_j)
            if len_i < len_j:
                for k in range(len_j - len_i + 1):
                    if seq_j[k:k+len_i] == seq_i:
                        is_sub = True
                        break
            
            if is_sub:
                patterns_info[s_i]["te_variants_mes_llargues"].append({
                    "patro": s_j,
                    "freq_variant": accepted_seqs[seq_j]
                })
                patterns_info[s_j]["es_variant_de"].append(s_i)

    finals = {''.join(k): v for k, v in accepted_seqs.items()}
    return finals, patterns_info

# ─────────────────────────────────────────────
# 3. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

# Busquem a la carpeta i a totes les seves subcarpetes recursivament
fitxers_paths = [p for p in Path(CARPETA).rglob("*.txt") if p.is_file()]
print(f"Iniciant processament de {len(fitxers_paths)} teixits...\n")

dataset = []

for path_obj in sorted(fitxers_paths):
    path = str(path_obj)          # Ruta completa (inclou subcarpeta)
    fitxer = path_obj.name        # Només el nom de l'arxiu
    nom = netejar_nom(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    punts_inicials = None
    inici_lectura = 0
    if lines:
        primera = lines[0].strip()
        
        # Núm. punts incials: Busca números (ex: "8") o números amb barra (ex: "8/10")
        m = re.match(r'^(\d+(?:/\d+)?)p$', primera, re.IGNORECASE)
        
        if m:
            # Ho guardem com a text perquè pot ser "8" o "8/10"
            punts_inicials = m.group(1) 
            inici_lectura = 1

    instruction_texts = []
    tokenized_lines = []
    row_nums = []
    instruccions_enriched = []

    for raw_line in lines[inici_lectura:]:
        raw_line = raw_line.strip()
        if not raw_line: continue
        m = re.match(r'^(\d+)\s+(.*)', raw_line)
        if not m: continue
        volta_num = int(m.group(1))
        row_nums.append(volta_num)
        instr = m.group(2)
        instruction_texts.append(instr)
        toks = tokenize(instr)
        if toks:
            tokenized_lines.append(toks)
            instruccions_enriched.append({
                "volta": volta_num,
                "instruccio_raw": instr,
                "num_tokens": len(toks),
                "tokens": enrich_tokens(toks),
            })

    patrons, patterns_info = extract_repeated_patterns(
        tokenized_lines,
        min_freq=MIN_FREQ_DINS_TEIXIT,
        min_len=MIN_LEN_TOKENS,
        max_len=MAX_LEN_TOKENS,
    )

    # Ordenem per FREQÜÈNCIA INDEPENDENT (les vegades que surt sol) * LONGITUD.
    dominants = sorted(
        patterns_info.items(), 
        key=lambda x: -(x[1]["freq_independent"] * len(x[0]))
    )[:5]

    all_instr_text = ''.join(instruction_texts)
    char_counts = Counter(c for c in all_instr_text if not c.isspace())

    print(f"{'─'*40}")
    print(f"TEIXIT : {nom}")
    print(f"Fitxer : {fitxer}")
    print(f"Forma  : {FORMES.get(nom, '—')}")
    print(f"Punts inicials  : {punts_inicials if punts_inicials else '—'}")
    print(f"Instruccions    : {len(instruction_texts)}")
    rang = f"{min(row_nums)} – {max(row_nums)}" if row_nums else "—"
    print(f"Rang de voltes  : {rang}")
    print(f"Patrons repetits (≥{MIN_FREQ_DINS_TEIXIT} línies): {len(patrons)}")
    print(f"Tokens dominants: {[t for t, _ in dominants]}")
    top5_chars = sorted(char_counts.items(), key=lambda x: -x[1])[:8]
    print(f"Caràcters freqs : {top5_chars}")
    print()

    # ── CÀLCUL DE MÈTRIQUES ESTADÍSTIQUES ──
    # Unim el text respectant l'ordre exacte de les instruccions
    text_complet = ''.join(instruction_texts)
    data_bytes = text_complet.encode("utf-8")
    tot_chars = len(text_complet)
    c_chars = Counter(text_complet)

    # m1. Freq. mitjana de caràcters
    m1 = sum(v/tot_chars for v in c_chars.values()) / len(c_chars) if c_chars else 0.0

    # m2. Freq. mitjana bigrames
    bg = [text_complet[i:i+2] for i in range(len(text_complet)-1)]
    c_bg = Counter(bg)
    m2 = sum(v/len(bg) for v in c_bg.values()) / len(c_bg) if c_bg else 0.0

    # m3. Freq. mitjana trigrames
    tg = [text_complet[i:i+3] for i in range(len(text_complet)-2)]
    c_tg = Counter(tg)
    m3 = sum(v/len(tg) for v in c_tg.values()) / len(c_tg) if c_tg else 0.0

    # m4. Ràtio de compressió (Zlib)
    m4 = len(zlib.compress(data_bytes)) / len(data_bytes) if data_bytes else 0.0

    # m5. Ràtio de repetició (finestra de 5)
    subs = [text_complet[i:i+5] for i in range(len(text_complet)-4)]
    m5 = 1.0 - (len(set(subs)) / len(subs)) if subs else 0.0

    # m6. Simetria de línies (Palíndroms exactes)
    simetriques = sum(1 for linia in instruction_texts if linia == linia[::-1])
    m6 = simetriques / len(instruction_texts) if instruction_texts else 0.0

    # m7. Creixement numèric (Progressió de les voltes)
    deltas = [b - a for a, b in zip(row_nums[:-1], row_nums[1:])]
    m7 = sum(deltas) / len(deltas) if deltas else 0.0

    # m8. Entropia de Shannon (Desordre de la informació)
    m8 = -sum((v/tot_chars)*math.log2(v/tot_chars) for v in c_chars.values()) if tot_chars else 0.0

    # Agrupem les mètriques en un diccionari
    metriques = {
        "m1_char_freq": round(m1, 5),
        "m2_bigram": round(m2, 5),
        "m3_trigram": round(m3, 5),
        "m4_compressio": round(m4, 5),
        "m5_repeticio": round(m5, 5),
        "m6_simetria": round(m6, 5),
        "m7_creixement": round(m7, 2),
        "m8_entropia": round(m8, 4)
    }


    dataset.append({
        "nom": nom,
        "fitxer": fitxer,
        "path_complet": path,
        "forma": FORMES.get(nom, ""),
        "punts_inicials": punts_inicials,
        "num_instruccions": len(instruction_texts),
        "rang_voltes": [min(row_nums), max(row_nums)] if row_nums else None,
        "metriques": metriques,
        "tokens_dominants": [t for t, _ in dominants],
        "patterns_freq": patrons,
        "patterns_info": patterns_info,
        "caracters_individuals": dict(sorted(char_counts.items())),
        "instruccions": instruccions_enriched
    })

# ─────────────────────────────────────────────
# 4. EXPORTACIÓ
# ─────────────────────────────────────────────

ensure_dir(RES_FASE_1)

with open(F1_TEIXITS_DATASET, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

# ── Taula patrons compartits ──
aparicions_per_fitxer = Counter()
for t in dataset:
    for patro in t["patterns_freq"]:
        aparicions_per_fitxer[patro] += 1

# Ens quedem amb els patrons que apareixen com a mínim a 2 teixits
tokens_compartits = sorted([p for p, count in aparicions_per_fitxer.items() if count >= MIN_TEIXITS_COMPARTIT])

with open(F1_PATRONS_COMPARTITS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["teixit", "forma"] + tokens_compartits)
    writer.writeheader()
    
    for t in dataset:
        fila = {"teixit": t["nom"], "forma": t["forma"]}
        
        # Per a cada teixit, mirem el text original (totes les instruccions juntes)
        # Així podem fer un recompte ràpid i precís de quantes vegades apareix la cadena de text
        fitxer_path = os.path.join(CARPETA, t["path_complet"])
        with open(fitxer_path, "r", encoding="utf-8") as text_file:
            # Omplim les columnes amb freqüències reals i numèriques sense ajuntar línies
            for p in tokens_compartits:
                freq_absoluta = 0
                with open(fitxer_path, "r", encoding="utf-8") as text_file:
                    for l in text_file:
                        if re.match(r'^\d+\s+', l.strip()):
                            # Extraiem la instrucció neta de la línia
                            linia_neta = l.split(' ', 1)[1].replace(" ", "").strip()
                            freq_absoluta += linia_neta.count(p)
                
                fila[p] = freq_absoluta
            
        writer.writerow(fila)


# ── Taula símbols individuals ──
tots_simbols = set()
for t in dataset:
    tots_simbols.update(t["caracters_individuals"].keys())
simbols_ordenats = sorted(tots_simbols)

with open(F1_SIMBOLS_INDIVIDUALS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["teixit", "forma"] + simbols_ordenats)
    writer.writeheader()
    for t in dataset:
        fila = {"teixit": t["nom"], "forma": t["forma"]}
        for s in simbols_ordenats:
            fila[s] = t["caracters_individuals"].get(s, 0)
        writer.writerow(fila)


print("=" * 40)
print(f"Teixits processats            : {len(dataset)}")
print(f"Patrons sintàctics compartits : {len(tokens_compartits)}")
print(f"Símbols de hardware analitzats: {len(simbols_ordenats)}")
print()
print("Fitxers generats a la carpeta 'res_fase1/':")
print("  · teixits_dataset.json")
print("  · taula_patrons_compartits.csv")
print("  · taula_simbols_individuals.csv")