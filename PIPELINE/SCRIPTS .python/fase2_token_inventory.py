"""
FASE 2: DATASET D'INSTRUCCIONS EXPANDIDES
==========================================

Pipeline dual:
  1. Tokens RAW paleogràfics (heretats de Fase 1, amb posicions enriquides)
  2. Tokens expandits físics (macros desplegades)

Aquesta versió reaprofita el `teixits_dataset.json` generat per Fase 1
per evitar duplicar tokenitzador i lectura de fitxers. Així es garanteix
que totes les fases posteriors comparteixen exactament la mateixa
segmentació atòmica (incloent niuament correcte de parèntesis).

Calcula i exporta:
  - instruction_dataset   : per cada volta, tokens atòmics (enriquits)
                            + tokens expandits físicament + blocks_verticals
"""

import os
import re
import json
from collections import Counter

from config import RES_FASE_2, ensure_dir, F1_TEIXITS_DATASET, F2_INSTRUCTION_DATASET

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ensure_dir(RES_FASE_2)

# ─────────────────────────────────────────────
# EXPANSOR FÍSIC
# ─────────────────────────────────────────────
#
# Opera sobre la cadena `instruccio_raw` (ja al dataset de Fase 1).
# No fa cap suposició semàntica: només desplega macros sintàctiques
# del tipus (seq-Nv) → seq seq ... seq, i repeticions Nx → x x ... x.

def expand_instruction(text):
    """
    Retorna (tokens, blocks_verticals) on:
      - tokens: llista de tokens expandits. Els blocs verticals (seq-Nv) hi
        apareixen com a token compacte `<seq>` (sense N) per evitar explosió
        d'identitats hapax al clustering distribucional.
      - blocks_verticals: llista d'objectes
          {"pos": índex_dins_tokens, "seq": "la", "n_voltes": 2}
        que conserva la informació de N i la sub-seqüència original
        per a Fase 3.3 (signatura de volta) i Fase 6 (render).
    """
    s = text.strip()
    result = []
    blocks = []
    i = 0

    while i < len(s):

        # ignorar espais
        if s[i].isspace():
            i += 1
            continue

        # MARCADORS multi-caràcter (preservats com a tokens propis)
        if s[i:i+3] == "IMZ":
            result.append("IMZ")
            i += 3
            continue

        if s[i:i+2] == "MZ":
            result.append("MZ")
            i += 2
            continue

        # BLOCS (...) amb niuament correcte
        if s[i] == "(":
            depth = 1
            j = i + 1
            while j < len(s):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            inner = s[i+1:j]

            # patró de repetició VERTICAL: "seq-Nv"
            # Es desa com a un únic token `<seq>` (sense N a la identitat).
            # N s'arxiva a `blocks_verticals` com a metadada.
            m_repeat = re.match(r'^(.+)-(\d+)v$', inner)

            if m_repeat:
                bloc = m_repeat.group(1)
                n = int(m_repeat.group(2))
                blocks.append({
                    "pos": len(result),
                    "seq": bloc,
                    "n_voltes": n,
                })
                result.append(f"<{bloc}>")
            else:
                # (53) → es manté com a token propi
                if re.match(r'^\d+$', inner):
                    result.append(f"({inner})")
                else:
                    # bloc amb contingut alfabètic → expandir contingut
                    inner_toks, inner_blocks = expand_instruction(inner)
                    base = len(result)
                    result.extend(inner_toks)
                    for b in inner_blocks:
                        blocks.append({**b, "pos": b["pos"] + base})

            i = j + 1
            continue

        # REPETICIONS numèriques: 3a → a a a, 4l → l l l l
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            n = int(s[i:j])

            # 3> → token "3>"
            if j < len(s) and s[j] == ">":
                result.append(f"{n}>")
                i = j + 1
                continue

            # NIMZ → IMZ × N (preservar marcador multi-char com unitat)
            if s[j:j+3] == "IMZ":
                result.extend(["IMZ"] * n)
                i = j + 3
                continue

            # NMZ → MZ × N (preservar marcador multi-char com unitat)
            if s[j:j+2] == "MZ":
                result.extend(["MZ"] * n)
                i = j + 2
                continue

            # 3a → a a a
            if j < len(s) and s[j].isalpha():
                tok = s[j]
                result.extend([tok] * n)
                i = j + 1
                continue

            # número solt
            result.append(str(n))
            i = j
            continue

        # <4
        if s[i] == "<":
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            result.append(s[i:j])
            i = j
            continue

        # lletra simple
        if s[i].isalpha():
            result.append(s[i])
            i += 1
            continue

        # altres símbols
        result.append(s[i])
        i += 1

    return result, blocks

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

with open(F1_TEIXITS_DATASET, "r", encoding="utf-8") as f:
    teixits_dataset = json.load(f)

global_freq = Counter()
instruction_dataset = []

for teixit in teixits_dataset:
    nom = teixit.get("nom")
    fitxer = teixit.get("fitxer")
    punts_inicials = teixit.get("punts_inicials")

    for inst in teixit.get("instruccions", []):
        instruccio_raw = inst["instruccio_raw"]
        tokens_atomics = inst["tokens"]  # llista d'objectes enriquits (Fase 1)

        tokens_expandits, blocks_verticals = expand_instruction(instruccio_raw)

        if not tokens_expandits:
            continue

        instruction_dataset.append({
            "teixit": nom,
            "fitxer": fitxer,
            "volta": inst["volta"],
            "instruccio_raw": instruccio_raw,
            "punts_inicials": punts_inicials,
            "tokens_atomics": tokens_atomics,        # enriquits, heretats de Fase 1
            "tokens_expandits": tokens_expandits,    # macros desplegades
            "blocks_verticals": blocks_verticals,    # metadades (seq, n_voltes) per posició
        })

        # Estadística sobre tokens EXPANDITS (per Markov i fases posteriors)
        for tok in tokens_expandits:
            global_freq[tok] += 1

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

with open(
    F2_INSTRUCTION_DATASET,
    "w",
    encoding="utf-8"
) as f:
    json.dump(instruction_dataset, f, indent=2, ensure_ascii=False)

print("\nFASE 2 COMPLETADA")
print(f"  Teixits processats : {len(teixits_dataset)}")
print(f"  Voltes             : {len(instruction_dataset)}")
print(f"  Tokens únics       : {len(global_freq)}")
