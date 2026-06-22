# -*- coding: utf-8 -*-
"""
FASE 5.2B — TRIANGULACIÓ D'ANOMALIES GRAMATICALS × LÈXIQUES SOBRE SEGMENTS
=========================================================================

Justificació metodològica
-------------------------
Tenim tres senyals independents sobre el corpus:

  · FASE 5.1  (anomaly_report.json)  → anomalies GRAMATICALS per volta.
      Voltes que trenquen la gramàtica del teixit (surprisal macro/micro
      alt, transicions no validades, salt distribucional). Interessen per
      DESCARTAR que el missatge sigui una transcripció neta d'abecedari:
      si una volta és gramaticalment anòmala, probablement és farciment
      estructural i NO una seqüència de lletres xifrades.

  · FASE 5.1b (crypto_validation.json) → anomalies LÈXIQUES per volta.
      Voltes amb concentració de tokens C_RARE (hapax, marcadors, números,
      símbols) i/o agrupacions no aleatòries (runs test). Interessen per
      CONFIRMAR la hipòtesi contrària: si el missatge SÍ es desxifra en
      abecedari, viuria en aquests marcadors lèxics camuflats.

  · FASE 5.2  (anomaly_segments.json) → SEGMENTS contigus de voltes amb
      score d'anomalia elevat (candidats a fragment amagat).

Aquesta fase NO repeteix cap test criptogràfic. Fa una TRIANGULACIÓ:
per a cada segment de 5.2, recorre les seves voltes i les creua amb els
veredictes de 5.1 (gramatical) i 5.1b (lèxic), classificant cada volta i,
per agregació, cada segment.

Classes de volta
----------------
  CONVERGENT : la volta és anòmala gramaticalment (5.1) I lèxicament (5.1b).
               Senyal més fort: trenca la gramàtica i porta marcadors rars.
  GRAMATICAL : només 5.1. Compatible amb farciment estructural (no abecedari).
  LEXICA     : només 5.1b. Compatible amb codi camuflat (possible abecedari).
  CAP        : dins el segment d'alt score però sense senyal individual.

Veredicte per segment
----------------------
  CONVERGENCIA      : ≥1 volta CONVERGENT  → inspecció prioritària.
  PERFIL_LEXIC      : voltes LEXICA i cap GRAMATICAL → candidat abecedari.
  PERFIL_GRAMATICAL : voltes GRAMATICAL i cap LEXICA → candidat no abecedari.
  MIXT              : tots dos perfils presents en voltes diferents.
  SENSE_SENYAL      : ni 5.1 ni 5.1b marquen cap volta del segment.

Sortides
--------
- res_fase5.2b/triangulacio_segments.json
- res_fase5.2b/triangulacio_segments.csv
- res_fase5.2b/triangulacio_segments_report.txt
"""

from __future__ import annotations
import csv
import json
from collections import Counter

from config import (
    F5_1_ANOMALY_REPORT_JSON,
    F5_1B_CRYPTO_VALIDATION,
    F5_2_SEGMENTS_JSON,
    F5_2B_TRIANGULACIO_JSON,
    F5_2B_TRIANGULACIO_CSV,
    F5_2B_TRIANGULACIO_REPORT,
    F5_2B_ORFENES_CSV,
    RES_FASE_5_2B,
    ensure_dir,
)


# ───────────────────────── PARÀMETRES ────────────────────────────────────
# Categories de fase 5.1 que es consideren "anomalia gramatical" rellevant.
GRAMMAR_CATS = {"CRÍTICA", "ALTA", "MODERADA"}

# Ordre de presentació dels veredictes (de més a menys interessant).
ORDRE_VEREDICTE = {
    "CONVERGENCIA": 0,
    "PERFIL_LEXIC": 1,
    "MIXT": 2,
    "PERFIL_GRAMATICAL": 3,
    "SENSE_SENYAL": 4,
}


# ───────────────────────── CÀRREGA ───────────────────────────────────────
def carrega_gramaticals() -> dict[tuple[str, int], dict]:
    """Index (teixit, volta) → registre gramatical de fase 5.1."""
    doc = json.load(open(F5_1_ANOMALY_REPORT_JSON, encoding="utf-8"))
    out: dict[tuple[str, int], dict] = {}
    for r in doc.get("anomalies", []):
        flags_actius = [k for k, v in r.items()
                        if k.startswith("flag_") and v]
        out[(r["teixit"], int(r["volta"]))] = {
            "categoria": r.get("categoria"),
            "score_anomalia": r.get("score_anomalia"),
            "num_flags_actius": r.get("num_flags_actius", 0),
            "cross_canal": bool(r.get("cross_canal", False)),
            "flags_actius": flags_actius,
        }
    return out


def carrega_lexiques() -> dict[tuple[str, int], dict]:
    """Index (teixit, volta) → registre lèxic de fase 5.1b.

    Una volta es considera anomalia lèxica si fase 5.1b l'ha marcada amb
    tokens C_RARE i/o amb runs test anòmal (apareix a candidats_steganografia).
    """
    doc = json.load(open(F5_1B_CRYPTO_VALIDATION, encoding="utf-8"))
    out: dict[tuple[str, int], dict] = {}
    for c in doc.get("candidats_steganografia", []):
        key = (c["teixit"], int(c["volta"]))
        out[key] = {
            "te_rare": bool(c.get("te_rare", False)),
            "n_rare": int(c.get("n_rare", 0)),
            "tokens_rare": list(c.get("tokens_rare", []) or []),
            "z_runs": c.get("z_runs"),
            "interpretacio_runs": c.get("interpretacio_runs"),
            "prioritat": c.get("prioritat"),
        }
    return out


def carrega_segments() -> dict:
    return json.load(open(F5_2_SEGMENTS_JSON, encoding="utf-8"))


# ───────────────────────── CLASSIFICACIÓ ─────────────────────────────────
def classe_volta(es_gram: bool, es_lex: bool) -> str:
    if es_gram and es_lex:
        return "CONVERGENT"
    if es_gram:
        return "GRAMATICAL"
    if es_lex:
        return "LEXICA"
    return "CAP"


def veredicte_segment(n_conv: int, n_gram: int, n_lex: int) -> tuple[str, str]:
    """Deriva el veredicte agregat del segment a partir dels comptadors
    de voltes per classe (n_gram i n_lex són exclusius de convergents)."""
    if n_conv >= 1:
        return ("CONVERGENCIA",
                "Hi ha voltes gramaticalment I lèxicament anòmales: inspecció "
                "prioritària (trenquen la gramàtica i porten marcadors rars).")
    if n_lex >= 1 and n_gram == 0:
        return ("PERFIL_LEXIC",
                "Senyal lèxic sense soroll gramatical: compatible amb codi "
                "camuflat → candidat a desxiframent en abecedari.")
    if n_gram >= 1 and n_lex == 0:
        return ("PERFIL_GRAMATICAL",
                "Senyal gramatical sense marcadors lèxics: compatible amb "
                "farciment estructural → improbable transcripció d'abecedari.")
    if n_gram >= 1 and n_lex >= 1:
        return ("MIXT",
                "Senyals gramaticals i lèxics presents en voltes diferents del "
                "segment: perfil ambigu, requereix lectura qualitativa.")
    return ("SENSE_SENYAL",
            "Segment d'alt score agregat però sense voltes marcades "
            "individualment per 5.1 ni 5.1b.")


# ───────────────────────── PROCÉS PRINCIPAL ──────────────────────────────
def main() -> None:
    ensure_dir(RES_FASE_5_2B)
    print("== FASE 5.2b · Triangulació gramatical × lèxica sobre segments ==")

    gram = carrega_gramaticals()
    lex = carrega_lexiques()
    doc_seg = carrega_segments()
    segments = doc_seg.get("segments", [])

    # Universos de voltes atípiques (per a la triangulació i les òrfenes)
    gram_keys = {k for k, v in gram.items() if v["categoria"] in GRAMMAR_CATS}
    lex_keys = {k for k, v in lex.items()
                if (v["te_rare"] or v["z_runs"] is not None)}
    atipiques = gram_keys | lex_keys

    print(f"   Voltes 5.1 avaluades       : {len(gram)}")
    print(f"   · gramaticals ({'/'.join(sorted(GRAMMAR_CATS))}): {len(gram_keys)}")
    print(f"   Voltes lèxiques (5.1b)     : {len(lex_keys)}")
    print(f"   Total voltes atípiques     : {len(atipiques)}")
    print(f"   Segments a triangular (5.2): {len(segments)}")

    cobertes: set[tuple[str, int]] = set()
    resultats: list[dict] = []
    for s in segments:
        teixit = s["teixit"]
        voltes_detall = s.get("voltes_detall") or []
        if not voltes_detall:
            # fallback: rang volta_inici..volta_final
            voltes_detall = [
                {"volta": v}
                for v in range(int(s["volta_inici"]), int(s["volta_final"]) + 1)
            ]

        detall: list[dict] = []
        tokens_rare_seg: list[str] = []
        n_conv = n_gram = n_lex = n_cap = 0

        for vd in voltes_detall:
            v = int(vd["volta"])
            key = (teixit, v)
            cobertes.add(key)
            g = gram.get(key)
            l = lex.get(key)

            es_gram = bool(g and g["categoria"] in GRAMMAR_CATS)
            es_lex = bool(l and (l["te_rare"] or l["z_runs"] is not None))
            cls = classe_volta(es_gram, es_lex)

            if cls == "CONVERGENT":
                n_conv += 1
            elif cls == "GRAMATICAL":
                n_gram += 1
            elif cls == "LEXICA":
                n_lex += 1
            else:
                n_cap += 1

            if es_lex and l:
                tokens_rare_seg.extend(l["tokens_rare"])

            detall.append({
                "volta": v,
                "classe": cls,
                # gramatical (5.1)
                "categoria_5_1": (g or {}).get("categoria"),
                "score_5_1": (g or {}).get("score_anomalia"),
                "flags_5_1": (g or {}).get("flags_actius", []),
                # lèxic (5.1b)
                "te_rare": bool(l and l["te_rare"]),
                "n_rare": (l or {}).get("n_rare", 0),
                "tokens_rare": (l or {}).get("tokens_rare", []),
                "z_runs": (l or {}).get("z_runs"),
                "interpretacio_runs": (l or {}).get("interpretacio_runs"),
            })

        ver, nota = veredicte_segment(n_conv, n_gram, n_lex)
        n_voltes = len(detall)
        rare_counter = Counter(tokens_rare_seg)
        # totals incloent convergents
        tot_gram = n_gram + n_conv
        tot_lex = n_lex + n_conv

        resultats.append({
            "teixit": teixit,
            "volta_inici": s["volta_inici"],
            "volta_final": s["volta_final"],
            "n_voltes": n_voltes,
            "score_mitja_5_2": s.get("score_mitja"),
            "q_valor_5_2": s.get("q_valor"),
            "significatiu_FDR_5_2": s.get("significatiu_FDR"),
            "n_convergents": n_conv,
            "n_gramaticals": n_gram,
            "n_lexiques": n_lex,
            "n_cap": n_cap,
            "frac_gramatical": round(tot_gram / n_voltes, 3) if n_voltes else 0.0,
            "frac_lexica": round(tot_lex / n_voltes, 3) if n_voltes else 0.0,
            "voltes_convergents": [d["volta"] for d in detall if d["classe"] == "CONVERGENT"],
            "voltes_gramaticals": [d["volta"] for d in detall if d["classe"] == "GRAMATICAL"],
            "voltes_lexiques": [d["volta"] for d in detall if d["classe"] == "LEXICA"],
            "tokens_rare_segment": [
                {"token": t, "freq": c} for t, c in rare_counter.most_common()
            ],
            "veredicte": ver,
            "nomes_senyal_agregat": bool(ver == "SENSE_SENYAL"),
            "nota": nota,
            "voltes_detall": detall,
        })

    # ordena: convergència primer, després perfil lèxic, etc.
    resultats.sort(key=lambda r: (
        ORDRE_VEREDICTE.get(r["veredicte"], 9),
        -r["n_convergents"],
        -r["frac_lexica"],
    ))

    counts_ver = Counter(r["veredicte"] for r in resultats)

    # ── VOLTES ATÍPIQUES ÒRFENES (atípiques però fora de qualsevol segment) ──
    orfenes_raw = sorted(atipiques - cobertes)
    ordre_classe = {"CONVERGENT": 0, "LEXICA": 1, "GRAMATICAL": 2}
    orfenes: list[dict] = []
    for (teixit, v) in orfenes_raw:
        g = gram.get((teixit, v))
        l = lex.get((teixit, v))
        es_gram = bool(g and g["categoria"] in GRAMMAR_CATS)
        es_lex = bool(l and (l["te_rare"] or l["z_runs"] is not None))
        cls = classe_volta(es_gram, es_lex)
        orfenes.append({
            "teixit": teixit,
            "volta": v,
            "classe": cls,
            "categoria_5_1": (g or {}).get("categoria"),
            "score_5_1": (g or {}).get("score_anomalia"),
            "flags_5_1": (g or {}).get("flags_actius", []),
            "te_rare": bool(l and l["te_rare"]),
            "n_rare": (l or {}).get("n_rare", 0),
            "tokens_rare": (l or {}).get("tokens_rare", []),
            "z_runs": (l or {}).get("z_runs"),
            "interpretacio_runs": (l or {}).get("interpretacio_runs"),
        })
    orfenes.sort(key=lambda o: (ordre_classe.get(o["classe"], 9),
                                o["teixit"], o["volta"]))
    counts_orf = Counter(o["classe"] for o in orfenes)

    doc = {
        "metadades": {
            "grammar_cats": sorted(GRAMMAR_CATS),
            "n_voltes_5_1_avaluades": len(gram),
            "n_voltes_gramaticals": len(gram_keys),
            "n_voltes_lexiques": len(lex_keys),
            "n_voltes_atipiques_total": len(atipiques),
            "n_voltes_atipiques_en_segment": len(atipiques & cobertes),
            "n_voltes_atipiques_orfenes": len(orfenes),
            "n_segments": len(resultats),
            "n_convergencia": counts_ver.get("CONVERGENCIA", 0),
            "n_perfil_lexic": counts_ver.get("PERFIL_LEXIC", 0),
            "n_mixt": counts_ver.get("MIXT", 0),
            "n_perfil_gramatical": counts_ver.get("PERFIL_GRAMATICAL", 0),
            "n_sense_senyal": counts_ver.get("SENSE_SENYAL", 0),
            "orfenes_per_classe": {
                "CONVERGENT": counts_orf.get("CONVERGENT", 0),
                "LEXICA": counts_orf.get("LEXICA", 0),
                "GRAMATICAL": counts_orf.get("GRAMATICAL", 0),
            },
            "interpretacio": (
                "Triangulació de 5.1 (anomalies gramaticals) i 5.1b (anomalies "
                "lèxiques) sobre les voltes dels segments de 5.2. CONVERGENCIA i "
                "PERFIL_LEXIC són els perfils compatibles amb un missatge "
                "desxifrable en abecedari; PERFIL_GRAMATICAL apunta a farciment "
                "estructural (no abecedari). SENSE_SENYAL = segment detectat "
                "només pel score agregat de 5.2, sense cap volta atípica "
                "individual (control). Les voltes òrfenes són atípiques que "
                "5.2 no va agrupar en cap segment (especialment rellevants les "
                "CONVERGENT aïllades)."
            ),
        },
        "resultats": resultats,
        "voltes_orfenes": orfenes,
    }

    F5_2B_TRIANGULACIO_JSON.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV pla
    cols = [
        "teixit", "volta_inici", "volta_final", "n_voltes",
        "n_convergents", "n_gramaticals", "n_lexiques", "n_cap",
        "frac_gramatical", "frac_lexica",
        "score_mitja_5_2", "q_valor_5_2", "significatiu_FDR_5_2",
        "veredicte",
    ]
    with open(F5_2B_TRIANGULACIO_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["voltes_convergents", "tokens_rare_segment"])
        for r in resultats:
            conv = "|".join(str(v) for v in r["voltes_convergents"])
            rares = "|".join(f"{d['token']}×{d['freq']}"
                             for d in r["tokens_rare_segment"])
            w.writerow([r[c] for c in cols] + [conv, rares])

    # CSV de voltes òrfenes (atípiques fora de segment)
    with open(F5_2B_ORFENES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teixit", "volta", "classe", "categoria_5_1", "score_5_1",
                    "n_rare", "tokens_rare", "z_runs", "flags_5_1"])
        for o in orfenes:
            w.writerow([
                o["teixit"], o["volta"], o["classe"],
                o["categoria_5_1"] or "", o["score_5_1"] if o["score_5_1"] is not None else "",
                o["n_rare"],
                "|".join(o["tokens_rare"]) if o["tokens_rare"] else "",
                o["z_runs"] if o["z_runs"] is not None else "",
                "|".join(o["flags_5_1"]) if o["flags_5_1"] else "",
            ])

    # Report llegible
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(" FASE 5.2b — TRIANGULACIÓ GRAMATICAL × LÈXICA SOBRE SEGMENTS")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Per a cada segment de 5.2, es classifica cada volta segons si")
    lines.append("és anòmala a 5.1 (gramatical), a 5.1b (lèxica), a totes dues")
    lines.append("(convergent) o cap. Veredicte agregat per segment:")
    lines.append("  · CONVERGENCIA      → ≥1 volta gramatical I lèxica")
    lines.append("  · PERFIL_LEXIC      → marcadors lèxics, gramàtica neta "
                 "(candidat abecedari)")
    lines.append("  · PERFIL_GRAMATICAL → soroll gramatical, sense marcadors "
                 "(farciment)")
    lines.append("  · MIXT / SENSE_SENYAL")
    lines.append("")
    md = doc["metadades"]
    lines.append(f"Voltes atípiques totals : {md['n_voltes_atipiques_total']}  "
                 f"(gramaticals={md['n_voltes_gramaticals']}, "
                 f"lèxiques={md['n_voltes_lexiques']})")
    lines.append(f"  · dins de segment     : {md['n_voltes_atipiques_en_segment']}")
    lines.append(f"  · òrfenes (fora seg.) : {md['n_voltes_atipiques_orfenes']}")
    lines.append("")
    lines.append(f"Segments triangulats : {md['n_segments']}")
    lines.append(f"  CONVERGENCIA       : {md['n_convergencia']}")
    lines.append(f"  PERFIL_LEXIC       : {md['n_perfil_lexic']}")
    lines.append(f"  MIXT               : {md['n_mixt']}")
    lines.append(f"  PERFIL_GRAMATICAL  : {md['n_perfil_gramatical']}")
    lines.append(f"  SENSE_SENYAL       : {md['n_sense_senyal']}  "
                 f"(només senyal agregat — control)")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"{'#':>3} {'teixit':<22} {'voltes':<14} {'nV':>3} "
                 f"{'conv':>4} {'gram':>4} {'lex':>4} "
                 f"{'fG':>5} {'fL':>5} {'veredicte':<18}")
    lines.append("-" * 78)
    for i, r in enumerate(resultats, 1):
        rng_s = f"v{r['volta_inici']}-v{r['volta_final']}"
        lines.append(
            f"{i:>3} {r['teixit'][:22]:<22} {rng_s:<14} {r['n_voltes']:>3} "
            f"{r['n_convergents']:>4} {r['n_gramaticals']:>4} {r['n_lexiques']:>4} "
            f"{r['frac_gramatical']:>5.2f} {r['frac_lexica']:>5.2f} "
            f"{r['veredicte']:<18}"
        )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" DETALL PER SEGMENT")
    lines.append("-" * 78)
    for i, r in enumerate(resultats, 1):
        lines.append("")
        lines.append(f" [{i}] {r['teixit']}  v{r['volta_inici']}-v{r['volta_final']}  "
                     f"({r['n_voltes']} voltes)  →  {r['veredicte']}")
        sig = "YES" if r["significatiu_FDR_5_2"] else "no"
        q = r["q_valor_5_2"]
        q_s = f"{q:.3f}" if isinstance(q, (int, float)) else str(q)
        lines.append(f"     fase 5.2 sig FDR : {sig}  (q={q_s})")
        lines.append(f"     classes voltes   : convergents={r['n_convergents']}  "
                     f"gramaticals={r['n_gramaticals']}  "
                     f"lèxiques={r['n_lexiques']}  cap={r['n_cap']}")
        if r["voltes_convergents"]:
            lines.append(f"     voltes CONVERGENT: "
                         f"{', '.join('v'+str(v) for v in r['voltes_convergents'])}")
        if r["voltes_lexiques"]:
            lines.append(f"     voltes LEXICA    : "
                         f"{', '.join('v'+str(v) for v in r['voltes_lexiques'])}")
        if r["voltes_gramaticals"]:
            lines.append(f"     voltes GRAMATICAL: "
                         f"{', '.join('v'+str(v) for v in r['voltes_gramaticals'])}")
        if r["tokens_rare_segment"]:
            rares = ", ".join(f"{d['token']}×{d['freq']}"
                              for d in r["tokens_rare_segment"])
            lines.append(f"     tokens C_RARE    : {rares}")
        lines.append(f"     >> {r['nota']}")

    # ── Voltes atípiques òrfenes ──
    lines.append("")
    lines.append("=" * 78)
    lines.append(" VOLTES ATÍPIQUES ÒRFENES (fora de qualsevol segment de 5.2)")
    lines.append("=" * 78)
    of = md["orfenes_per_classe"]
    lines.append(f" CONVERGENT (5.1∩5.1b) : {of['CONVERGENT']}   "
                 f"← les més rellevants (anòmales però 5.2 no les va agrupar)")
    lines.append(f" LEXICA (només 5.1b)   : {of['LEXICA']}")
    lines.append(f" GRAMATICAL (només 5.1): {of['GRAMATICAL']}")
    lines.append("")
    if orfenes:
        lines.append(f"{'teixit':<22} {'volta':>6} {'classe':<11} "
                     f"{'cat_5.1':<9} {'n_rare':>6}  tokens_rare / flags")
        lines.append("-" * 78)
        for o in orfenes:
            extra = ("|".join(o["tokens_rare"]) if o["tokens_rare"]
                     else "|".join(o["flags_5_1"]))
            lines.append(
                f"{o['teixit'][:22]:<22} {o['volta']:>6} {o['classe']:<11} "
                f"{(o['categoria_5_1'] or '-'):<9} {o['n_rare']:>6}  {extra}"
            )
    else:
        lines.append(" (cap volta atípica fora de segment)")

    lines.append("")
    F5_2B_TRIANGULACIO_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"   Convergència      : {md['n_convergencia']}")
    print(f"   Perfil lèxic      : {md['n_perfil_lexic']}")
    print(f"   Perfil gramatical : {md['n_perfil_gramatical']}")
    print(f"   Mixt / sense      : {md['n_mixt']} / {md['n_sense_senyal']}")
    print(f"   Voltes òrfenes    : {md['n_voltes_atipiques_orfenes']} "
          f"(conv={of['CONVERGENT']}, lex={of['LEXICA']}, gram={of['GRAMATICAL']})")
    print()
    print(f"   JSON   → {F5_2B_TRIANGULACIO_JSON.name}")
    print(f"   CSV    → {F5_2B_TRIANGULACIO_CSV.name}")
    print(f"   ÒRFENES→ {F5_2B_ORFENES_CSV.name}")
    print(f"   REPORT → {F5_2B_TRIANGULACIO_REPORT.name}")
    print("== FASE 5.2b COMPLETADA ==")


if __name__ == "__main__":
    main()
