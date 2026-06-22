# -*- coding: utf-8 -*-
"""
FASE 6.2 — Visualitzacions del pipeline distribucional
=====================================================

Genera tots els PNG d'anàlisi a partir dels outputs de Fases 3.x, 4.x i 5.
No conté cap suposició semàntica: tot ve de les classes/tipus emergents.

Carpetes generades a res_fase6/:
  01_token_classes/      (Fase 3.2) — 1 PNG: pca_tokens
  03_volta_types/        (Fase 3.4) — 1 PNG: sequencies_per_teixit
  04_grammar_macro/      (Fase 4.1) — 1 PNG: transition_matrix_macro
  05_grammar_micro/      (Fase 4.2) — 1 PNG: transition_matrix_micro
  06_anomalies/          (Fase 5.1)  — 3 PNG: per_teixit, scatter, top20
  07_segments_estegano/  (Fase 5.2)  — 3 PNG: detall_segments_significatius, panel_segments_per_teixit, ranquing_segments
  08_crypto_segments/    (Fase 5.2b) — 2 PNG: mapa_per_teixit, veredicte_resum
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    BASE, RES_FASE_6,
    F3_2_DISTRIB_CLASSES, F3_4_VOLTA_CLASSES, F3_4_SEQUENCIES_TEIXITS,
    F4_1_MACRO_GRAMMAR, F4_2_MICRO_GRAMMAR, F5_1_ANOMALY_REPORT_JSON,
    F5_2_SEGMENTS_JSON, F5_2B_TRIANGULACIO_JSON,
)

OUT_PNG = RES_FASE_6

# Carpetes per fase
DIRS = {
    "tokens":     OUT_PNG / "01_token_classes",
    "vtypes":     OUT_PNG / "03_volta_types",
    "macro":      OUT_PNG / "04_grammar_macro",
    "micro":      OUT_PNG / "05_grammar_micro",
    "anomalies":  OUT_PNG / "06_anomalies",
    "segments":   OUT_PNG / "07_segments_estegano",
    "crypto":     OUT_PNG / "08_crypto_segments",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Inputs
IN_CLASSES = F3_2_DISTRIB_CLASSES
IN_VTYPES  = F3_4_VOLTA_CLASSES
IN_SEQS    = F3_4_SEQUENCIES_TEIXITS
IN_MACRO   = F4_1_MACRO_GRAMMAR
IN_MICRO   = F4_2_MICRO_GRAMMAR
IN_ANOM    = F5_1_ANOMALY_REPORT_JSON
IN_SEG     = F5_2_SEGMENTS_JSON
IN_CRYPTO  = F5_2B_TRIANGULACIO_JSON

sns.set_theme(style="white", palette="muted")
DPI = 150


# ===========================================================================
# 1) Token classes  (Fase 3.2)
# ===========================================================================
def viz_token_classes():
    print("[1/5] Token classes...")
    with open(IN_CLASSES, "r", encoding="utf-8") as f:
        doc = json.load(f)

    tokens = doc["tokens"]
    classes = sorted(doc["classes"].keys(),
                     key=lambda c: (0, int(c[1:])) if c[1:].isdigit() else (1, c))
    palette = sns.color_palette("tab20", len(classes))
    cls_to_color = dict(zip(classes, palette))

    # PCA 2D scatter — MILLORAT:
    # · Etiquetes amb caixa blanca per llegibilitat
    # · Marcadors més grans per outliers (estrelles)
    # · Centroides de classe destacats amb halo
    # · Línies tènues del centroide cap a cada membre per veure pertinença
    fig, ax = plt.subplots(figsize=(14, 10))

    def _has_xy(xy):
        return xy is not None and len(xy) == 2 and xy[0] is not None and xy[1] is not None

    # Primer dibuixem línies tènues centroide -> membres
    for c in classes:
        cxy = doc["classes"][c].get("centroide_2d")
        if not _has_xy(cxy):
            continue  # C_RARE: tokens fora del clustering, sense PCA
        cx, cy = cxy
        col = cls_to_color[c]
        for tok in doc["classes"][c]["membres"]:
            if tok in tokens:
                txy = tokens[tok].get("vector_2d")
                if not _has_xy(txy):
                    continue
                tx, ty = txy
                ax.plot([cx, tx], [cy, ty], color=col, lw=0.4, alpha=0.25, zorder=1)

    # Punts dels tokens
    for tok, info in tokens.items():
        xy = info.get("vector_2d")
        if not _has_xy(xy):
            continue
        x, y = xy
        c = cls_to_color[info["classe"]]
        if info.get("es_outlier"):
            ax.scatter(x, y, c=[c], s=140, marker="*",
                       edgecolors="black", linewidths=1.0, zorder=3)
        else:
            ax.scatter(x, y, c=[c], s=90, marker="o",
                       edgecolors="black", linewidths=0.4, zorder=3)
        ax.annotate(tok, (x, y), fontsize=8, xytext=(5, 5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85),
                    zorder=4)

    # Centroides amb halo
    for c in classes:
        cxy = doc["classes"][c].get("centroide_2d")
        if not _has_xy(cxy):
            continue
        cx, cy = cxy
        col = cls_to_color[c]
        ax.scatter(cx, cy, c=[col], s=380, marker="o", alpha=0.25,
                   edgecolors="none", zorder=2)
        ax.text(cx, cy, c, ha="center", va="center", fontsize=11,
                fontweight="bold", color="black", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=col, linewidth=1.5))

    handles = [mpatches.Patch(color=cls_to_color[c],
                              label=f"{c} ({doc['classes'][c]['num_tokens']} tokens)")
               for c in classes]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="gray", markeredgecolor="black",
                              markersize=14, label="outlier (z>2)", linewidth=0))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True)
    ax.set_title("Classes distribucionals de tokens (PCA 2D)\n"
                 "Caixes blanques = tokens · cercles grans = centroides de classe · "
                 "estrelles = outliers",
                 fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(DIRS["tokens"] / "pca_tokens.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ===========================================================================
# 2) Volta types — stripe plots  (Fase 3.4)
# ===========================================================================
def viz_volta_types():
    print("[2/5] Volta types (stripe plots)...")
    with open(IN_SEQS, "r", encoding="utf-8") as f:
        seqs_doc = json.load(f)

    alphabet = seqs_doc["metadades"]["alfabet"]
    type_to_idx = {t: i for i, t in enumerate(alphabet)}
    palette = sns.color_palette("tab10", len(alphabet))

    # Plot únic apilat amb tots els teixits — MILLORAT:
    # · Files separades amb gap
    # · Etiquetes de volta cada 10 posicions
    # · Marques verticals cada 10 posicions
    teixits = list(seqs_doc["sequencies"].keys())
    max_len = max(len(s["cadena"]) for s in seqs_doc["sequencies"].values())

    row_h = 0.85   # alçada de la barra
    row_gap = 0.25 # separació entre files
    row_total = row_h + row_gap

    fig, ax = plt.subplots(
        figsize=(min(22, 6 + max_len * 0.10),
                 row_total * len(teixits) + 1.8)
    )
    for row, teixit in enumerate(teixits):
        y0 = row * row_total
        seq = seqs_doc["sequencies"][teixit]["cadena"]
        for col, tipus in enumerate(seq):
            color = palette[type_to_idx.get(tipus, len(alphabet) - 1)]
            ax.add_patch(plt.Rectangle((col, y0), 1, row_h,
                                       facecolor=color, edgecolor="white",
                                       linewidth=0.3))
        ax.text(-1.0, y0 + row_h / 2, teixit, ha="right", va="center",
                fontsize=10, fontweight="bold")
        # Etiqueta amb longitud al final
        ax.text(len(seq) + 0.5, y0 + row_h / 2, f"n={len(seq)}",
                ha="left", va="center", fontsize=8, color="#666")

    # Reixeta de referència vertical cada 10 voltes
    for x in range(0, max_len + 1, 10):
        ax.axvline(x, color="#cccccc", lw=0.4, ls=":", zorder=0)
        ax.text(x, -0.6, str(x), ha="center", va="top", fontsize=7, color="#888")

    ax.set_xlim(-15, max_len + 8)
    ax.set_ylim(-1.5, row_total * len(teixits) + 0.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("volta (posició dins del teixit)")
    ax.set_title("Seqüències de tipus de volta per teixit\n"
                 "Cada color = tipus VTx · barres verticals cada 10 voltes",
                 fontsize=12)
    handles = [mpatches.Patch(color=palette[i], label=t)
               for i, t in enumerate(alphabet)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.06),
              fontsize=9, ncol=min(len(alphabet), 8), frameon=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(DIRS["vtypes"] / "sequencies_per_teixit.png", dpi=DPI,
                bbox_inches="tight")
    plt.close()


# ===========================================================================
# 3) Macro grammar transition matrix  (Fase 4.1)
# ===========================================================================
def viz_macro_grammar():
    print("[3/5] Macro grammar...")
    with open(IN_MACRO, "r", encoding="utf-8") as f:
        doc = json.load(f)
    alphabet = doc["metadades"]["alfabet"]  # inclou possible VOID
    symbols = [s for s in alphabet if s != "VOID"]

    probs = doc["model"]["transition_probabilities"]
    targets = symbols + ["<EOS>"]
    contexts = ["<BOS>"] + symbols

    M = np.zeros((len(contexts), len(targets)))
    for i, ctx in enumerate(contexts):
        row = probs.get(ctx, {})
        for j, sym in enumerate(targets):
            M[i, j] = row.get(sym, 0.0)

    # Annotacions: només mostrem nombres on P >= 0.05 (filtre soroll)
    annot = np.where(M >= 0.05, np.round(M, 2).astype(object), "")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(M, annot=annot, fmt="", cmap="rocket_r",
                xticklabels=targets, yticklabels=contexts,
                linewidths=0.5, linecolor="#eeeeee",
                cbar_kws={"label": "P(símbol | context)"},
                annot_kws={"size": 10, "weight": "bold"})
    ax.set_title(
        "Macro: matriu de transició bigrama (smoothed)\n"
        "Llegir: P(columna | fila) · només s'anoten valors ≥ 0.05\n"
        "VTx-en-teixit (Fase 4.1)",
        fontsize=11)
    ax.set_xlabel("→ següent (VTx o EOS)")
    ax.set_ylabel("← context (VTx o BOS)")
    plt.tight_layout()
    plt.savefig(DIRS["macro"] / "transition_matrix_macro.png", dpi=DPI)
    plt.close()


# ===========================================================================
# 4) Micro grammar transition matrix  (Fase 4.2)
# ===========================================================================
def viz_micro_grammar():
    print("[4/5] Micro grammar...")
    with open(IN_MICRO, "r", encoding="utf-8") as f:
        doc = json.load(f)
    symbols = doc["metadades"]["alfabet"]
    probs = doc["model"]["transition_probabilities"]
    targets = symbols + ["<EOS>"]
    contexts = ["<BOS>"] + symbols

    M = np.zeros((len(contexts), len(targets)))
    for i, ctx in enumerate(contexts):
        row = probs.get(ctx, {})
        for j, sym in enumerate(targets):
            M[i, j] = row.get(sym, 0.0)

    # Annotacions només per P >= 0.05 (la resta queda buida → tinta només on importa)
    annot = np.where(M >= 0.05, np.round(M, 2).astype(object), "")

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(M, annot=annot, fmt="", cmap="rocket_r",
                xticklabels=targets, yticklabels=contexts,
                linewidths=0.5, linecolor="#eeeeee",
                annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"label": "P(símbol | context)"})
    ax.set_title(
        "Micro: matriu de transició bigrama (smoothed)\n"
        "Llegir: P(columna | fila) · només s'anoten valors ≥ 0.05\n"
        "Cx-en-volta (Fase 4.2)",
        fontsize=11)
    ax.set_xlabel("→ següent (Cx o EOS)")
    ax.set_ylabel("← context (Cx o BOS)")
    plt.tight_layout()
    plt.savefig(DIRS["micro"] / "transition_matrix_micro.png", dpi=DPI)
    plt.close()


# ===========================================================================
# 5) Anomaly report  (Fase 5)
# ===========================================================================
def viz_anomalies():
    print("[5/5] Anomalies...")
    with open(IN_ANOM, "r", encoding="utf-8") as f:
        doc = json.load(f)

    records = doc["anomalies"]
    per_teixit = doc["agregat_per_teixit"]

    # 1) Score total per teixit (stacked: CRÍTICA / ALTA / MODERADA)
    teixits = sorted(per_teixit.keys(),
                     key=lambda t: per_teixit[t]["score_total"], reverse=True)
    crit = [per_teixit[t]["n_CRITICA"] for t in teixits]
    alta = [per_teixit[t]["n_ALTA"] for t in teixits]
    mod  = [per_teixit[t]["n_MODERADA"] for t in teixits]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(teixits))
    ax.bar(x, crit, color="#cc0000", label="CRÍTICA")
    ax.bar(x, alta, bottom=crit, color="#ee9900", label="ALTA")
    ax.bar(x, mod, bottom=np.array(crit) + np.array(alta),
           color="#dddd55", label="MODERADA")
    ax.set_xticks(x); ax.set_xticklabels(teixits, rotation=30, ha="right")
    ax.set_title("Anomalies per teixit (recompte)")
    ax.set_ylabel("# voltes anòmales")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DIRS["anomalies"] / "anomalies_per_teixit.png", dpi=DPI)
    plt.close()

    # 2) Scatter z_micro vs z_macro (Fase 5: detecció cross-canal)
    xs, ys, cats, recs = [], [], [], []
    for r in records:
        zm = r.get("z_surprisal_macro")
        zu = r.get("z_mean_surprisal_micro")
        if zm is None or zu is None:
            continue
        xs.append(zm); ys.append(zu); cats.append(r["categoria"]); recs.append(r)
    cat_colors = {"CRÍTICA": "#cc0000", "ALTA": "#ee9900",
                  "MODERADA": "#999933", "normal": "#bbbbbb"}
    cat_sizes = {"CRÍTICA": 90, "ALTA": 55, "MODERADA": 28, "normal": 14}
    cat_alpha = {"CRÍTICA": 0.95, "ALTA": 0.85, "MODERADA": 0.55, "normal": 0.20}

    fig, ax = plt.subplots(figsize=(11, 9))

    # Quadrants ombrejats per ajudar a llegir la imatge
    ax.axhspan(2, max(ys + [3]) + 0.5, xmin=0.5, alpha=0.05, color="red", zorder=0)
    ax.axvspan(2, max(xs + [3]) + 0.5, ymin=0.5, alpha=0.05, color="red", zorder=0)

    # Punts
    for cat in ("normal", "MODERADA", "ALTA", "CRÍTICA"):
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                   c=cat_colors[cat], label=f"{cat} (n={len(idx)})",
                   s=cat_sizes[cat], alpha=cat_alpha[cat],
                   edgecolors="black" if cat in ("CRÍTICA", "ALTA") else "none",
                   linewidths=0.5)

    # Etiquetes per les anomalies CRÍTICA i ALTA
    label_idx = [i for i, c in enumerate(cats) if c in ("CRÍTICA", "ALTA")]
    for i in label_idx:
        rec = recs[i]
        label = f"{rec['teixit'][:8]} v{rec['volta']}"
        ax.annotate(label, (xs[i], ys[i]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="#888",
                              alpha=0.85, linewidth=0.5))

    ax.axhline(2, color="black", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(2, color="black", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(0, color="#bbbbbb", lw=0.4)
    ax.axvline(0, color="#bbbbbb", lw=0.4)
    ax.set_xlabel("Z(surprisal MACRO) → quant inesperat és el tipus de volta\n"
                  "dins del seu teixit", fontsize=10)
    ax.set_ylabel("Z(H̄ MICRO) → quant inesperada és la composició\n"
                  "interna de la volta", fontsize=10)
    ax.set_title(
        "Detecció d'anomalies en dos canals independents\n"
        "Quadrant superior-dret (z≥2 als dos eixos) = senyal més fiable",
        fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(DIRS["anomalies"] / "scatter_micro_vs_macro.png", dpi=DPI)
    plt.close()

    # 3) Top-20 voltes amb score més alt (barres horitzontals)
    top = records[:20]  # ja venen ordenades
    labels = [f"{r['teixit']} v{r['volta']}" for r in top]
    scores = [r["score_anomalia"] for r in top]
    colors = [cat_colors.get(r["categoria"], "#888") for r in top]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top))
    ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("score d'anomalia")
    ax.set_title("Top-20 voltes per score d'anomalia")
    handles = [mpatches.Patch(color=cat_colors[c], label=c)
               for c in ("CRÍTICA", "ALTA", "MODERADA", "normal")]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(DIRS["anomalies"] / "top20_anomalies.png", dpi=DPI)
    plt.close()


# ===========================================================================
# 6) Segments anòmals contigus  (Fase 5.7 — candidats a esteganografia)
# ===========================================================================
def viz_segments():
    print("[6/6] Segments anòmals contigus (Fase 5.7)...")
    with open(IN_ANOM, "r", encoding="utf-8") as f:
        anom_doc = json.load(f)
    with open(IN_SEG, "r", encoding="utf-8") as f:
        seg_doc = json.load(f)

    threshold = seg_doc["metadades"]["threshold_score"]
    pct = seg_doc["metadades"]["threshold_percentile"]
    alpha = seg_doc["metadades"]["alpha_fdr"]
    segments = seg_doc["segments"]

    # Construïm sèrie completa de scores per teixit (de res_fase5)
    series: dict[str, list[dict]] = defaultdict(list)
    for r in anom_doc["anomalies"]:
        series[r["teixit"]].append({
            "volta": r["volta"],
            "score": r["score_anomalia"],
            "categoria": r["categoria"],
        })
    for t in series:
        series[t].sort(key=lambda x: x["volta"])

    # Agrupem segments per teixit
    seg_per_teixit: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        seg_per_teixit[s["teixit"]].append(s)

    cat_color = {
        "CRÍTICA": "#cc0000", "ALTA": "#ee9900",
        "MODERADA": "#dddd55", "normal": "#cccccc",
    }
    BAND_SIG = "#22aa55"   # verd: q ≤ α
    BAND_NS  = "#888888"   # gris: candidat no significatiu

    # ── A) Panel global: una fila per teixit, sèrie temporal de score ──
    teixits = sorted(series.keys())
    n_t = len(teixits)
    fig, axes = plt.subplots(n_t, 1, figsize=(15, max(8, 1.3 * n_t)),
                             sharex=False)
    if n_t == 1:
        axes = [axes]

    for ax, teixit in zip(axes, teixits):
        recs = series[teixit]
        xs = [r["volta"] for r in recs]
        ys = [r["score"] for r in recs]

        # Bandes de segments (sota la línia)
        for s in seg_per_teixit.get(teixit, []):
            color = BAND_SIG if s["significatiu_FDR"] else BAND_NS
            alpha_band = 0.30 if s["significatiu_FDR"] else 0.15
            ax.axvspan(s["volta_inici"] - 0.5, s["volta_final"] + 0.5,
                       color=color, alpha=alpha_band, zorder=1)
            # Etiqueta amb q-valor i longitud
            mid = (s["volta_inici"] + s["volta_final"]) / 2
            tag = f"L={s['n_voltes']} q={s['q_valor']:.3f}"
            if s["significatiu_FDR"]:
                tag += " *"
            ax.text(mid, max(ys) * 1.02, tag,
                    ha="center", va="bottom", fontsize=7,
                    color=("#0a5520" if s["significatiu_FDR"] else "#444"),
                    fontweight="bold" if s["significatiu_FDR"] else "normal")

        # Línia de score
        ax.plot(xs, ys, color="#222244", lw=0.8, alpha=0.7, zorder=2)
        # Punts colorits per categoria
        for r in recs:
            ax.plot(r["volta"], r["score"], "o",
                    color=cat_color.get(r["categoria"], "#999"),
                    markersize=3.2,
                    markeredgecolor="black" if r["categoria"] in ("CRÍTICA", "ALTA") else "none",
                    markeredgewidth=0.4, zorder=3)

        # Llindar
        ax.axhline(threshold, color="#aa2222", ls="--", lw=0.6,
                   alpha=0.7, zorder=1)

        ax.set_ylabel(teixit, fontsize=8.5, rotation=0,
                      ha="right", va="center", labelpad=42)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    axes[-1].set_xlabel("volta", fontsize=10)

    # Llegenda global
    handles = [
        mpatches.Patch(color=BAND_SIG, alpha=0.30,
                       label=f"segment significatiu (q ≤ {alpha})"),
        mpatches.Patch(color=BAND_NS, alpha=0.15,
                       label="segment candidat (q > α)"),
        plt.Line2D([0], [0], color="#aa2222", ls="--", lw=0.8,
                   label=f"llindar score (P{pct} = {threshold:.2f})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_color["CRÍTICA"],
                   markeredgecolor="black", markersize=6, label="CRÍTICA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_color["ALTA"],
                   markeredgecolor="black", markersize=6, label="ALTA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_color["MODERADA"],
                   markersize=6, label="MODERADA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_color["normal"],
                   markersize=6, label="normal"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        "Fase 5.7 — Sèrie de score d'anomalia per teixit · "
        "bandes = segments contigus detectats",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    plt.savefig(DIRS["segments"] / "panel_segments_per_teixit.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()

    # ── B) Zoom dels segments significatius ──
    sig = [s for s in segments if s["significatiu_FDR"]]
    if sig:
        n_s = len(sig)
        fig, axes = plt.subplots(n_s, 1, figsize=(11, 3.0 * n_s))
        if n_s == 1:
            axes = [axes]
        for ax, s in zip(axes, sig):
            voltes = [d["volta"] for d in s["voltes_detall"]]
            scores = [d["score_anomalia"] for d in s["voltes_detall"]]
            cats   = [d["categoria"] for d in s["voltes_detall"]]
            colors = [cat_color.get(c, "#999") for c in cats]

            x = np.arange(len(voltes))
            ax.bar(x, scores, color=colors, edgecolor="black",
                   linewidth=0.5, zorder=2)
            ax.axhline(threshold, color="#aa2222", ls="--", lw=0.8, alpha=0.8)

            # Etiquetes sobre cada barra
            for xi, (v, sc, cat, d) in enumerate(zip(voltes, scores, cats,
                                                      s["voltes_detall"])):
                tipus = d["tipus_volta"]
                ax.text(xi, sc + 0.15, f"{sc:.1f}",
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
                ax.text(xi, -0.6, f"v{v}\n{tipus}",
                        ha="center", va="top", fontsize=7, color="#444")

            ax.set_xticks([])
            ax.set_xlim(-0.6, len(voltes) - 0.4)
            ax.set_ylim(-1.6, max(scores) * 1.18 + 0.3)
            ax.set_ylabel("score", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)

            ax.set_title(
                f"{s['teixit']}   ·   v{s['volta_inici']}–v{s['volta_final']}   "
                f"(L={s['n_voltes']}, mean={s['score_mitja']:.2f}, "
                f"sum={s['score_acumulat']:.1f}, "
                f"p={s['p_valor']:.3f}, q={s['q_valor']:.3f})",
                fontsize=10, loc="left",
            )

        handles = [
            mpatches.Patch(color=cat_color["CRÍTICA"], label="CRÍTICA"),
            mpatches.Patch(color=cat_color["ALTA"], label="ALTA"),
            mpatches.Patch(color=cat_color["MODERADA"], label="MODERADA"),
            mpatches.Patch(color=cat_color["normal"], label="normal"),
            plt.Line2D([0], [0], color="#aa2222", ls="--", lw=0.8,
                       label=f"llindar (P{pct})"),
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0),
                   ncol=5, fontsize=8.5, frameon=False)
        fig.suptitle(
            "Fase 5.7 — Detall dels segments significatius (q ≤ α)",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(DIRS["segments"] / "detall_segments_significatius.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()

    # ── C) Bar chart resum: score acumulat per segment, ordenat ──
    if segments:
        ordered = sorted(segments, key=lambda s: -s["score_acumulat"])
        labels = [f"{s['teixit'][:14]} v{s['volta_inici']}-v{s['volta_final']}"
                  for s in ordered]
        sums = [s["score_acumulat"] for s in ordered]
        colors = [BAND_SIG if s["significatiu_FDR"] else BAND_NS
                  for s in ordered]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(ordered))))
        y = np.arange(len(ordered))
        ax.barh(y, sums, color=colors, edgecolor="black", linewidth=0.4)
        for yi, s in enumerate(ordered):
            ax.text(sums[yi] + 0.4, yi,
                    f"q={s['q_valor']:.3f}  L={s['n_voltes']}",
                    va="center", fontsize=7,
                    color="#0a5520" if s["significatiu_FDR"] else "#444")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("score d'anomalia acumulat (Σ del segment)")
        ax.set_title(
            f"Fase 5.7 — Rànquing de segments candidats "
            f"(verd = significatiu FDR α={alpha})",
            fontsize=11,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(DIRS["segments"] / "ranquing_segments.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()


# ===========================================================================
# 7) Triangulació gramatical × lèxica per segment  (Fase 5.2b)
# ===========================================================================
def viz_crypto():
    print("[7/7] Triangulació gramatical × lèxica (Fase 5.2b)...")
    if not IN_CRYPTO.exists():
        print("   (saltat: encara no s'ha executat fase5.2b)")
        return
    with open(IN_CRYPTO, "r", encoding="utf-8") as f:
        doc = json.load(f)
    res = doc["resultats"]
    orfenes = doc.get("voltes_orfenes", [])
    if not res and not orfenes:
        print("   (cap segment ni volta a visualitzar)")
        return

    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    veredicte_color = {
        "CONVERGENCIA":      "#22aa55",
        "PERFIL_LEXIC":      "#3377cc",
        "MIXT":              "#dd9900",
        "PERFIL_GRAMATICAL": "#aa4444",
        "SENSE_SENYAL":      "#bbbbbb",
    }
    veredicte_tag = {
        "CONVERGENCIA": "CONV", "PERFIL_LEXIC": "LÈX", "MIXT": "MIXT",
        "PERFIL_GRAMATICAL": "GRAM", "SENSE_SENYAL": "—",
    }
    # marker, color, mida per classe de volta
    marker_style = {
        "CONVERGENT": ("*", "#117733", 280),
        "LEXICA":     ("o", "#3377cc", 90),
        "GRAMATICAL": ("^", "#cc3333", 90),
    }

    # ── Construeix dades per teixit ──
    per_teixit: dict[str, dict] = defaultdict(
        lambda: {"segments": [], "voltes": []})
    for r in res:
        t = r["teixit"]
        per_teixit[t]["segments"].append(
            (int(r["volta_inici"]), int(r["volta_final"]), r["veredicte"]))
        for d in r.get("voltes_detall", []):
            if d["classe"] != "CAP":
                per_teixit[t]["voltes"].append((int(d["volta"]), d["classe"], True))
    for o in orfenes:
        per_teixit[o["teixit"]]["voltes"].append(
            (int(o["volta"]), o["classe"], False))

    teixits = sorted(per_teixit.keys())
    n = len(teixits)

    # ── A) MAPA PER TEIXIT (ideograma) ──
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3.0, 0.75 * n)),
                             squeeze=False)
    axes = axes[:, 0]
    for ax, t in zip(axes, teixits):
        data = per_teixit[t]
        # rang de l'eix x
        xs_all = [v for v, _, _ in data["voltes"]]
        for vi, vf, _ in data["segments"]:
            xs_all += [vi, vf]
        if not xs_all:
            xs_all = [0, 1]
        xmin, xmax = min(xs_all), max(xs_all)
        pad = max(2, int(0.04 * (xmax - xmin + 1)))

        # bandes de segment
        for vi, vf, ver in data["segments"]:
            ax.axvspan(vi - 0.5, vf + 0.5,
                       color=veredicte_color.get(ver, "#bbbbbb"), alpha=0.20)
            ax.text((vi + vf) / 2.0, 0.92, veredicte_tag.get(ver, ""),
                    ha="center", va="center", fontsize=6.5,
                    color=veredicte_color.get(ver, "#666"), fontweight="bold")

        # rails de les dues files: y=+0.5 dins de segment, y=-0.5 òrfenes
        Y_DINS, Y_ORF = 0.5, -0.5
        ax.axhline(Y_DINS, color="#e9e9e9", lw=0.7, zorder=0)
        ax.axhline(Y_ORF, color="#e9e9e9", lw=0.7, zorder=0)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.004, Y_DINS, "dins", transform=trans, fontsize=5.5,
                color="#999", va="center", ha="left")
        ax.text(1.004, Y_ORF, "òrf.", transform=trans, fontsize=5.5,
                color="#999", va="center", ha="left")

        # marcadors de volta (separats per fila segons in_segment)
        for in_seg in (True, False):
            y = Y_DINS if in_seg else Y_ORF
            for classe in ("GRAMATICAL", "LEXICA", "CONVERGENT"):
                xs = [v for v, c, s in data["voltes"]
                      if c == classe and s == in_seg]
                if not xs:
                    continue
                mk, col, size = marker_style[classe]
                ax.scatter(
                    xs, [y] * len(xs), marker=mk, s=size, c=col,
                    edgecolors="white", linewidths=0.5,
                    zorder=4 if classe == "CONVERGENT" else 3,
                )

        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylabel(t, rotation=0, ha="right", va="center", fontsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
    axes[-1].set_xlabel("número de volta", fontsize=9)

    # llegenda global
    leg_classe = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#117733",
               markersize=15, label="CONVERGENT (5.1 ∩ 5.1b)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3377cc",
               markersize=9, label="LÈXICA (5.1b)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#cc3333",
               markersize=9, label="GRAMATICAL (5.1)"),
        mpatches.Patch(facecolor="#22aa55", alpha=0.3, label="banda = segment 5.2"),
    ]
    fig.legend(handles=leg_classe, loc="upper center", ncol=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "Fase 5.2b — Mapa d'anomalies per teixit: voltes gramaticals (5.1), "
        "lèxiques (5.1b) i convergents.\nA cada teixit, la fila superior «dins» "
        "són voltes dins d'un segment de 5.2; la fila inferior «òrf.» són voltes "
        "atípiques que cauen FORA de tot segment",
        fontsize=11, y=1.05,
    )
    plt.tight_layout()
    plt.savefig(DIRS["crypto"] / "mapa_per_teixit.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()

    # ── B) Resum global: recompte de veredictes + òrfenes ──
    ordre = ["CONVERGENCIA", "PERFIL_LEXIC", "MIXT",
             "PERFIL_GRAMATICAL", "SENSE_SENYAL"]
    cnt = {k: 0 for k in ordre}
    for r in res:
        cnt[r["veredicte"]] = cnt.get(r["veredicte"], 0) + 1
    of = doc.get("metadades", {}).get("orfenes_per_classe", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
    vals = [cnt[k] for k in ordre]
    cols = [veredicte_color[k] for k in ordre]
    bars = ax1.barh(ordre, vals, color=cols, edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax1.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("nombre de segments")
    ax1.set_title(f"Veredicte dels {len(res)} segments (Fase 5.2b)", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.4)

    orf_keys = ["CONVERGENT", "LEXICA", "GRAMATICAL"]
    orf_vals = [of.get(k, 0) for k in orf_keys]
    orf_cols = [marker_style[k][1] for k in orf_keys]
    bars2 = ax2.barh(orf_keys, orf_vals, color=orf_cols,
                     edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars2, orf_vals):
        ax2.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("nombre de voltes")
    ax2.set_title("Voltes atípiques ÒRFENES (fora de segment)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(DIRS["crypto"] / "veredicte_resum.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()


# ===========================================================================
def main():
    viz_token_classes()
    viz_volta_types()
    viz_macro_grammar()
    viz_micro_grammar()
    viz_anomalies()
    viz_segments()
    viz_crypto()
    print("\nFASE 6 COMPLETADA. Carpetes generades:")
    for k, d in DIRS.items():
        n = len(list(d.glob("*.png")))
        print(f"  {d.relative_to(BASE.parent)}  ({n} PNG)")


if __name__ == "__main__":
    main()
