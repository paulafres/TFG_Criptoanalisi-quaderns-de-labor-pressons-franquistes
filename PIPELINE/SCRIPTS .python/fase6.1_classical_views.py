"""
GENERADOR DE VISUALITZACIONS (CLASSICAL VIEWS)
======================================================
Aquest script té les funcionalitats de visualització morfològica
i l'anàlisi estadístic de la informació (mètriques).

Genera a la carpeta 'res_fase6/':
 1. Arbres Morfològics (individuals)
 2. Dashboards combinats individuals (Hardware + Patrons + Mètriques Z-score)
 3. Globals: Heatmap Hardware, PCA, Heatmap Mètriques, Correlacions, 
    i les Matrius de Bombolles de patrons compartits.
"""

import json
import math
import os
import re
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓ I CONSTANTS
# ─────────────────────────────────────────────
from config import (
    BASE as _BASE, RES_FASE_6,
    F1_TEIXITS_DATASET, F1_PATRONS_COMPARTITS, F1_SIMBOLS_INDIVIDUALS,
)

_PNG_BASE = RES_FASE_6

# Canvi de directori actiu perquè totes les rutes relatives heretades segueixin
# resolent-se respecte a "CUADERNO CARMEN MACHADO .txt"
os.chdir(str(_BASE))

DATASET_FILE    = str(F1_TEIXITS_DATASET)
CSV_COMPARTITS  = str(F1_PATRONS_COMPARTITS)
CSV_SIMBOLS     = str(F1_SIMBOLS_INDIVIDUALS)

MAX_LABEL_CHARS = 22
LEVEL_H         = 260
DPI_IND         = 150   # Resolució dashboards individuals
DPI_GLOBALS     = 200   # Resolució gràfics globals i arbres
BARYCENTER_ITER = 8
NODE_W_MIN      = 70
NODE_W_MAX      = 170

# Estil global
sns.set_theme(style="white", palette="muted")

out_dir = RES_FASE_6 

# Creació de carpetes de sortida (Dins de la carpeta de resultats activa)
os.makedirs(str(out_dir / "Dashboards_Individuals"), exist_ok=True)
os.makedirs(str(out_dir / "Arbres_Morfologics"),     exist_ok=True)
os.makedirs(str(out_dir / "Globals"),                exist_ok=True)

# Noms per a les mètriques
METRIC_LABELS = {
    "m1_char_freq":   "Freqüència lletres",
    "m2_bigram":      "Freqüència bigrames",
    "m3_trigram":     "Freqüència trigrames",
    "m4_compressio":  "Ràtio de compressió (Zlib)",
    "m5_repeticio":   "Redundància local",
    "m6_simetria":    "% Línies simètriques",
    "m7_creixement":  "Creixement de voltes",
    "m8_entropia":    "Entropia de Shannon",
}

# ─────────────────────────────────────────────
# 2. CÀRREGA DE DADES
# ─────────────────────────────────────────────
print("Carregant dades des de JSON i CSVs...")
try:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Índex ràpid per nom per facilitar cerques
    dataset_by_nom = {t["nom"]: t for t in dataset}
    
    print(f"  OK. Carregats {len(dataset)} teixits.")
except FileNotFoundError:
    print(f"Error crític: No es troba '{DATASET_FILE}'. Executa el parser primer.")
    exit()

# ═══════════════════════════════════════════════════════
# BLOC A — FUNCIONS SUPORT (Layout, Text, Grafs)
# ═══════════════════════════════════════════════════════

def truncar(text, n=MAX_LABEL_CHARS):
    """Trunca el text afegint punts suspensius si supera n caràcters."""
    return text if len(text) <= n else text[:n-1] + "..."

def construir_graf(patterns_info):
    """Crea un DiGraph de NetworkX basat en la genealogia de patterns_info."""
    G = nx.DiGraph()
    for p, info in patterns_info.items():
        G.add_node(p, freq_total=info.get("freq_total", 0),
                   freq_ind=info.get("freq_independent", 0))
    
    for p, info in patterns_info.items():
        # Obtenim fills directes (evitant transitius)
        fills = [d["patro"] for d in info.get("te_variants_mes_llargues", [])
                 if d["patro"] in patterns_info]
        for fill in fills:
            directe = True
            for inter in fills:
                if inter == fill: continue
                inter_fills = [d["patro"] for d in patterns_info.get(inter, {})
                               .get("te_variants_mes_llargues", [])
                               if d["patro"] in patterns_info]
                if fill in inter_fills:
                    directe = False; break
            if directe:
                G.add_edge(p, fill)
    return G

def layout_sugiyama(G):
    """Calcula un layout jeràrquic (Sugiyama) per al graf."""
    nodes = list(G.nodes())
    if not nodes: return {}, ([], [], NODE_W_MAX, 6.5)
    
    aillats    = [n for n in nodes if G.degree(n) == 0]
    connectats = [n for n in nodes if G.degree(n) > 0]
    pos = {}
    
    if connectats:
        G_sub = G.subgraph(connectats).copy()
        level = {n: 0 for n in connectats}
        try:
            for nd in nx.topological_sort(G_sub):
                for s in G_sub.successors(nd):
                    if level[s] < level[nd] + 1:
                        level[s] = level[nd] + 1
        except nx.NetworkXUnfeasible:
            # Si hi ha cicles (no hauria), layout simple
            for i, nd in enumerate(connectats): level[nd] = i % 5
            
        nivells = {}
        for nd, lv in level.items(): nivells.setdefault(lv, []).append(nd)
        max_lv = max(nivells.keys())
        max_n  = max(len(v) for v in nivells.values())
        
        # Mida de node i font dinàmica segons densitat
        node_w = max(NODE_W_MIN, min(NODE_W_MAX, int(9000 / max(max_n, 1))))
        font_sz = max(4.5, min(6.5, node_w / 26.0))
        
        x_pos = {}
        for lv in sorted(nivells):
            for i, nd in enumerate(nivells[lv]): x_pos[nd] = float(i)
            
        # Ordenació per baricentre per reduir creuaments
        for _ in range(BARYCENTER_ITER):
            for lv in sorted(nivells):
                if lv == 0: continue
                preds_count = {nd: list(G_sub.predecessors(nd)) for nd in nivells[lv]}
                bc = {nd: sum(x_pos[p] for p in preds_count[nd])/len(preds_count[nd]) 
                      if preds_count[nd] else x_pos[nd] for nd in nivells[lv]}
                ordered = sorted(nivells[lv], key=lambda n: bc[n])
                for i, nd in enumerate(ordered): x_pos[nd] = float(i)
                nivells[lv] = ordered
            # Fase descendent omesa per brevetat, similar a l'ascendent
            
        # Assignació de coordenades finals
        for lv, nds in nivells.items():
            n = len(nds)
            for i, nd in enumerate(nds):
                pos[nd] = ((i - (n - 1) / 2.0) * node_w, -lv * LEVEL_H)
    else:
        node_w = NODE_W_MAX; font_sz = 6.5
        
    # Layout per a nodes aïllats (graella a la dreta)
    if aillats:
        x_base = max((p[0] for p in pos.values()), default=0) + node_w * 2.5
        for i, nd in enumerate(aillats):
            pos[nd] = (x_base + (i % 7) * node_w, -(i // 7) * LEVEL_H * 0.55)
            
    return pos, (aillats, connectats, node_w, font_sz)

# ═══════════════════════════════════════════════════════
# BLOC B — VISUALITZACIONS INDIVIDUALS (Arbre + Dashboard)
# ═══════════════════════════════════════════════════════

def dibuixar_arbre(G, pos, info_parts, dominants, nom, patterns_info):
    """Genera i guarda el gràfic de l'arbre morfològic."""
    aillats, connectats, node_w, font_sz = info_parts
    if not pos: return
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    
    # Mida de figura dinàmica
    fig, ax = plt.subplots(figsize=(max(14, (max(xs) - min(xs) + node_w * 2.5) / 95),
                                  max(7, (max(ys) - min(ys) + LEVEL_H * 2.5) / 95)))
    
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b8b8b8", arrows=True,
                               arrowsize=7, width=0.65, alpha=0.5,
                               min_source_margin=16, min_target_margin=16)
    
    for node, (x, y) in pos.items():
        info = patterns_info.get(node, {})
        is_dom = node in dominants; is_ail = node in aillats
        
        # Color segons jerarquia
        f_col = "#ffdddd" if is_dom else ("#f5f5f5" if is_ail else "#f0f7ff")
        e_col = "#cc0000" if is_dom else ("#cccccc" if is_ail else "#4477cc")
        
        ax.text(x, y, f"{truncar(node)}\nT:{info.get('freq_total',0)}  S:{info.get('freq_independent',0)}",
                fontsize=font_sz, ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor=f_col, edgecolor=e_col, boxstyle="round,pad=0.28", alpha=1, linewidth=0.8))
    
    if aillats and connectats:
        x_sep = max(pos[n][0] for n in connectats) + node_w * 1.5
        ax.axvline(x=x_sep, color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(x_sep + 8, max(ys) + LEVEL_H * 0.25, "Patrons sense jerarquia",
                fontsize=6.5, color="#bbbbbb", ha="left", style="italic")
        
    ax.set_title(f"Jerarquia Morfologica — {nom}", fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")
    ax.set_xlim(min(xs) - node_w * 0.7, max(xs) + node_w * 0.7)
    ax.set_ylim(min(ys) - LEVEL_H * 0.7, max(ys) + LEVEL_H * 0.4)
    
    # Llegenda
    ax.legend(handles=[mpatches.Patch(color="#ffdddd", label="Token dominant"),
                        mpatches.Patch(color="#f0f7ff", label="Token jerarquic"),
                        mpatches.Patch(color="#f5f5f5", label="Sense connexions")],
              loc="upper left", fontsize=7)
    
    plt.savefig(f"{out_dir}/Arbres_Morfologics/Arbre_{nom}.png",
                dpi=DPI_GLOBALS, bbox_inches="tight")
    plt.clf(); plt.close("all")

def dibuixar_dashboard(t, dataset_complet):
    """Genera el dashboard combinat de 3 panells (Hardware, Patrons, Mètriques Z-score)."""
    nom = t["nom"]
    char_counts = t.get("caracters_individuals", {})
    patterns_info = t.get("patterns_info", {})
    dominants = t.get("tokens_dominants", [])
    n_instr = max(t.get("num_instruccions", 1), 1)
    rang = t.get("rang_voltes") or ["-", "-"]
    forma = t.get("forma") or "—"
    punts_inicials = t.get("punts_inicials") or "-"
    met = t.get("metriques", {}) 

    if not met:
        print(f"  Avis: Sense mètriques per a {nom}. Dashboard incomplet.")
        return

    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor("#fafafa")
    
    # Capçalera
    fig.text(0.5, 0.97, nom, ha="center", va="top", fontsize=18, fontweight="bold", color="#222222")
    subtitol = (f"Forma: {forma}  ·  Instruccions: {n_instr}  ·  "
                f"Voltes: {rang[0]}–{rang[1]}  ·  Patrons repetits: {len(patterns_info)}  ·  Punts inicials: {punts_inicials}")
    fig.text(0.5, 0.92, subtitol, ha="center", va="top", fontsize=10, color="#666666")

    # Per evitar solapament: top=0.80 abaixa les gràfiques per donar espai a la capçalera
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.98, top=0.80, bottom=0.13, wspace=0.38)

    # ── Panell 1: Hardware (Freq per instrucció) ──
    ax1 = fig.add_subplot(gs[0])
    if char_counts:
        df_hw = (pd.DataFrame.from_dict(char_counts, orient="index", columns=["cnt"])
                 .assign(fpi=lambda d: d["cnt"] / n_instr)
                 .sort_values("fpi", ascending=False))
        bars = ax1.bar(df_hw.index, df_hw["fpi"], color="#4c72b0", edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, df_hw["fpi"]):
            if v == 0: bar.set_color("white"); bar.set_edgecolor("#dddddd")
        ax1.set_title("Ús d'alfabet Hardware\n(freq / instrucció)", fontsize=11, pad=12)
        ax1.set_xlabel("Caràcter", fontsize=9); ax1.set_ylabel("Freq / inst.", fontsize=9)
        ax1.tick_params(axis="x", labelsize=9); ax1.tick_params(axis="y", labelsize=9)
        ax1.spines[["top","right"]].set_visible(False)

    # ── Panell 2: Patrons (Top 10 + Dominants) ──
    ax2 = fig.add_subplot(gs[1])
    if patterns_info:
        dom_items = [(p,v) for p,v in patterns_info.items() if p in dominants]
        resta_items = [(p,v) for p,v in patterns_info.items() if p not in dominants]
        top_resta = sorted(resta_items, key=lambda x: -x[1].get("freq_independent", 0))[:10]
        # Juntem dominants obligatoris i top restants, i ordenem per freq independent descendent
        top_pats = sorted(dom_items + top_resta, key=lambda x: -x[1].get("freq_independent", 0))
        
        labels = [truncar(p, 26) for p,_ in top_pats]
        vals_fi = [v.get("freq_independent", 0) for _,v in top_pats]
        vals_ft = [v.get("freq_total", 0) for _,v in top_pats]
        
        # Colors condicionals segons si és dominant o no
        colors_ft = ["#ffdddd" if p in dominants else "#aac4e8" for p,_ in top_pats]
        colors_fi = ["#cc0000" if p in dominants else "#2a6496" for p,_ in top_pats]
        
        y = np.arange(len(labels)); h = 0.38
        # Barra de freqüència total (fons)
        ax2.barh(y + h/2, vals_ft, height=h, color=colors_ft, edgecolor="white", linewidth=0.4)
        # Barra de freqüència independent (davant)
        ax2.barh(y - h/2, vals_fi, height=h, color=colors_fi, edgecolor="white", linewidth=0.4)
        
        ax2.set_yticks(y)
        f_size = 7 if len(labels) > 12 else 8
        ax2.set_yticklabels(labels, fontsize=f_size, fontfamily="monospace")
        ax2.invert_yaxis()
        ax2.set_title("Top patrons sintàctics repetits\n(Dominants garantits + 10 restants)", fontsize=11, pad=12)
        ax2.set_xlabel("Nombre d'aparicions", fontsize=9)
        ax2.tick_params(axis="x", labelsize=9)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.legend(handles=[mpatches.Patch(color="#aac4e8", label="Freq. total"),
                            mpatches.Patch(color="#2a6496", label="Freq. independent"),
                            mpatches.Patch(color="#cc0000", label="Token dominant")],
                  fontsize=8, loc="lower right")

    # ── Panell 3: Mètriques Estadístiques (Context Global Z-score) ──
    ax3 = fig.add_subplot(gs[2])
    # Excloem creixement (m7) si sol ser constant (evita std error)
    metric_keys_analysis = [k for k in METRIC_LABELS.keys() if k != "m7_creixement"]
    
    # Obtenim poblacions globals per a cada mètrica
    all_vals_global = {k: [t2.get("metriques", {}).get(k, 0) for t2 in dataset_complet] 
                       for k in metric_keys_analysis}
    
    # EIX X FIX: Calculem quin és el Z-score màxim global de tot el projecte
    max_z_global = 2.5 # Posem un mínim raonable per si les dades varien poc
    for k in metric_keys_analysis:
        pob = all_vals_global[k]
        std = np.std(pob)
        if std > 0:
            mu = np.mean(pob)
            max_local = max(abs((val - mu) / std) for val in pob)
            if max_local > max_z_global:
                max_z_global = max_local
                
    # Arrodonim el límit a l'enter superior (Ex: si el màxim és 2.7, l'eix anirà de -3 a +3)
    limit_x = math.ceil(max_z_global)

    # Recopilem valors bruts d'aquest teixit per calcular el seu Z-score
    val_teixit = [met.get(k, 0) for k in metric_keys_analysis]
    
    # Calculem Z-scores ( (x - media) / std )
    z_scores = []
    for k, v in zip(metric_keys_analysis, val_teixit):
        poblacio = all_vals_global[k]
        mu = np.mean(poblacio)
        std = np.std(poblacio)
        z_scores.append((v - mu) / std if std > 0 else 0.0)

    short_labels = [METRIC_LABELS[k] for k in metric_keys_analysis]
    
    # Colors: Vermell si és alt (>1 std), Blau si és baix (<-1 std), Gris normal
    colors_z = ["#cc0000" if z > 1 else ("#4c72b0" if z < -1 else "#aaaaaa") for z in z_scores]
    
    y3 = np.arange(len(short_labels))
    ax3.barh(y3, z_scores, color=colors_z, edgecolor="white", linewidth=0.4)
    
    # Línies de referència
    ax3.axvline(0, color="#888888", linewidth=0.8) # Mitjana
    ax3.axvline(1,  color="#cc0000", linewidth=0.6, linestyle="--", alpha=0.5) # +1 std
    ax3.axvline(-1, color="#4c72b0", linewidth=0.6, linestyle="--", alpha=0.5) # -1 std
    
    ax3.set_yticks(y3)
    ax3.set_yticklabels(short_labels, fontsize=9)
    ax3.invert_yaxis()
    
    # APLIQUEM EL LÍMIT GLOBAL CALCULAT
    ax3.set_xlim(-limit_x, limit_x)
    
    ax3.set_title("Perfil estadístic de la informació\n(Z-score vs. resta de teixits)", fontsize=11, pad=12)
    ax3.set_xlabel("Z-score (Desviacions Estàndard)", fontsize=9)
    ax3.tick_params(axis="x", labelsize=9)
    ax3.spines[["top","right"]].set_visible(False)
    
    # Llegenda per Z-score
    ax3.legend(handles=[mpatches.Patch(color="#cc0000", label="Mètrica alta (>1 std)"),
                        mpatches.Patch(color="#4c72b0", label="Mètrica baixa (<-1 std)"),
                        mpatches.Patch(color="#aaaaaa", label="Dins la normalitat")],
              fontsize=8, loc="lower right")

    plt.savefig(f"{out_dir}/Dashboards_Individuals/Dashboard_{nom}.png",
                dpi=DPI_IND, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.clf(); plt.close("all")


# ═══════════════════════════════════════════════════════
# BLOC C — VISUALITZACIONS GLOBALS (Comparatives)
# ═══════════════════════════════════════════════════════

def generar_globals(dataset, simbols_csv, compartits_csv):
    """Genera tots els gràfics comparatius globals."""
    print("\nGenerant gràfics globals...")
    
    # Colormap personalitzat: blanc (0) -> blau fosc (alt)
    NDap_wb = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#c6dbef", "#4292c6", "#08306b"])

    # ── 1. Heatmap Hardware Normalitzat ──
    try:
        print("  Generant Heatmap Hardware...")
        df_sim = pd.read_csv(simbols_csv).set_index("teixit").drop(columns=["forma"], errors="ignore")
        
        # Mapa de núm. instruccions per normalitzar
        n_map = {t["nom"]: max(t.get("num_instruccions", 1), 1) for t in dataset}
        instr_s = pd.Series(n_map).reindex(df_sim.index).fillna(1)
        
        # Normalització (freq / instrucció)
        df_norm = df_sim.div(instr_s, axis=0).round(3)
        mask0 = df_norm == 0 # Mascara per pintar els zeros de blanc pur
        
        fig, ax = plt.subplots(figsize=(18, 7))
        # Pintem valors > 0
        sns.heatmap(df_norm, annot=True, fmt=".2f", cmap=NDap_wb, mask=mask0,
                    linewidths=0.4, linecolor="#eeeeee", ax=ax, cbar_kws={"label": "freq / instrucció", "shrink": 0.7})
        # Pintem valors = 0 (text gris fluix)
        sns.heatmap(df_norm, annot=True, fmt=".2f", cmap=["#ffffff"], mask=~mask0,
                    linewidths=0.4, linecolor="#eeeeee", ax=ax, cbar=False, annot_kws={"color": "#cccccc"})
        
        ax.set_title("Ús relatiu de caràcters Hardware per teixit\n(Normalitzat per núm. instruccions)", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Globals/Heatmap_simbols.png", dpi=DPI_GLOBALS, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Error Heatmap Hardware: {e}")

    # ── 2. PCA + Biplot de Teixits ──
    try:
        print("  Generant PCA de teixits...")
        # Preparem matriu de dades (Teixits x Mètriques)
        metric_keys_analysis = [k for k in METRIC_LABELS.keys() if k != "m7_creixement"]
        X_raw = np.array([[t["metriques"].get(k, 0) for k in metric_keys_analysis] for t in dataset])
        noms_pca = [t["nom"] for t in dataset]
        
        # Estandardització (Z-score) obligatòria per PCA
        X_std = StandardScaler().fit_transform(X_raw)
        
        # Execució PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_std)
        var_exp = pca.explained_variance_ratio_
        
        # Coeficients (loadings) per al biplot
        loadings = pd.DataFrame(pca.components_.T, index=metric_keys_analysis, columns=["PC1", "PC2"])

        # ── Colors dinàmics per forma ──
        # 1. Busquem totes les formes úniques al dataset (ignorant les buides)
        formes_uniques = sorted(list(set(t.get("forma", "") for t in dataset if t.get("forma", ""))))
        
        # 2. Generem una paleta de colors automàtica amb Seaborn
        paleta_colors = sns.color_palette("Set2", len(formes_uniques)).as_hex()
        
        # 3. Creem el diccionari ajuntant cada forma amb el seu color
        colors_forma = dict(zip(formes_uniques, paleta_colors))
        
        # 4. Afegim el gris per als teixits que no tenen forma
        colors_forma[""] = "#888888"
        colors_forma[None] = "#888888" # Per seguretat, si ve un 'null' del JSON
        
        fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle("Anàlisi de Components Principals (PCA) de Teixits\nBasat en mètriques estadístiques de la informació", fontsize=15, fontweight="bold", y=1.02)

        # A) Scatter Plot de Teixits
        for i, nom in enumerate(noms_pca):
            forma = dataset[i].get("forma", "")
            ax_p.scatter(X_pca[i, 0], X_pca[i, 1], color=colors_forma.get(forma, "#888888"), s=120, zorder=3, edgecolors="white", linewidth=1.2)
            ax_p.annotate(nom, (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#333333", alpha=0.8)
            
        ax_p.axhline(0, color="#cccccc", lw=1, zorder=1); ax_p.axvline(0, color="#cccccc", lw=1, zorder=1)
        ax_p.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variància explicada)", fontsize=11)
        ax_p.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variància explicada)", fontsize=11)
        ax_p.set_title("Projecció dels Teixits", fontsize=12, fontweight="bold")
        ax_p.grid(True, linestyle=":", alpha=0.6)
        
        # Llegenda formes
        leg_hand = [mpatches.Patch(color=v, label=k if k else "No definida") for k, v in colors_forma.items()]
        ax_p.legend(handles=leg_hand, title="Forma del teixit", fontsize=9, title_fontsize=10, loc="best")

        # B) Biplot de Loadings (Mètriques)
        for met_key in metric_keys_analysis:
            c1, c2 = loadings.loc[met_key, "PC1"], loadings.loc[met_key, "PC2"]
            # Vector arrow
            ax_b.arrow(0, 0, c1, c2, color="#cc0000", alpha=0.7, width=0.01, head_width=0.03)
            # Text label, una mica desplaçat de la punta
            ax_b.text(c1 * 1.15, c2 * 1.15, METRIC_LABELS[met_key], color="#333333", ha="center", va="center", fontsize=9, fontweight="bold")
            
        # Cercle unitari de correlació (opcional, ajuda a interpretar)
        circle = plt.Circle((0,0), 1, color='#dddddd', fill=False, linestyle='--')
        ax_b.add_artist(circle)
        
        ax_b.axhline(0, color="#cccccc", lw=1); ax_b.axvline(0, color="#cccccc", lw=1)
        ax_b.set_xlim(-1.2, 1.2); ax_b.set_ylim(-1.2, 1.2)
        ax_b.set_xlabel("PC1 (Loadings)", fontsize=11); ax_b.set_ylabel("PC2 (Loadings)", fontsize=11)
        ax_b.set_title("Influència de les Mètriques (Loadings)", fontsize=12, fontweight="bold")
        ax_b.grid(True, linestyle=":", alpha=0.6)
        ax_b.set_aspect('equal') # Important per interpretar angles

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{out_dir}/Globals/PCA_teixits.png", dpi=DPI_GLOBALS, bbox_inches="tight")
        plt.close()
    except Exception as e:
        import traceback; print(f"  Error PCA: {e}"); traceback.print_exc()

    # ── 3. Heatmap Mètriques Z-score Global ──
    try:
        print("  Generant Heatmap Mètriques Global...")
        metric_keys_analysis = [k for k in METRIC_LABELS.keys() if k != "m7_creixement"]
        
        # Creem DataFrame de mètriques brutes
        dades_met_brutes = []
        for t in dataset:
            fila = {"teixit": t["nom"]}
            for k in metric_keys_analysis:
                fila[METRIC_LABELS[k]] = t["metriques"].get(k, 0)
            dades_met_brutes.append(fila)
            
        df_met_global = pd.DataFrame(dades_met_brutes).set_index("teixit")
        
        # Estandardització global (per columnes)
        scaler_g = StandardScaler()
        df_met_zglobal = pd.DataFrame(scaler_g.fit_transform(df_met_global), 
                                      index=df_met_global.index, 
                                      columns=df_met_global.columns)
        
        fig, ax = plt.subplots(figsize=(16, 7))
        # Use RdBu_r palette (Vermell alt, Blau baix, Blanc mitjana)
        sns.heatmap(df_met_zglobal, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
                    linewidths=0.5, linecolor="#eeeeee", ax=ax, cbar_kws={"label": "Z-score Global"})
        
        ax.set_title("Mapa comparatiu de perfils de informació (Z-score Global)\nVermell = Valor molt superior a la mitjana | Blau = Molt inferior", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Globals/Heatmap_metriques_global.png", dpi=DPI_GLOBALS, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Error Heatmap Mètriques Global: {e}")

    # ── 4. Matriu de Correlació entre Mètriques ──
    try:
        print("  Generant Matriu de Correlació...")
        # Reutilitzem df_met_global del pas anterior
        corr_matrix = df_met_global.corr()
        
        # Mascara per mostrar només la meitat inferior
        mask_corr = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor="#eeeeee", ax=ax)
        
        ax.set_title("Matriu de Correlació de Pearson entre Mètriques", fontsize=14, fontweight="bold", pad=15)
        ax.tick_params(axis="x", labelsize=10, rotation=35)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", labelsize=10, rotation=0)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Globals/Correlacio_metriques.png", dpi=DPI_GLOBALS, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Error Correlació: {e}")

    # ── 5. Matrius de Bombolles de Patrons Compartits (Absoluta i Relativa) ──
    try:
        print("  Generant Matrius de Bombolles de patrons compartits...")
        df_comp = pd.read_csv(compartits_csv)
        
        if not df_comp.empty and len(df_comp.columns) > 2:
            # Neteja de columnes no patrons
            df_comp_clean = df_comp.drop(columns=["forma"], errors="ignore")
                
            # "Desenrotllem" (Melt) la taula per tenir format llarg: teixit | Patró | Freqüència
            df_long = df_comp_clean.melt(id_vars=["teixit"], var_name="Patró", value_name="Freqüència")
            
            # Assegurem numèric i filtrem zeros
            df_long["Freqüència"] = pd.to_numeric(df_long["Freqüència"], errors="coerce").fillna(0)
            df_bomb = df_long[df_long["Freqüència"] > 0].copy()
            
            if not df_bomb.empty:
                # Creuem dades amb JSON per obtenir jerarquia i instruccions per normalitzar
                global dataset_by_nom # Use global index
                
                def info_extra(row):
                    t_nom = row["teixit"]; p_nom = row["Patró"]
                    t_json = dataset_by_nom.get(t_nom, {})
                    
                    doms = t_json.get("tokens_dominants", [])
                    tipus = "Dominant (Arrel)" if p_nom in doms else "Subseqüència"
                    instr = max(t_json.get("num_instruccions", 1), 1)
                    return pd.Series([tipus, instr])
                    
                df_bomb[["Tipus", "Instruccions"]] = df_bomb.apply(info_extra, axis=1)
                
                # Càlcul Freqüència Relativa (per instrucció)
                df_bomb["Freq_Relativa"] = df_bomb["Freqüència"] / df_bomb["Instruccions"]
                
                # Format visual text patró (truncat i net)
                df_bomb["Patró_Visual"] = df_bomb["Patró"].apply(lambda p: truncar(p, 40))
                
                # Ordenació eix Y (patrons) per freqüència total descendent
                ordre_pats = df_bomb.groupby("Patró_Visual")["Freqüència"].sum().sort_values(ascending=True).index
                alcada_din = max(10, len(ordre_pats) * 0.4)
                
                # Palette jerarquia
                pal_jer = {"Dominant (Arrel)": "#cc0000", "Subseqüència": "#4c72b0"}

                # A) MATRIU ABSOLUTA
                print("    Dibuixant Matriu Absoluta...")
                fig_abs, ax_abs = plt.subplots(figsize=(18, alcada_din))
                fig_abs.patch.set_facecolor("#fafafa")
                
                sns.scatterplot(data=df_bomb, x="teixit", y="Patró_Visual", size="Freqüència", hue="Tipus",
                                sizes=(150, 800), palette=pal_jer, alpha=0.5, ax=ax_abs, edgecolor="white", linewidth=1.5)
                
                # Text del número exacte dins la bombolla
                for _, row in df_bomb.iterrows():
                    ax_abs.text(row["teixit"], row["Patró_Visual"], str(int(row["Freqüència"])), 
                                horizontalalignment='center', va='center', size=8, color='black', weight='bold')
                
                ax_abs.grid(True, linestyle=":", linewidth=1, color="#dddddd", alpha=0.8)
                ax_abs.set_title("Matriu de Patrons Sintàctics Compartits - Freqüència Absoluta\n(Extret de taula_patrons_compartits.csv)", fontsize=16, fontweight="bold", pad=20)
                ax_abs.set_ylabel("Patrons Sintàctics Compartits", fontsize=12, fontweight="bold"); ax_abs.set_xlabel("")
                ax_abs.tick_params(axis="x", rotation=45, labelsize=10)
                ax_abs.tick_params(axis="y", labelsize=9, labelfontfamily="monospace")
                
                # Llegenda jerarquia neta
                hand, lab = ax_abs.get_legend_handles_labels()
                leg_idx = [i for i, l in enumerate(lab) if l in pal_jer.keys()]
                if leg_idx: ax_abs.legend([hand[i] for i in leg_idx], [lab[i] for i in leg_idx], bbox_to_anchor=(1.01, 1), loc='upper left', title="Jerarquia")
                
                plt.subplots_adjust(left=0.25, right=0.85, bottom=0.15)
                plt.savefig(f"{out_dir}/Globals/Matriu_Bombolles_Patrons_Absoluta.png", dpi=DPI_GLOBALS, bbox_inches="tight")
                plt.close()

                # B) MATRIU RELATIVA (Normalitzada per instruccions)
                print("    Dibuixant Matriu Relativa...")
                fig_rel, ax_rel = plt.subplots(figsize=(18, alcada_din))
                fig_rel.patch.set_facecolor("#fafafa")
                
                sns.scatterplot(data=df_bomb, x="teixit", y="Patró_Visual", size="Freq_Relativa", hue="Tipus",
                                sizes=(150, 800), palette=pal_jer, alpha=0.5, ax=ax_rel, edgecolor="white", linewidth=1.5)
                
                # Text del número relatiu (arrodonit) dins la bombolla
                for _, row in df_bomb.iterrows():
                    # Format: 2 decimals, traient zeros inútils (ex: 1.2, 0.5)
                    txt_rel = f"{round(row['Freq_Relativa'], 2):g}"
                    ax_rel.text(row["teixit"], row["Patró_Visual"], txt_rel, 
                                horizontalalignment='center', va='center', size=7, color='black', weight='bold')
                
                ax_rel.grid(True, linestyle=":", linewidth=1, color="#dddddd", alpha=0.8)
                ax_rel.set_title("Matriu de Patrons Sintàctics Compartits - Normalitzada\n(Freqüència d'aparició per instrucció)", fontsize=16, fontweight="bold", pad=20)
                ax_rel.set_ylabel("Patrons Sintàctics Compartits", fontsize=12, fontweight="bold"); ax_rel.set_xlabel("")
                ax_rel.tick_params(axis="x", rotation=45, labelsize=10)
                ax_rel.tick_params(axis="y", labelsize=9, labelfontfamily="monospace")
                
                # Llegenda rel
                hand, lab = ax_rel.get_legend_handles_labels()
                leg_idx = [i for i, l in enumerate(lab) if l in pal_jer.keys()]
                if leg_idx: ax_rel.legend([hand[i] for i in leg_idx], [lab[i] for i in leg_idx], bbox_to_anchor=(1.01, 1), loc='upper left', title="Jerarquia")

                plt.subplots_adjust(left=0.25, right=0.85, bottom=0.15)
                plt.savefig(f"{out_dir}/Globals/Matriu_Bombolles_Patrons_Relativa.png", dpi=DPI_GLOBALS, bbox_inches="tight")
                plt.close()
                print("  OK Matrius de Bombolles.")
            else:
                print("  Avis: El CSV de compartits no té valors > 0.")
        else:
            print("  Avis: El CSV de compartits està buit o té format incorrecte.")
    except Exception as e:
        import traceback; print(f"  Error Matrius Bombolles: {e}"); traceback.print_exc()

# ═══════════════════════════════════════════════════════
# 3. EXECUTOR PRINCIPAL (Pipeline)
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("INICIANT GENERACIÓ INTEGRAL DE GRÀFICS")
    print("="*60)
    
    # --- PIPELINE PER TEIXIT (Individuals) ---
    print("\n[1/2] Processant teixits individuals...")
    global_ref_dataset = dataset # Referència per calcular Z-scores contextuals
    
    for t in dataset:
        nom = t["nom"]
        p_info = t.get("patterns_info", {})
        doms = t.get("tokens_dominants", [])
        print(f"  Processing: {nom}...")
        
        # 1. Dashboard combinat (Hardware + Patrons + Mètriques)
        dibuixar_dashboard(t, global_ref_dataset)
        
        # 2. Arbre Morfològic
        if p_info:
            try:
                G_morf = construir_graf(p_info)
                pos_morf, info_morf = layout_sugiyama(G_morf)
                if pos_morf:
                    dibuixar_arbre(G_morf, pos_morf, info_morf, doms, nom, p_info)
            except Exception as e:
                print(f"    Error generant arbre per {nom}: {e}")
        plt.close("all") # Neteja memòria
        
    # --- PIPELINE GLOBAL ---
    print("\n[2/2] Processant gràfics comparatius globals...")
    generar_globals(dataset, CSV_SIMBOLS, CSV_COMPARTITS)
    
    print("\n" + "="*60)
    print("PROCÉS FINALITZAT REEIXIDAMENT.")
    print("Revisa la carpeta 'res_fase6/' per veure els gràfics.")
    print("="*60)