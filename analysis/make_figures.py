#!/usr/bin/env python3
"""Draw every figure of the paper from the released result records.

Usage:
    python analysis/make_figures.py --paper results/paper/canonical_results.json \
        --routing results/paper/routing_summary.json \
        --followup results/followup/summary.json --out paper/figures

One visual language is used throughout: ProofKV is blue, interventions that damage a chain are orange,
oracles are green, and every reference method is a neutral gray. Text is always ink-colored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # ProofKV
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"    # chain-damaging interventions
ORANGE_LIGHT = "#fbe1d6"
GREEN = "#1baf7a"     # oracle / gold-informed references
GREEN_LIGHT = "#d3f0e5"
YELLOW = "#eda100"
GRAY = "#898781"      # reference methods
GRAY_DARK = "#52514e"
GRAY_LIGHT = "#efeeea"
WHITE = "#ffffff"


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 7.8,
        "axes.titlesize": 8.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.4,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,
        "axes.linewidth": 0.7,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.6,
        "lines.markeredgecolor": WHITE,
        "lines.markeredgewidth": 0.8,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
    })


OUT = Path("figures")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def one(rows, **criteria):
    hits = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(hits) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(hits)}")
    return hits[0]


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=5)


# ----------------------------------------------------------------------------- Figure 1
def rbox(ax, x, y, w, h, fc, ec, lw=0.8, r=0.012, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, a, b, color=INK2, lw=0.9, style="-|>", rad=0.0, z=4, ms=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def overview() -> None:
    fig = plt.figure(figsize=(6.3, 2.52))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- stage frames
    frames = [
        (0.006, 0.262, "1  Index once", "per document, question-agnostic"),
        (0.283, 0.352, "2  Compose and select", "per question, before prefill"),
        (0.650, 0.344, "3  Materialize", "compact prefill or paged KV"),
    ]
    for x, w, title, sub in frames:
        rbox(ax, x, 0.20, w, 0.775, WHITE, AXIS, lw=0.8, r=0.014, z=1)
        ax.text(x + 0.012, 0.935, title, fontsize=8.65, fontweight="bold", va="center")
        ax.text(x + 0.012, 0.885, sub, fontsize=7.0, color=INK2, va="center")

    # ---- stage 1: document spans + candidate links
    sx, sw, sh = 0.020, 0.128, 0.040
    ys = [0.800, 0.742, 0.684, 0.626, 0.568, 0.510, 0.452, 0.394, 0.336]
    labels = {1: "A", 4: "B", 7: "C"}
    texts = {1: "… played X", 4: "… starred in F", 7: "D directed F"}
    for i, y in enumerate(ys):
        hot = i in labels
        rbox(ax, sx, y, sw, sh, BLUE_LIGHT if hot else GRAY_LIGHT, BLUE if hot else GRAY_LIGHT, lw=0.9 if hot else 0, r=0.008)
        if hot:
            ax.text(sx + 0.007, y + sh / 2, labels[i], fontsize=7.85, fontweight="bold", va="center")
            ax.text(sx + 0.026, y + sh / 2, texts[i], fontsize=7.0, va="center", color=INK)
        else:
            ax.plot([sx + 0.010, sx + sw - 0.018 - 0.02 * (i % 3)], [y + sh / 2] * 2, color=AXIS, lw=1.6, solid_capstyle="round", zorder=3)
    link_x = sx + sw + 0.003
    for a_, b_, rad in [(0, 3, -0.42), (2, 6, -0.36), (5, 8, -0.42), (3, 8, -0.30)]:
        arrow(ax, (link_x, ys[a_] + sh / 2), (link_x, ys[b_] + sh / 2), color=AXIS, lw=0.7, style="-", rad=rad, z=2)
    arrow(ax, (link_x, ys[1] + sh / 2), (link_x, ys[4] + sh / 2), color=BLUE, lw=1.2, style="-", rad=-0.42, z=3)
    arrow(ax, (link_x, ys[4] + sh / 2), (link_x, ys[7] + sh / 2), color=BLUE, lw=1.2, style="-", rad=-0.42, z=3)
    for k, line in enumerate(["spans", "KV cost", "lexical cues", "links"]):
        ax.text(0.179, 0.690 - 0.064 * k, line, fontsize=7.0, color=INK2, ha="left", va="center")
    ax.text(0.137, 0.290, "Evidence Index", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(0.137, 0.243, "reused by every question", fontsize=7.0, color=INK2, ha="center", va="center")

    # ---- stage 2: question, scorer, chain, selection, route
    qx, qw = 0.295, 0.328
    rbox(ax, qx, 0.748, qw, 0.102, GRAY_LIGHT, GRAY_LIGHT, lw=0, r=0.010)
    ax.text(qx + qw / 2, 0.822, "“Who directed the film starring", fontsize=7.0, style="italic", ha="center", va="center")
    ax.text(qx + qw / 2, 0.777, "the actor who played X?”", fontsize=7.0, style="italic", ha="center", va="center")
    ny = 0.600
    nodes = [(qx + 0.048, "A"), (qx + qw / 2, "B"), (qx + qw - 0.048, "C")]
    for x, lab in nodes:
        rbox(ax, x - 0.030, ny - 0.040, 0.060, 0.080, BLUE, BLUE, lw=0, r=0.012, z=3)
        ax.text(x, ny, lab, fontsize=9.45, fontweight="bold", color=WHITE, ha="center", va="center", zorder=5)
    arrow(ax, (nodes[0][0] + 0.034, ny), (nodes[1][0] - 0.034, ny), color=BLUE, lw=1.3, ms=8)
    arrow(ax, (nodes[1][0] + 0.034, ny), (nodes[2][0] - 0.034, ny), color=BLUE, lw=1.3, ms=8)
    ax.text(nodes[0][0], ny + 0.072, "start  r", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text((nodes[0][0] + nodes[1][0]) / 2, ny + 0.035, "link  e", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text((nodes[1][0] + nodes[2][0]) / 2, ny + 0.035, "link  e", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(nodes[2][0], ny + 0.072, "end  t", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(qx + qw / 2, 0.506, "beam search over  P = r · t · Π γe", fontsize=7.0, color=INK2, ha="center", va="center")

    bx, bw, by, bh = qx, 0.154, 0.228, 0.240
    rbox(ax, bx, by, bw, bh, WHITE, AXIS, lw=0.8, r=0.010)
    ax.text(bx + bw / 2, by + bh - 0.040, "Select in budget", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(bx + bw / 2, by + bh - 0.097, "whole chains (AND)", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(bx + bw / 2, by + bh - 0.143, "any full chain (OR)", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(bx + bw / 2, by + 0.038, "A, B, C  or none", fontsize=7.0, ha="center", va="center")
    rx = qx + qw - bw
    rbox(ax, rx, by, bw, bh, WHITE, AXIS, lw=0.8, r=0.010)
    ax.text(rx + bw / 2, by + bh - 0.040, "Margin route", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(rx + bw / 2, by + bh - 0.097, "m = P(1st) − P(2nd)", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(rx + bw / 2, by + bh - 0.143, "m > 0.17:  chains", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(rx + bw / 2, by + 0.038, "else:  entity graph", fontsize=7.0, color=INK2, ha="center", va="center")
    arrow(ax, (bx + bw + 0.003, by + bh / 2), (rx - 0.003, by + bh / 2), color=INK2, lw=0.9)

    # ---- stage 3: full prompt vs compact KV
    px, pw = 0.666, 0.312
    ax.text(px, 0.815, "prompt positions", fontsize=7.0, color=INK2, va="center")
    n = 48
    cell = pw / n
    kept = {0: "sink", 1: "sink", 2: "sink", 3: "sink"}
    for i in range(9, 14):
        kept[i] = "A"
    for i in range(22, 27):
        kept[i] = "B"
    for i in range(34, 39):
        kept[i] = "C"
    for i in range(44, 48):
        kept[i] = "tail"
    for i in range(n):
        role = kept.get(i)
        color = BLUE if role in ("A", "B", "C") else (GRAY_DARK if role else GRAY_LIGHT)
        ax.add_patch(Rectangle((px + i * cell + 0.0006, 0.728), cell - 0.0012, 0.052, fc=color, ec="none", zorder=3))
    for lab, mid in (("A", 11.5), ("B", 24.5), ("C", 36.5)):
        ax.add_patch(Rectangle((px + (mid - 2.5) * cell + 0.0006, 0.728), 5 * cell - 0.0012, 0.052, fc=BLUE, ec="none", zorder=4))
        ax.text(px + mid * cell, 0.754, lab, fontsize=7.25, fontweight="bold", color=WHITE, ha="center", va="center", zorder=5)
    order = [i for i in range(n) if i in kept]
    cw = 0.0108
    cx0 = px + (pw - cw * len(order)) / 2
    cy = 0.560
    for j, i in enumerate(order):
        role = kept[i]
        if role in ("sink", "tail"):
            ax.add_patch(Rectangle((cx0 + j * cw + 0.0006, cy), cw - 0.0012, 0.062, fc=GRAY_DARK, ec="none", zorder=3))
    for lab, start in (("A", 4), ("B", 9), ("C", 14)):
        ax.add_patch(Rectangle((cx0 + start * cw + 0.0008, cy), 5 * cw - 0.0016, 0.062, fc=BLUE, ec="none", zorder=3))
        ax.text(cx0 + (start + 2.5) * cw, cy + 0.031, lab, fontsize=7.25, fontweight="bold", color=WHITE, ha="center", va="center", zorder=5)
    ax.text(cx0 + 2 * cw, cy - 0.034, "sinks", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(cx0 + 20.4 * cw, cy - 0.034, "recent tail", fontsize=7.0, color=INK2, ha="center", va="center")
    for mid_full, start in ((11.5, 4), (24.5, 9), (36.5, 14)):
        arrow(ax, (px + mid_full * cell, 0.724), (cx0 + (start + 2.5) * cw, cy + 0.066), color=BLUE, lw=0.8, ms=6)
    ax.text(px + pw / 2, 0.468, "KV only for kept positions,", fontsize=7.0, color=INK2, ha="center", va="center")
    ax.text(px + pw / 2, 0.424, "at their original position IDs", fontsize=7.0, color=INK2, ha="center", va="center")
    rbox(ax, px, 0.228, pw, 0.150, GRAY_LIGHT, GRAY_LIGHT, lw=0, r=0.010)
    ax.text(px + pw / 2, 0.332, "4.2% of an 8k prompt is prefilled", fontsize=7.3, fontweight="bold", ha="center", va="center")
    ax.text(px + pw / 2, 0.276, "71.9 F1 vs. 62.8 full prompt (HotpotQA)", fontsize=7.0, color=INK2, ha="center", va="center")

    # ---- arrows between stages
    arrow(ax, (0.269, 0.590), (0.282, 0.590), color=INK2, lw=1.0, ms=8)
    arrow(ax, (0.636, 0.590), (0.649, 0.590), color=INK2, lw=1.0, ms=8)

    # ---- bottom strip: why completeness matters
    rbox(ax, 0.006, 0.020, 0.988, 0.150, WHITE, AXIS, lw=0.8, r=0.014, z=1)
    ax.text(0.020, 0.095, "Why chains", fontsize=7.85, fontweight="bold", va="center")

    def mini(x, labs, colors, note):
        for k, (lab, col) in enumerate(zip(labs, colors)):
            solid = col != "none"
            rbox(ax, x + k * 0.050, 0.062, 0.038, 0.066, col if solid else WHITE, col if solid else ORANGE, lw=0 if solid else 1.0, r=0.010, z=3)
            ax.text(x + k * 0.050 + 0.019, 0.095, lab, fontsize=7.65, fontweight="bold", color=WHITE if solid else ORANGE, ha="center", va="center", zorder=5)
            if k < len(labs) - 1:
                ok = solid and colors[k + 1] != "none"
                ax.plot([x + k * 0.050 + 0.039, x + k * 0.050 + 0.049], [0.095, 0.095], color=BLUE if ok else AXIS, lw=1.2, zorder=3,
                        linestyle="-" if ok else (0, (1, 1.2)))
        ax.text(x + 3 * 0.050 - 0.002, 0.095, note, fontsize=7.0, color=INK2, va="center")

    mini(0.132, ["A", "B", "C"], [GRAY_DARK, "none", GRAY_DARK], "independent ranking drops B, so D is lost")
    mini(0.606, ["A", "B", "C"], [BLUE, BLUE, BLUE], "ProofKV keeps the chain whole")
    save(fig, "overview")


# ----------------------------------------------------------------------------- Figure 2
def evidence(paper: dict, follow: dict) -> None:
    fig = plt.figure(figsize=(6.3, 2.21))
    bottom, height = 0.165, 0.715

    # ---- (a) premise test: two aligned bar panels sharing the method axis
    prem = follow["premise_test"]
    comp = prem["compressors"]
    rows = [
        ("SnapKV", comp["snapkv"]["gold_token_recall"], comp["snapkv"]["f1"], GRAY),
        ("PyramidKV", comp["pyramidkv"]["gold_token_recall"], comp["pyramidkv"]["f1"], GRAY),
        ("H$_2$O-style", comp["observed_attention_h2o_style"]["gold_token_recall"], comp["observed_attention_h2o_style"]["f1"], GRAY),
        ("StreamingLLM", comp["streamingllm"]["gold_token_recall"], comp["streamingllm"]["f1"], GRAY),
        ("ProofKV chain", prem["chain"]["proof_core"]["gold_token_recall"], prem["chain"]["proof_core"]["f1"], BLUE),
    ]
    y = list(range(len(rows)))[::-1]
    ax_tok = fig.add_axes([0.108, bottom, 0.100, height])
    ax_f1 = fig.add_axes([0.224, bottom, 0.080, height])
    for ax, col, xmax, title in ((ax_tok, 1, 135, "gold tokens\nkept (%)"), (ax_f1, 2, 52, "answer\nF1")):
        for ypos, row in zip(y, rows):
            ax.barh(ypos, row[col], height=0.56, color=row[3], zorder=3)
            ax.text(row[col] + xmax * 0.035, ypos, f"{row[col]:.1f}", va="center", ha="left", fontsize=7.2, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(title)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    ax_tok.set_yticks(y, [r[0] for r in rows])
    ax_tok.set_xticks([0, 50, 100])
    ax_f1.set_yticks([])
    ax_f1.set_xticks([0, 20, 40])
    ax_f1.spines["left"].set_visible(False)
    fig.text(0.006, 0.955, "(a) Tokens kept, chain lost", fontsize=8.4, fontweight="bold", ha="left", va="center")

    # ---- (b) held-out selector test
    ax = fig.add_axes([0.458, bottom, 0.212, height])
    pcr = follow["heldout_selector"]["pcr"]
    methods = [
        ("similarity_graph", "Similarity graph", GRAY),
        ("hybrid_reranker", "Hybrid reranker", GRAY),
        ("bm25", "BM25", GRAY),
        ("same_scorer_independent", "Same scorer, indep.", GRAY_DARK),
        ("single_chain", "Single chain", BLUE),
        ("chains", "ProofKV chains", BLUE),
        ("oracle", "Oracle", GREEN),
    ]
    yb = list(range(len(methods)))[::-1]
    for ypos, (key, label, color) in zip(yb, methods):
        lo, hi = pcr[key]["256"], pcr[key]["512"]
        ax.plot([lo, hi], [ypos, ypos], color=GRID, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter(lo, ypos, s=24, facecolor=WHITE, edgecolor=color, linewidth=1.3, zorder=4)
        ax.scatter(hi, ypos, s=32, facecolor=color, edgecolor=WHITE, linewidth=0.8, zorder=5)
        ax.text(103, ypos, f"{hi:.1f}", va="center", ha="right", fontsize=7.2, fontweight="bold" if key == "chains" else "normal")
    ax.set_yticks(yb, [m[1] for m in methods])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("questions with a complete\ngold chain (%)")
    ax.grid(axis="x")
    ax.scatter([], [], s=24, facecolor=WHITE, edgecolor=INK2, linewidth=1.3, label="256 tokens")
    ax.scatter([], [], s=32, facecolor=INK2, edgecolor=WHITE, linewidth=0.8, label="512 tokens")
    ax.legend(loc="center right", bbox_to_anchor=(0.80, 0.735), handletextpad=0.1, borderpad=0.1, labelspacing=0.35)
    fig.text(0.325, 0.955, "(b) Held-out MuSiQue test", fontsize=8.4, fontweight="bold", ha="left", va="center")

    # ---- (c) link deletion forest plot
    ax = fig.add_axes([0.812, bottom, 0.182, height])
    li = follow["link_interventions"]
    entries = [
        ("MuSiQue (128)", li["musique_slice"]["loss"]["broken_middle"]),
        ("MuSiQue (372)", li["musique_pooled"]["loss"]["broken_middle"]),
        ("HotpotQA para.", li["hotpot_paragraph"]["loss"]["broken_middle"]),
        ("HotpotQA sent.", li["hotpot_sentence"]["loss"]["broken_middle"]),
        ("2WikiMultiHop", li["twowiki"]["loss"]["broken_middle"]),
    ]
    yc = list(range(len(entries)))[::-1]
    for ypos, (label, s) in zip(yc, entries):
        ax.plot([s["lo"], s["hi"]], [ypos, ypos], color=ORANGE, lw=1.5, solid_capstyle="round", zorder=3)
        ax.scatter(s["delta"], ypos, s=34, color=ORANGE, edgecolor=WHITE, linewidth=0.8, zorder=4)
        ax.text(33.5, ypos, f"{s['delta']:.1f}", va="center", ha="right", fontsize=7.2)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_yticks(yc, [e[0] for e in entries])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.5, 34)
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("F1 lost when one\nlink is removed")
    ax.grid(axis="x")
    fig.text(0.692, 0.955, "(c) One link removed, equal bytes", fontsize=8.4, fontweight="bold", ha="left", va="center")
    save(fig, "evidence")


# ----------------------------------------------------------------------------- appendix figures
def budget_regime(follow: dict) -> None:
    br = follow["budget_regime"]["budgets"]
    budgets = [64, 96, 128, 192, 256, 384, 512]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.12), gridspec_kw={"width_ratios": [1.0, 1.0]})
    specs = [("chains", "ProofKV chains", BLUE, "o"), ("iterative_hop_order", "Iterative retrieval", ORANGE, "s"),
             ("same_scorer_independent", "Same scorer, indep.", GRAY, "^")]
    for key, label, color, marker in specs:
        ax1.plot(range(len(budgets)), [br[str(b)]["f1"][key] for b in budgets], color=color, marker=marker, label=label)
    ax1.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    ax1.set_xlabel("retained tokens")
    ax1.set_ylabel("answer F1")
    ax1.set_ylim(0, 50)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left")
    panel_title(ax1, "(a) MuSiQue generation, equal token-level fill")
    gains = [br[str(b)]["gain"]["iterative_hop_order"] for b in budgets]
    xs = list(range(len(budgets)))
    ax2.axhline(0, color=INK2, lw=0.8)
    for x, g in zip(xs, gains):
        ax2.plot([x, x], [g["lo"], g["hi"]], color=BLUE, lw=1.5, solid_capstyle="round", zorder=3)
    ax2.scatter(xs, [g["delta"] for g in gains], s=34, color=BLUE, edgecolor=WHITE, linewidth=0.8, zorder=4)
    ax2.set_xticks(xs, [f"{b}\n{br[str(b)]['gold_fits_pct']:.0f}%" for b in budgets])
    ax2.set_xlabel("retained tokens / questions whose gold chain fits")
    ax2.set_ylabel("chains − iterative retrieval (F1)")
    ax2.set_ylim(-9, 23)
    ax2.grid(axis="y")
    panel_title(ax2, "(b) Gain of composing the chain, 95% intervals")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.89, wspace=0.24)
    save(fig, "budget_regime")


def checkpoint_selection(paper: dict) -> None:
    rows = paper["checkpoint_interpolation"]["rows"]
    alphas = [0.70, 0.78, 0.84, 0.90, 0.95]
    hp = [100 * one(rows, alpha=a, gate="hotpot_selector", method="learned_topk_paths")["pcr"] for a in alphas]
    mq = [100 * one(rows, alpha=a, gate="musique_dev_selector", method="learned_topk_paths")["pcr"] for a in alphas]
    f1 = []
    for a in alphas:
        hits = [r for r in rows if r["alpha"] == a and r["gate"] == "hotpot_downstream" and r["method"] == "learned_topk_paths"]
        f1.append(100 * hits[0]["answer_f1"] if hits else None)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.98))
    for ax, vals, title, ylab in ((axes[0], hp, "(a) HotpotQA validation", "PCR at 8% budget"),
                                  (axes[1], mq, "(b) MuSiQue development", "PCR at 512 tokens"),
                                  (axes[2], f1, "(c) HotpotQA validation", "answer F1 at 8% budget")):
        xs = [a for a, v in zip(alphas, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color=GRAY, marker="o", zorder=3)
        sel = alphas.index(0.90)
        ax.scatter([0.90], [vals[sel]], s=46, color=BLUE, edgecolor=WHITE, linewidth=0.9, zorder=5)
        ax.annotate(f"selected: {vals[sel]:.1f}", (0.90, vals[sel]), textcoords="offset points", xytext=(-4, 8), ha="right", fontsize=7.2)
        ax.set_xticks(alphas, [f"{a:.2f}" for a in alphas])
        ax.set_xlabel("interpolation weight α")
        ax.set_ylabel(ylab)
        span = max(ys) - min(ys)
        ax.set_ylim(min(ys) - 0.25 * span - 0.3, max(ys) + 0.45 * span + 0.3)
        ax.grid(axis="y")
        panel_title(ax, title)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.21, top=0.88, wspace=0.36)
    save(fig, "checkpoint_selection")


def downstream_landscape(paper: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.07), sharex=True)
    tasks = [("hotpot_qwen25", "HotpotQA", 128), ("twowiki_qwen25", "2WikiMultiHopQA", 100), ("musique_qwen25", "MuSiQue", 128)]
    methods = [("oracle_adaptive", "Oracle", GREEN, "D"), ("learned_topk_paths", "ProofKV chains", BLUE, "o"),
               ("learned_and_or_aux", "AND/OR + auxiliary fill", YELLOW, "^"), ("entity_graph_adaptive", "Entity graph", GRAY, "s")]
    budgets = [128, 256, 512]
    for ax, (key, label, n) in zip(axes, tasks):
        rows = paper["downstream_matrices"][key]["rows"]
        for method, mlabel, color, marker in methods:
            vals = [100 * one(rows, budget_name=f"fixed{b}", method=method)["answer_f1"] for b in budgets]
            ax.plot(range(3), vals, color=color, marker=marker, label=mlabel, zorder=4 if method == "learned_topk_paths" else 3)
        ax.set_xticks(range(3), [str(b) for b in budgets])
        ax.set_xlabel("retained tokens")
        ax.grid(axis="y")
        panel_title(ax, f"{label} ($n$={n})")
    axes[0].set_ylabel("answer F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.02), columnspacing=1.6)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.80, wspace=0.20)
    save(fig, "downstream_landscape")


def topology_depth(paper: dict) -> None:
    rows = paper["topology_full_matrix"]["records"]
    hops = ["1", "2", "3", "4", "5", "6", "8"]
    specs = [("oracle", "Oracle", GREEN, "D"), ("learned_and_or", "Intact chains", BLUE, "o"),
             ("broken_proof_same_count", "One span replaced", YELLOW, "^"), ("edge_shuffled", "Link targets shuffled", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(6.3, 2.12))
    for method, label, color, marker in specs:
        vals = [100 * one(rows, axis="hops", axis_value=h, budget_spec="512", method=method)["pcr"] for h in hops]
        ax.plot([int(h) for h in hops], vals, color=color, marker=marker, label=label)
        if method in ("oracle", "learned_and_or"):
            ax.text(8.18, vals[-1], f"{vals[-1]:.1f}", va="center", ha="left", fontsize=7.2)
    ax.set_xlabel("links required by the gold chain")
    ax.set_ylabel("proof-complete recall (%)")
    ax.set_xticks([int(h) for h in hops])
    ax.set_xlim(0.7, 8.9)
    ax.set_ylim(-4, 106)
    ax.grid(axis="y")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.17), columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.86)
    save(fig, "topology_depth")


def packed_scaling(paper: dict) -> None:
    rows = paper["packed_hotpot_scale"]["rows"]
    lengths = [8000, 16000, 32000]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.07))
    specs = [("oracle_adaptive", "Oracle", GREEN, "D"), ("learned_topk_paths", "ProofKV chains", BLUE, "o"),
             ("full", "Full context", GRAY_DARK, "s"), ("entity_graph_adaptive", "Entity graph", GRAY, "^")]
    for method, label, color, marker in specs:
        vals = [100 * one(rows, target_prompt_tokens=n, method=method)["answer_f1"] for n in lengths]
        ax1.plot(range(3), vals, color=color, marker=marker, label=label, zorder=4 if method == "learned_topk_paths" else 3)
    ax1.set_xticks(range(3), ["8k", "16k", "32k"])
    ax1.set_xlabel("prompt length (tokens)")
    ax1.set_ylabel("answer F1")
    ax1.set_ylim(48, 101)
    ax1.set_yticks([50, 60, 70, 80, 90])
    ax1.grid(axis="y")
    ax1.legend(loc="upper right", ncol=2, columnspacing=1.4, labelspacing=0.3, borderpad=0.1)
    panel_title(ax1, "(a) Accuracy as distractors grow")
    proof = [n * one(rows, target_prompt_tokens=n, method="learned_topk_paths")["retained_ratio"] for n in lengths]
    entity = [n * one(rows, target_prompt_tokens=n, method="entity_graph_adaptive")["retained_ratio"] for n in lengths]
    ax2.plot(range(3), entity, color=GRAY, marker="^", label="Entity graph (8% of prompt)")
    ax2.plot(range(3), proof, color=BLUE, marker="o", label="ProofKV chains")
    for x, v in zip(range(3), proof):
        ax2.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7.2)
    ax2.annotate(f"{entity[-1]:.0f}", (2, entity[-1]), textcoords="offset points", xytext=(-6, -2), ha="right", fontsize=7.2)
    ax2.set_xticks(range(3), ["8k", "16k", "32k"])
    ax2.set_xlabel("prompt length (tokens)")
    ax2.set_ylabel("retained prompt tokens")
    ax2.set_ylim(0, 2900)
    ax2.grid(axis="y")
    ax2.legend(loc="upper left")
    panel_title(ax2, "(b) Memory actually retained")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.89, wspace=0.24)
    save(fig, "packed_scaling")


def routing_frontier(routing: dict) -> None:
    rows = routing["V25"]["budget_rows"]
    budgets = [r["budget_percent"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.07), gridspec_kw={"width_ratios": [0.9, 1.1]})
    f1 = [r["controller_f1"] for r in rows]
    ret = [r["retained_percent"] for r in rows]
    ax1.plot(ret, f1, color=BLUE, marker="o")
    for x, yv, b in zip(ret, f1, budgets):
        ax1.annotate(f"{yv:.1f}  (budget {b:.0f}%)", (x, yv), textcoords="offset points", xytext=(7, -3), ha="left", fontsize=7.2)
    ax1.set_xlabel("retained prompt tokens (%)")
    ax1.set_ylabel("task-macro F1")
    ax1.set_xlim(2.5, 15.5)
    ax1.set_ylim(31.5, 42.5)
    ax1.grid(axis="y")
    panel_title(ax1, "(a) Margin-routed selection")
    width = 0.24
    specs = [("delta_entity", "vs entity graph", GRAY), ("delta_root", "vs same scorer, independent", GRAY_DARK), ("delta_and_or", "vs chains only", BLUE)]
    for k, (key, label, color) in enumerate(specs):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(rows))]
        vals = [r[key] for r in rows]
        ax2.bar(xs, vals, width=width, color=color, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax2.text(x, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_xticks(range(len(rows)), [f"{b:.0f}%" for b in budgets])
    ax2.set_xlabel("requested budget")
    ax2.set_ylabel("F1 gain of routing")
    ax2.set_ylim(0, 13.8)
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", ncol=1)
    panel_title(ax2, "(b) Gain over each fixed selector")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.89, wspace=0.22)
    save(fig, "routing_frontier")


def serving(paper: dict) -> None:
    audit = paper["targeted_independent_audit"]["group1"]
    labels = ["8k in\n256 out", "8k in\n1024 out", "32k in\n256 out", "30k in\n1024 out"]
    by = {label: {} for label in labels}
    for group in audit:
        prefix = "8k" if "8k" in group["path"] else ("32k" if "32k" in group["path"] else "30k")
        for row in group["comparisons"]:
            by[f"{prefix} in\n{row['output_tokens']} out"][row["mode"]] = row["paired_speedup"]
    fig, ax = plt.subplots(figsize=(3.03, 2.55))
    specs = [("precompiled", "Routes compiled in advance", GRAY, "o"), ("shared_index", "Reused index, end to end", BLUE, "s"),
             ("cold", "Index built per question", ORANGE, "^")]
    for (mode, label, color, marker), off in zip(specs, (-0.2, 0.0, 0.2)):
        means = [by[l][mode]["mean"] for l in labels]
        lo = [m - by[l][mode]["ci95_low"] for l, m in zip(labels, means)]
        hi = [by[l][mode]["ci95_high"] - m for l, m in zip(labels, means)]
        ax.errorbar([i + off for i in range(4)], means, yerr=[lo, hi], fmt=marker, color=color, markersize=5,
                    markeredgecolor=WHITE, markeredgewidth=0.8, elinewidth=1.1, capsize=0, label=label, zorder=3)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(range(4), labels)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.5, 4.9)
    ax.set_ylabel("speedup over full context (×)")
    ax.grid(axis="y")
    handles = [Line2D([], [], marker=marker, linestyle="none", color=color, markersize=5, markeredgecolor=WHITE, markeredgewidth=0.8, label=label)
               for _, label, color, marker in specs]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.01), borderpad=0.1, labelspacing=0.3, handletextpad=0.3)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.79)
    save(fig, "serving")


def reuse_and_gate(paper: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.02))
    records = paper["multiquery"]["records"]
    q = [r["queries"] for r in records]
    axes[0].plot(range(len(q)), [r["amortized_offline_index_ms"] for r in records], color=BLUE, marker="o")
    axes[0].set_ylabel("index cost per question (ms)")
    panel_title(axes[0], "(a) Index cost amortizes")
    axes[1].plot(range(len(q)), [100 * r["union_context_ratio"] for r in records], color=BLUE, marker="o")
    axes[1].set_ylabel("union of selections (% of context)")
    axes[1].set_ylim(0, 16)
    panel_title(axes[1], "(b) Selections overlap")
    for ax in axes[:2]:
        ax.set_xticks(range(len(q)), [str(v) for v in q])
        ax.set_xlabel("questions per document")
        ax.grid(axis="y")
    curve = paper["risk_controller"]["fixed_8pct"]["test_curve"]
    cov = [100 * r["coverage"] for r in curve]
    axes[2].plot(cov, [100 * r["empirical_risk"] for r in curve], color=ORANGE, label="observed error")
    axes[2].plot(cov, [100 * r["conservative_risk_upper"] for r in curve], color=GRAY, label="upper bound")
    axes[2].axhline(2, color=INK2, lw=0.8)
    axes[2].text(99, 2.6, "2% target", fontsize=7.0, color=INK2, ha="right", va="bottom")
    axes[2].set_xlabel("compressed coverage (%)")
    axes[2].set_ylabel("selective error (%)")
    axes[2].grid(axis="y")
    axes[2].legend(loc="upper left")
    panel_title(axes[2], "(c) Confidence-only gate")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.21, top=0.88, wspace=0.36)
    save(fig, "reuse_and_gate")


def compiler_diagnostics(paper: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.02))
    halos = [0, 8, 16]
    rows = paper["contextualization_gap"]["sample_records"]
    acc = [100 * one(rows, task_type="multi_hop", repair_halo=h)["compact_correctness"] for h in halos]
    axes[0].bar(range(3), acc, width=0.5, color=[GRAY, BLUE, BLUE], zorder=3)
    for x, v in zip(range(3), acc):
        axes[0].text(x, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    axes[0].set_ylim(88, 102.5)
    axes[0].set_xticks(range(3), [str(h) for h in halos])
    axes[0].set_xlabel("boundary halo (tokens per side)")
    axes[0].set_ylabel("multi-hop exactness (%)")
    axes[0].grid(axis="y")
    panel_title(axes[0], "(a) Boundary halo")
    pilot = paper["tiered_precision_pilot"]["summaries"]
    hold = paper["tiered_precision"]["summaries"][0]
    labels = ["INT8\npilot", "INT8\nholdout", "INT4\npilot"]
    storage = [pilot[0]["mean_storage_reduction_factor"], hold["mean_storage_reduction_factor"], pilot[1]["mean_storage_reduction_factor"]]
    exact = [100 * pilot[0]["exact_generation_match_rate"], 100 * hold["exact_generation_match_rate"], 100 * pilot[1]["exact_generation_match_rate"]]
    axes[1].bar(range(3), storage, width=0.5, color=[BLUE, BLUE, ORANGE], zorder=3)
    for x, (s, e) in enumerate(zip(storage, exact)):
        axes[1].text(x, s + 0.05, f"{e:.0f}%", ha="center", va="bottom", fontsize=7.2)
    axes[1].set_xticks(range(3), labels)
    axes[1].set_ylabel("storage reduction (×)")
    axes[1].set_ylim(0, 2.5)
    axes[1].grid(axis="y")
    axes[1].text(0.03, 0.97, "labels: share of exact outputs", transform=axes[1].transAxes, fontsize=6.8, color=INK2, va="top")
    panel_title(axes[1], "(b) Auxiliary precision")
    primary = [r for r in paper["physical_route_frontier"]["comparisons"] if r["primary"]]
    for out_tokens, color, marker in ((256, BLUE, "o"), (1024, GRAY, "s")):
        sel = sorted([r for r in primary if r["output_tokens"] == out_tokens], key=lambda r: r["context_length_requested"])
        axes[2].plot(range(len(sel)), [r["total_speedup_vs_gather"] for r in sel], color=color, marker=marker, label=f"{out_tokens} output tokens")
        axes[2].set_xticks(range(len(sel)), [f"{r['context_length_requested'] // 1000}k" for r in sel])
    axes[2].axhline(1.0, color=INK2, lw=0.8)
    axes[2].set_xlabel("context length (tokens)")
    axes[2].set_ylabel("speedup over gather (×)")
    axes[2].set_ylim(0.8, 3.4)
    axes[2].grid(axis="y")
    axes[2].legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    panel_title(axes[2], "(c) Route prefill")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.21, top=0.88, wspace=0.36)
    save(fig, "compiler_diagnostics")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--routing", required=True)
    parser.add_argument("--followup", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    OUT = Path(args.out)
    setup()
    paper = json.load(open(args.paper))
    routing = json.load(open(args.routing))
    follow = json.load(open(args.followup))
    overview()
    evidence(paper, follow)
    budget_regime(follow)
    checkpoint_selection(paper)
    downstream_landscape(paper)
    topology_depth(paper)
    packed_scaling(paper)
    routing_frontier(routing)
    serving(paper)
    reuse_and_gate(paper)
    compiler_diagnostics(paper)


if __name__ == "__main__":
    main()
