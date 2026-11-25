"""Render both planar GB comparison figures from exported tables."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
DATA=ROOT/'results'
METHODS=('EDH','EFS','Torque_anchor+EFS','Shell_torque')
TESTS=(410082,410093)
WIDTHS=(500,1000,2000)

COLORS = ("#202020", "#1266AA", "#A21B59", "#AA4B17")
SYMBOLS = ("o", "s", "D", "^")
LABELS = ("Direct count", "Count anchor + force", "Torque anchor + force", "Shell torque")


def read(name):
    with (DATA / name).open(encoding="utf-8") as stream:
        return [{k: (v if k in ("method", "quantity") else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def match(rows, **values):
    return next(r for r in rows if all(r[k] == v for k, v in values.items()))


def points(ax, x, rows, y, lo, hi, color, marker, label=None):
    value = np.array([r[y] for r in rows])
    lower = np.array([r[lo] for r in rows])
    upper = np.array([r[hi] for r in rows])
    ax.vlines(x, lower, upper, color=color, lw=.9)
    ax.plot(x, value, marker, color=color, ms=4.6, mfc="white", label=label)


def save(fig, n):
    (OUT / "figures").mkdir(exist_ok=True)
    stem = OUT / "figures" / f"Fig. {n}"
    fig.savefig(str(stem)+".eps")
    fig.savefig(str(stem)+".png", dpi=250)
    plt.close(fig)


def plot_joint():
    rows = read("joint_errors.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 6.5))
    fig.subplots_adjust(left=.14, right=.98, bottom=.10, top=.87, hspace=.46, wspace=.47)
    n = np.array([1, 3, 6, 12])
    x = np.arange(len(n))
    shown = [r for r in rows if r["M"] == 1000 and r["bootstrap_block"] == 500
             and r["quantity"] == "full"]
    ratio_limits = (min(r["ratio_lo"] for r in shown)-.045,
                    max(1., max(r["ratio_hi"] for r in shown))+.045)
    for j, seed in enumerate(TESTS):
        ax = axes[0, j]
        subset = [r for r in rows if r["seed"] == seed and r["M"] == 1000
                  and r["bootstrap_block"] == 500]
        r = lambda q, d: [match(subset, bins=b, quantity=q, dr=d) for b in n]
        points(ax, x-.17, r("anchor", .025), "EDH", "EDH_lo", "EDH_hi",
               "#705030", "^", "Angular-cell mean")
        for off, name, c, symbol in zip((0., .17), ("EDH", "EFS"), COLORS, SYMBOLS):
            points(ax, x+off, r("shape", .025), name, name+"_lo", name+"_hi",
                   c, symbol, f"Radial shape ({name})")
        ax.set_title(f"({chr(97+j)})  Seed {seed}")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("RMS\nerror", rotation=0, labelpad=26, va="center")
        ax.set_xticks(x, n**2)
        ax = axes[1, j]
        for off, d, color, mark in zip((-.10, .10), (.025, .05), COLORS, SYMBOLS):
            points(ax, x+off, r("full", d), "ratio", "ratio_lo", "ratio_hi", color, mark,
                   rf"$\Delta r={d}$")
        ax.axhline(1, color=".55", lw=.8, ls="--")
        ax.set_ylim(*ratio_limits)
        ax.set_xticks(x, n**2)
        ax.set_xlabel("Number of angular cells")
        ax.set_ylabel("RMS ratio\nEFS / EDH", rotation=0, labelpad=26, va="center")
        ax.set_title(f"({chr(99+j)})  Full annulus")
    fig.legend(*axes[0,0].get_legend_handles_labels(), ncol=3,
               loc="upper center", bbox_to_anchor=(.54, .99))
    axes[1, 0].legend(loc="lower right")
    save(fig, 2)


def plot_mode():
    rows, profiles = read("mode_errors.csv"), read("mode_profiles.csv")
    use = METHODS[:len(set(r["method"] for r in rows))]
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.5))
    fig.subplots_adjust(left=.14, right=.98, bottom=.10, top=.88, hspace=.46, wspace=.50)
    for j, seed in enumerate(TESTS):
        ax = axes[0, j]
        for k, method in enumerate(use):
            frame = [r for r in profiles if r["seed"] == seed and r["block"] == 500
                     and r["method"] == method and r["left"] >= .9-1e-10 and r["right"] <= 5.5+1e-10]
            x = [(r["left"]+r["right"])/2 for r in frame]
            if k == 0:
                ref = np.array([r["reference"] for r in frame])
                ax.fill_between(x, 1000*(np.array([r["reference_lo"] for r in frame])-ref),
                                1000*(np.array([r["reference_hi"] for r in frame])-ref),
                                facecolor=".91", label="Reference uncertainty")
            ax.plot(x, [1000*r["difference"] for r in frame], color=COLORS[k], lw=.9,
                    ls=("-", "-", "--", "-.")[k], label=LABELS[k])
        ax.axhline(0, color=".6", lw=.7)
        ax.set_title(f"({chr(97+j)})  Seed {seed}")
        ax.set_xlabel(r"$r/\sigma_0$")
        ax.set_ylabel(r"$10^3$" + "\nResidual", rotation=0, labelpad=24, va="center")
        ax = axes[1, j]
        for k, method in enumerate(use):
            subset = [match(rows, seed=seed, bootstrap_block=500, method=method, M=m)
                      for m in WIDTHS]
            scaled = [{key: r[key]*1000 for key in ("RMS", "lo", "hi")} for r in subset]
            points(ax, np.arange(3)+(k-(len(use)-1)/2)*.105, scaled, "RMS", "lo", "hi",
                   COLORS[k], SYMBOLS[k])
        ax.set_xticks(np.arange(3), WIDTHS)
        ax.set_title(f"({chr(99+j)})  Seed {seed}")
        ax.set_xlabel("Retained configurations")
        ax.set_ylabel(r"$10^3$" + "\nRMS error", rotation=0, labelpad=24, va="center")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles[1:]+handles[:1], labels[1:]+labels[:1], ncol=len(labels),
               loc="upper center", bbox_to_anchor=(.54, .99), fontsize=9)
    save(fig, 3)


plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "custom", "mathtext.it":"Times New Roman:italic", "mathtext.rm":"Times New Roman", "mathtext.bf":"Times New Roman:bold", "legend.frameon": False,
                     "font.size": 10, "ps.fonttype": 42})

if __name__ == "__main__":
    plot_joint()
    plot_mode()
