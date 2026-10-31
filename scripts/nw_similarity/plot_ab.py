# -*- coding: utf-8 -*-
"""Plot A/B from the two supplied NW similarity matrices."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.edgecolor': '#BCC2C8',
                     'axes.linewidth': 0.8})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True, sharey=True)
bins = np.arange(0, 105, 5)
for ax, letter, train, test, color in zip(
        axes, ['A', 'B'], ['S8754', 'S4346'], ['S669', 'S571'],
        ['#2878A5', '#D5803F']):
    path = OUT / '{}_vs_{}_similarity_pct.csv'.format(train, test)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = csv.reader(handle)
        header = next(rows)
        values = [row[1:] for row in rows]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(header) - 1:
        raise ValueError('Invalid matrix dimensions: ' + str(path))
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 100)).any():
        raise ValueError('Similarity values must be finite percentages')
    xs = matrix.ravel()
    ax.hist(xs, bins=bins, weights=np.full(len(xs), 100 / len(xs)),
            color=color, edgecolor='white', linewidth=0.8, alpha=0.92)
    ax.set_title('{}  {} vs {}'.format(letter, train, test), loc='left',
                 fontsize=12, weight='bold', pad=12)
    ax.text(0.98, 0.97, 'All unique WT pairs\nn = {:,}\nMedian = {:.1f}%'.format(
        len(xs), np.median(xs)), ha='right', va='top', transform=ax.transAxes,
        fontsize=9.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 3})
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 41, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel('NW sequence similarity (%)', labelpad=8)
    ax.set_ylabel('Fraction of sequence pairs')
    ax.tick_params(axis='y', labelleft=True)
    ax.grid(axis='y', color='#E8E9EB', linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

fig.text(.5, .025,
         'Needleman-Wunsch global alignment | BLOSUM62 | linear gap = -5 (including terminal gaps)\n'
         'Similarity = residue pairs with BLOSUM62 > 0 / all alignment columns, including gaps; unique WT sequences',
         ha='center', va='bottom', fontsize=8, color='#535960', linespacing=1.6)
fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.25, wspace=.27)
fig.savefig(OUT / 'NW_similarity_AB.png', dpi=350, facecolor='white')
plt.close(fig)
print('Saved NW_similarity_AB.png')
