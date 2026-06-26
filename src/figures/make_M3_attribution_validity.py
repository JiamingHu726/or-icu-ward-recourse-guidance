#!/usr/bin/env python3
"""Reproduce Figure M3: attribution validity checks.

Panel (a) uses the deposited zero-intervention control-alignment table.
Panel (b) reads canonical selected-schedule hash comparisons for the four
positive scalar pressure weights against the matched zero-price arm.

The script is deliberately limited to the final M3 evidence. It does not
reproduce superseded threshold or stacked-median figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--alignment', default='results/derived_tables/M3_zero_intervention_alignment.csv')
    ap.add_argument('--hash-audit', default='metadata/price_arm_schedule_equality.csv')
    ap.add_argument('--output-dir', default='figures')
    args = ap.parse_args()

    alignment = pd.read_csv(args.alignment)
    audit = pd.read_csv(args.hash_audit)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    ax = axes[0]
    for dataset, sub in alignment.groupby('dataset', sort=False):
        ax.plot(sub['n'], sub['control_alignment_gap'], marker='o', label=dataset)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title('(a) Zero intervention validation')
    ax.set_xlabel('Candidate pool size $n$')
    ax.set_ylabel('Control alignment gap')
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.legend(frameon=False)
    ax.text(0.5, 0.88, 'GermanOR & Mannino:\ncontrol gap = 0 at all $n$',
            transform=ax.transAxes, ha='center', va='top')

    ax = axes[1]
    arms = ['price_cal_w0p25', 'price_cal_w0p5', 'price_cal_w1', 'price_cal_w2']
    labels = ['$\\lambda=0.25$', '$\\lambda=0.5$', '$\\lambda=1$', '$\\lambda=2$']
    fractions = []
    for arm in arms:
        sub = audit[(audit['comparison_arm'] == arm) & (audit['status'].astype(str) == 'ok')].copy()
        same = sub['same_canonical_sha256'].astype(bool) & sub['same_raw_sha256'].astype(bool)
        fractions.append(float(same.mean()) if len(sub) else float('nan'))
    y = list(range(len(arms)))
    ax.barh(y, fractions)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Fraction of paired instances identical to no-price guidance')
    ax.set_title('(b) Scalar pressure pricing')
    ax.text(0.5, 0.08, '160/160 identical — no dose response',
            transform=ax.transAxes, ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(out / 'M3_attribution_validity.pdf', bbox_inches='tight')
    fig.savefig(out / 'M3_attribution_validity.png', dpi=300, bbox_inches='tight')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
