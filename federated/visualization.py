# quick plot script
from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_sensitivity(csv_path: str, x_col: str, out_path: str):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4))
    if 'dataset' in df.columns:
        for name, g in df.groupby('dataset'):
            g = g.sort_values(x_col)
            plt.plot(g[x_col], g['dice'], marker='o', label=name)
        plt.legend()
    else:
        df = df.sort_values(x_col)
        plt.plot(df[x_col], df['dice'], marker='o')
    plt.xlabel(x_col)
    plt.ylabel('Average Dice')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--x_col', required=True, choices=['num_points', 'num_task_tokens'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot_sensitivity(args.csv, args.x_col, args.out)
