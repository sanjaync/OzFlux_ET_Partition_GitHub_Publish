"""
compare_and_plot_v3.py
======================
Cross-method comparison of ET partitioning results from:
  1. TEA (Nelson et al. 2018)       - from TEA_output_original/csv/
  2. Zhou/uWUE (Zhou et al. 2016)   - from output/
  3. Perez-Priego (2018)            - from output/

Generates:
  - Per-site daily comparison CSVs (output/comparison_v3/)
  - Per-site 4-panel diagnostic plots (output/plots_v3/)
  - Cross-site summary statistics CSV
  - Cross-site multi-panel overview plot

Author: Antigravity AI for sanjays
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# Configuration
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.edgecolor': '#888888',
    'axes.linewidth': 0.6,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.4,
    'figure.dpi': 150,
})

COLORS = {
    'TEA':          '#E74C3C',   # Rich Red
    'Zhou_uWUE':    '#2980B9',   # Ocean Blue
    'Perez-Priego': '#27AE60',   # Emerald Green
}

work_dir  = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition"
tea_csv_dir   = os.path.join(work_dir, "TEA_output_original", "csv")
output_dir    = os.path.join(work_dir, "output")
comparison_dir = os.path.join(output_dir, "comparison_v3")
plots_dir     = os.path.join(output_dir, "plots_v3")

os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ============================================================================
# Discover all sites that have all 3 methods
# ============================================================================

def discover_sites():
    """Return sorted list of sites that have TEA, Zhou, and PP daily CSVs."""
    tea_sites = set()
    for f in glob.glob(os.path.join(tea_csv_dir, "*_TEA_original_daily.csv")):
        site = os.path.basename(f).replace("_TEA_original_daily.csv", "")
        tea_sites.add(site)

    zhou_sites = set()
    for f in glob.glob(os.path.join(output_dir, "*_Zhou_daily.csv")):
        site = os.path.basename(f).replace("_Zhou_daily.csv", "")
        zhou_sites.add(site)

    pp_sites = set()
    for f in glob.glob(os.path.join(output_dir, "*_PerezPriego_daily.csv")):
        bn = os.path.basename(f)
        parts = bn.split('_')
        if len(parts) == 3:  # site_PerezPriego_daily.csv (skip year splits)
            pp_sites.add(parts[0])

    common = sorted(tea_sites & zhou_sites & pp_sites)
    print(f"Discovered {len(common)} sites with all 3 methods.")
    return common


def load_site_data(site):
    """Load and align daily data for a single site from all 3 methods.
    
    Returns a merged DataFrame with columns: date, TEA, Zhou_uWUE, Perez-Priego
    """
    # --- TEA (new outputs) ---
    tea_path = os.path.join(tea_csv_dir, f"{site}_TEA_original_daily.csv")
    df_tea = pd.read_csv(tea_path)
    df_tea['date'] = pd.to_datetime(df_tea['date'])
    df_tea = df_tea[['date', 'T_ET_ratio']].rename(columns={'T_ET_ratio': 'TEA'})

    # --- Zhou ---
    zhou_path = os.path.join(output_dir, f"{site}_Zhou_daily.csv")
    df_zhou = pd.read_csv(zhou_path)
    df_zhou['date'] = pd.to_datetime(df_zhou['date'])
    df_zhou = df_zhou[['date', 'T_ET_ratio']].rename(columns={'T_ET_ratio': 'Zhou_uWUE'})

    # --- Perez-Priego ---
    pp_path = os.path.join(output_dir, f"{site}_PerezPriego_daily.csv")
    df_pp = pd.read_csv(pp_path)
    df_pp['date'] = pd.to_datetime(df_pp['date'], errors='coerce')
    df_pp = df_pp.dropna(subset=['date'])
    df_pp = df_pp[['date', 'T_ET_ratio']].rename(columns={'T_ET_ratio': 'Perez-Priego'})

    # Merge on date (outer to keep all days)
    merged = pd.merge(df_tea, df_zhou, on='date', how='outer')
    merged = pd.merge(merged, df_pp, on='date', how='outer')
    merged.sort_values('date', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Clip to physical range [0, 1]
    for col in ['TEA', 'Zhou_uWUE', 'Perez-Priego']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        merged.loc[(merged[col] < 0) | (merged[col] > 1), col] = np.nan

    return merged


def compute_stats(merged, site):
    """Compute summary statistics for a merged site DataFrame."""
    stats = {'site': site}
    for method in ['TEA', 'Zhou_uWUE', 'Perez-Priego']:
        v = merged[method].dropna()
        stats[f'{method}_mean'] = v.mean() if len(v) > 0 else np.nan
        stats[f'{method}_median'] = v.median() if len(v) > 0 else np.nan
        stats[f'{method}_std'] = v.std() if len(v) > 0 else np.nan
        stats[f'{method}_n'] = len(v)

    overlap = merged.dropna(subset=['TEA', 'Zhou_uWUE', 'Perez-Priego'])
    n_overlap = len(overlap)
    stats['n_overlap'] = n_overlap

    if n_overlap > 30:
        stats['r_TEA_Zhou'] = overlap['TEA'].corr(overlap['Zhou_uWUE'])
        stats['r_TEA_PP']   = overlap['TEA'].corr(overlap['Perez-Priego'])
        stats['r_Zhou_PP']  = overlap['Zhou_uWUE'].corr(overlap['Perez-Priego'])
    else:
        stats['r_TEA_Zhou'] = np.nan
        stats['r_TEA_PP']   = np.nan
        stats['r_Zhou_PP']  = np.nan

    return stats


def plot_site(merged, site, stats, save_path):
    """Generate a 4-panel diagnostic figure for a single site."""
    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 2, hspace=0.30, wspace=0.28)
    fig.suptitle(f"ET Partitioning Comparison — {site}",
                 fontsize=15, fontweight='bold', y=0.97)

    methods = list(COLORS.keys())

    # ------------------------------------------------------------------
    # Panel A: Monthly Mean Timeseries
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("A.  Monthly Mean T/ET Timeseries", fontsize=11, fontweight='semibold', pad=8)

    ts = merged.set_index('date')
    monthly = ts[methods].resample('MS').mean()

    for m, c in COLORS.items():
        ax1.plot(monthly.index, monthly[m], color=c, linewidth=1.4, alpha=0.9, label=m)

    ax1.set_ylabel("T / ET")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='none', loc='best')

    # ------------------------------------------------------------------
    # Panel B: Seasonal Climatology with ± 1 SD
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("B.  Seasonal Climatology (Mean ± SD)", fontsize=11, fontweight='semibold', pad=8)

    merged_copy = merged.copy()
    merged_copy['month'] = merged_copy['date'].dt.month
    mmean = merged_copy.groupby('month')[methods].mean()
    mstd  = merged_copy.groupby('month')[methods].std()

    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    for m, c in COLORS.items():
        ax2.plot(mmean.index, mmean[m], color=c, marker='o', markersize=4, linewidth=1.8, label=m)
        lo = np.clip(mmean[m] - mstd[m], 0, 1)
        hi = np.clip(mmean[m] + mstd[m], 0, 1)
        ax2.fill_between(mmean.index, lo, hi, color=c, alpha=0.10)

    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_labels)
    ax2.set_ylabel("Mean T / ET")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ------------------------------------------------------------------
    # Panel C: Pairwise Scatter (Zhou vs PP coloured by TEA)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("C.  Zhou vs Pérez-Priego (colour = TEA)", fontsize=11, fontweight='semibold', pad=8)

    sc_df = merged.dropna(subset=['Zhou_uWUE', 'Perez-Priego', 'TEA'])
    if len(sc_df) > 10:
        sc = ax3.scatter(sc_df['Zhou_uWUE'], sc_df['Perez-Priego'],
                         c=sc_df['TEA'], cmap='RdYlGn', vmin=0, vmax=1,
                         s=5, alpha=0.35, edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax3, pad=0.02, shrink=0.85)
        cbar.set_label('TEA T/ET', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # 1:1 line
        ax3.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=0.8, label='1:1')

        # Linear fit
        x, y = sc_df['Zhou_uWUE'].values, sc_df['Perez-Priego'].values
        slope, intercept = np.polyfit(x, y, 1)
        ax3.plot([0, 1], [intercept, slope + intercept], color='#E74C3C', linewidth=1.2, label='Fit')

        r = stats.get('r_Zhou_PP', np.nan)
        r2 = r**2 if np.isfinite(r) else np.nan
        rmse = np.sqrt(np.mean((y - x)**2))
        bias = np.mean(y - x)
        txt = f"R² = {r2:.2f}\nRMSE = {rmse:.3f}\nBias = {bias:+.3f}\nSlope = {slope:.2f}"
        ax3.text(0.04, 0.96, txt, transform=ax3.transAxes, fontsize=8,
                 va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='none'))

        ax3.legend(fontsize=7, loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    else:
        ax3.text(0.5, 0.5, "Insufficient overlap", ha='center', va='center', fontsize=10)

    ax3.set_xlabel("Zhou/uWUE  T/ET")
    ax3.set_ylabel("Pérez-Priego  T/ET")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ------------------------------------------------------------------
    # Panel D: Violin / Histogram (overlaid PDFs)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("D.  T/ET Distribution (PDF)", fontsize=11, fontweight='semibold', pad=8)

    bins = np.linspace(0, 1, 41)
    for m, c in COLORS.items():
        v = merged[m].dropna().values
        if len(v) > 0:
            ax4.hist(v, bins=bins, density=True, histtype='stepfilled', color=c, alpha=0.12)
            ax4.hist(v, bins=bins, density=True, histtype='step', color=c, linewidth=1.4, label=m)

    ax4.set_xlabel("T / ET")
    ax4.set_ylabel("Probability Density")
    ax4.set_xlim(0, 1)
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='none')

    # Save
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_overview(all_stats, save_path):
    """Generate a 2-panel overview figure summarising all sites."""
    df = pd.DataFrame(all_stats).sort_values('site')
    n = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, n * 0.28)))
    fig.suptitle(f"Network-Wide T/ET Comparison — {n} OzFlux Sites",
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: Dot-plot of mean T/ET per method
    y_pos = np.arange(n)
    for method, color in COLORS.items():
        col = f'{method}_mean'
        ax1.scatter(df[col].values, y_pos, color=color, s=30, zorder=3, label=method, alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['site'].values, fontsize=7)
    ax1.set_xlabel("Mean Daily T/ET")
    ax1.set_xlim(0, 1)
    ax1.set_title("A.  Mean T/ET by Method", fontsize=12, fontweight='semibold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax1.legend(fontsize=8, loc='lower right', frameon=True)
    ax1.invert_yaxis()

    # Panel B: Correlation heatmap-style bar chart
    corr_cols = ['r_TEA_Zhou', 'r_TEA_PP', 'r_Zhou_PP']
    corr_labels = ['r(TEA, Zhou)', 'r(TEA, PP)', 'r(Zhou, PP)']
    bar_colors = ['#E74C3C', '#8E44AD', '#2980B9']
    width = 0.25

    for i, (col, label, bc) in enumerate(zip(corr_cols, corr_labels, bar_colors)):
        ax2.barh(y_pos + (i - 1) * width, df[col].values, height=width,
                 color=bc, alpha=0.7, label=label)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df['site'].values, fontsize=7)
    ax2.set_xlabel("Pearson r")
    ax2.set_xlim(-0.3, 1.0)
    ax2.axvline(0, color='#333333', linewidth=0.5)
    ax2.set_title("B.  Cross-Method Correlation", fontsize=12, fontweight='semibold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax2.legend(fontsize=7, loc='lower right', frameon=True)
    ax2.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    sites = discover_sites()
    all_stats = []

    for i, site in enumerate(sites):
        print(f"[{i+1}/{len(sites)}] {site} ... ", end='', flush=True)
        try:
            merged = load_site_data(site)
            stats = compute_stats(merged, site)
            all_stats.append(stats)

            # Save comparison CSV
            csv_path = os.path.join(comparison_dir, f"{site}_daily_comparison.csv")
            merged.to_csv(csv_path, index=False)

            # Per-site plot
            plot_path = os.path.join(plots_dir, f"{site}_comparison.png")
            plot_site(merged, site, stats, plot_path)

            tea_m  = stats.get('TEA_mean', np.nan)
            zhou_m = stats.get('Zhou_uWUE_mean', np.nan)
            pp_m   = stats.get('Perez-Priego_mean', np.nan)
            print(f"TEA={tea_m:.3f}  Zhou={zhou_m:.3f}  PP={pp_m:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(comparison_dir, "methods_comparison_summary_v3.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

        # Overview plot
        overview_path = os.path.join(plots_dir, "network_overview.png")
        plot_overview(all_stats, overview_path)
        print(f"Saved overview plot to: {overview_path}")

        # Print table
        print("\n" + "=" * 90)
        print(f"{'Site':>30s}  {'TEA':>7s}  {'Zhou':>7s}  {'PP':>7s}  {'r(T,Z)':>7s}  {'r(T,P)':>7s}  {'r(Z,P)':>7s}")
        print("=" * 90)
        for _, r in summary_df.iterrows():
            fmt = lambda x: f"{x:.3f}" if np.isfinite(x) else "  N/A "
            print(f"{r['site']:>30s}  {fmt(r['TEA_mean'])}  {fmt(r['Zhou_uWUE_mean'])}  "
                  f"{fmt(r['Perez-Priego_mean'])}  {fmt(r.get('r_TEA_Zhou', np.nan))}  "
                  f"{fmt(r.get('r_TEA_PP', np.nan))}  {fmt(r.get('r_Zhou_PP', np.nan))}")

    print("\nDone!")
