import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

work_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition"
plots_dir = os.path.join(work_dir, "output", "nelson_replication_plots")
os.makedirs(plots_dir, exist_ok=True)

# 1. Load summary data
summary_path = os.path.join(work_dir, "output", "comparison_v3", "methods_comparison_summary_v3.csv")
df_summary = pd.read_csv(summary_path)

# 2. Load metadata
meta_path = os.path.join(work_dir, "output", "comparison_v3", "site_metadata.csv")
if os.path.exists(meta_path):
    df_meta = pd.read_csv(meta_path)
    df_summary = df_summary.merge(df_meta, on='site', how='left')
else:
    print("Metadata not found, relying on basic summary.")
    df_summary['biome'] = 'Unknown'
    df_summary['lat'] = np.nan
    df_summary['lon'] = np.nan
    df_summary['MAT'] = np.nan
    df_summary['MAP'] = np.nan

# Clean up IGBP names if they are messy
def clean_igbp(b):
    b = str(b)
    for t in ['SAV', 'EBF', 'GRA', 'ENF', 'DBF', 'CRO', 'WSA', 'CSH', 'OSH']:
        if t in b: return t
    if 'Woodland' in b: return 'WSA'
    if 'Savanna' in b: return 'SAV'
    if 'Grassland' in b: return 'GRA'
    if 'Forest' in b: return 'EBF'
    return 'Other'

df_summary['IGBP'] = df_summary['biome'].apply(clean_igbp)

# 3. Load daily data for all sites for density/hexbin plots
daily_files = sorted(glob.glob(os.path.join(work_dir, "output", "comparison_v3", "*_comparison_v3.csv")))
df_daily_list = []
for f in daily_files:
    d = pd.read_csv(f)
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'])
    df_daily_list.append(d)
if df_daily_list:
    df_daily = pd.concat(df_daily_list, ignore_index=True)
else:
    df_daily = pd.DataFrame()


# =====================================================================
# FIGURE 1: Site Map and Biome Distribution (Replicating Nelson Fig 1)
# =====================================================================
print("Generating Figure 1: Site Map...")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Basic map of Australia using scatter
biomes = df_summary['IGBP'].unique()
colors = sns.color_palette("Set2", len(biomes))

for b, c in zip(biomes, colors):
    subset = df_summary[df_summary['IGBP'] == b]
    ax.scatter(subset['lon'], subset['lat'], label=b, color=c, s=100, edgecolor='k', alpha=0.8)

ax.set_aspect('equal')
ax.set_title('Figure 1: OzFlux Site Distribution by IGBP Biome')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(title="Biome", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "fig1_site_map.png"))
plt.close()


# =====================================================================
# FIGURE 2: T/ET Biome Distributions (Replicating Nelson Fig 2)
# =====================================================================
print("Generating Figure 2: T/ET Biome Distributions...")
# Melt summary to long format
df_long = df_summary.melt(
    id_vars=['site', 'IGBP'], 
    value_vars=['TEA_mean', 'Zhou_uWUE_mean', 'Perez-Priego_mean'],
    var_name='Method', value_name='T/ET'
)
df_long['Method'] = df_long['Method'].map({
    'TEA_mean': 'TEA',
    'Zhou_uWUE_mean': 'Zhou',
    'Perez-Priego_mean': 'Pérez-Priego'
})

fig = plt.figure(figsize=(12, 6))
sns.violinplot(x='IGBP', y='T/ET', hue='Method', data=df_long, 
               inner="quartile", palette="muted", split=False)
plt.title('Figure 2: Distribution of T/ET by Method and Biome')
plt.xlabel('IGBP Biome')
plt.ylabel('Mean T/ET Fraction')
plt.ylim(0, 1.0)
plt.legend(title="Method", loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "fig2_biome_distributions.png"))
plt.close()


# =====================================================================
# FIGURE 3: Inter-method Agreement Scatter/Hexbins (Replicating Nelson)
# =====================================================================
print("Generating Figure 3: Inter-Method Agreement...")
if not df_daily.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = [
        ('T_ET_TEA', 'T_ET_Zhou', 'TEA', 'Zhou/uWUE'),
        ('T_ET_TEA', 'T_ET_PP', 'TEA', 'Pérez-Priego'),
        ('T_ET_Zhou', 'T_ET_PP', 'Zhou/uWUE', 'Pérez-Priego')
    ]
    
    for i, (x_col, y_col, x_label, y_label) in enumerate(pairs):
        ax = axes[i]
        valid = df_daily.dropna(subset=[x_col, y_col])
        if len(valid) == 0: continue
        
        hb = ax.hexbin(valid[x_col], valid[y_col], gridsize=50, cmap='viridis', 
                       mincnt=1, bins='log', extent=[0, 1, 0, 1])
        
        # 1:1 line
        ax.plot([0, 1], [0, 1], 'r--', lw=2)
        
        ax.set_xlabel(f'{x_label} T/ET')
        ax.set_ylabel(f'{y_label} T/ET')
        ax.set_title(f'{x_label} vs {y_label}')
        
        # Calculate stats
        r = valid[x_col].corr(valid[y_col])
        rmse = np.sqrt(np.mean((valid[x_col] - valid[y_col])**2))
        bias = np.mean(valid[y_col] - valid[x_col])
        
        textstr = '\n'.join((
            f'r = {r:.2f}',
            f'RMSE = {rmse:.2f}',
            f'Bias = {bias:.2f}'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        fig.colorbar(hb, ax=ax, label='log10(N)')

    plt.suptitle('Figure 3: Daily T/ET Inter-Method Agreement (All Sites)', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig3_intermethod_hexbin.png"))
    plt.close()


# =====================================================================
# FIGURE 4: Seasonal Dynamics (Replicating Nelson)
# =====================================================================
print("Generating Figure 4: Seasonal Dynamics...")
if not df_daily.empty:
    df_daily['month'] = df_daily['date'].dt.month
    
    # Merge biome info
    df_daily_biome = df_daily.merge(df_summary[['site', 'IGBP']], on='site', how='left')
    
    # Pick top 4 biomes by data volume
    top_biomes = df_daily_biome['IGBP'].value_counts().index[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, biome in enumerate(top_biomes):
        ax = axes[i]
        subset = df_daily_biome[df_daily_biome['IGBP'] == biome]
        
        monthly = subset.groupby('month').agg({
            'T_ET_TEA': 'mean',
            'T_ET_Zhou': 'mean',
            'T_ET_PP': 'mean'
        }).reset_index()
        
        ax.plot(monthly['month'], monthly['T_ET_TEA'], 'o-', label='TEA', color='C0', lw=2)
        ax.plot(monthly['month'], monthly['T_ET_Zhou'], 's-', label='Zhou', color='C1', lw=2)
        ax.plot(monthly['month'], monthly['T_ET_PP'], '^-', label='Pérez-Priego', color='C2', lw=2)
        
        ax.set_title(f'Biome: {biome}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean T/ET')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.5)
        if i == 0:
            ax.legend(loc='lower center', ncol=3)
            
    plt.suptitle('Figure 4: Seasonal Climatology of T/ET by Biome', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig4_seasonal_dynamics.png"))
    plt.close()


# =====================================================================
# FIGURE 5: Environmental Drivers (Replicating Nelson)
# =====================================================================
print("Generating Figure 5: Environmental Drivers...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, var in enumerate(['MAP', 'LAI']):
    if var not in df_summary.columns or df_summary[var].isna().all():
        print(f"Skipping {var} - no data.")
        continue
        
    for j, method in enumerate(['TEA_mean', 'Zhou_uWUE_mean', 'Perez-Priego_mean']):
        ax = axes[idx, j]
        valid = df_summary.dropna(subset=[var, method])
        
        method_name = method.replace('_mean', '').replace('_uWUE', '')
        
        ax.scatter(valid[var], valid[method], alpha=0.7, edgecolor='k', s=60)
        
        if len(valid) > 2:
            m, b = np.polyfit(valid[var], valid[method], 1)
            ax.plot(valid[var], m*valid[var] + b, 'r--', lw=2)
            r = valid[var].corr(valid[method])
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        
        unit = '(mm/yr)' if var == 'MAP' else '(m²/m²)' if var == 'LAI' else ''
        ax.set_xlabel(f'{var} {unit}')
        ax.set_ylabel(f'{method_name} T/ET')
        ax.set_ylim(0, 1.0)
        if idx == 0:
            ax.set_title(f'{method_name}')
            
plt.suptitle('Figure 5: Environmental Drivers of T/ET', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "fig5_environmental_drivers.png"))
plt.close()

print("All plots generated successfully in output/nelson_replication_plots/")
