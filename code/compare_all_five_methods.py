import os
import pandas as pd
import numpy as np

# Set directories
tea_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition"
opt_bf_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia_agent_3_scientific/output_corrected/combined"
out_dir = os.path.join(tea_dir, "output", "comparison_v4")

os.makedirs(out_dir, exist_ok=True)

# 1. Load the 3 empirical methods from methods_comparison_summary_v3.csv
summary_v3_path = os.path.join(tea_dir, "output", "comparison_v3", "methods_comparison_summary_v3.csv")
df_empirical = pd.read_csv(summary_v3_path)

# 2. Load the Agent Model results (BF and Optimal)
df_opt = pd.read_csv(os.path.join(opt_bf_dir, "results_opt__ozflux_1.csv"))
df_bf = pd.read_csv(os.path.join(opt_bf_dir, "results_bf__ozflux_1.csv"))

# Map Short Names to Long Names
site_map = {
    'AU-Adr': 'AdelaideRiver',
    'AU-ASM': 'AliceSpringsMulga1', # or 2? Let's check overlap later
    'AU-Alp': 'AlpinePeatland',
    'AU-Boy': 'Boyagin',
    'AU-Cal': 'Calperum',
    'AU-Ctr': 'CapeTribulation',
    'AU-Col': 'Collie',
    'AU-Cow': 'CowBay',
    'AU-Cum': 'CumberlandPlain',
    'AU-DaP': 'DalyPasture',
    'AU-DaS': 'DalyUncleared',
    'AU-Dig': 'DigbyPlantation',
    'AU-Eme': 'Emerald',
    'AU-Fle': 'FletcherviewTropicalRangeland',
    'AU-Fog': 'FoggDam',
    'AU-Gat': 'GatumPasture',
    'AU-Gin': 'Gingin',
    'AU-GWW': 'GreatWesternWoodlands',
    'AU-How': 'HowardSprings',
    'AU-Lit': 'Litchfield',
    'AU-Lon': 'Longreach',
    'AU-Lox': 'Loxton',
    'AU-Otw': 'Otway',
    'AU-RDF': 'RedDirtMelonFarm',
    'AU-Rgf': 'Ridgefield',
    'AU-Rig': 'RiggsCreek',
    'AU-Rob': 'RobsonCreek',
    'AU-Sam': 'Samford',
    'AU-SiP': 'SilverPlains',
    'AU-Stp': 'SturtPlains',
    'AU-Tum': 'Tumbarumba',
    'AU-Wal': 'WallabyCreek',
    'AU-War': 'Warra',
    'AU-Whr': 'Whroo',
    'AU-YarI': 'YarramundiIrrigated',
    'AU-Ync': 'Yanco'
}

df_opt['site_long'] = df_opt['siteID'].map(site_map)
df_bf['site_long'] = df_bf['siteID'].map(site_map)

# Get the relevant columns: tet_part
df_opt = df_opt[['site_long', 'tet_part']].rename(columns={'tet_part': 'Optimal'})
df_bf = df_bf[['site_long', 'tet_part']].rename(columns={'tet_part': 'BF'})

# 3. Merge everything
df_merged = pd.merge(df_empirical, df_opt, left_on='site', right_on='site_long', how='inner')
df_merged = pd.merge(df_merged, df_bf, left_on='site', right_on='site_long', how='inner')

# Keep only the relevant columns for the 5-way comparison
df_final = df_merged[['site', 'TEA_mean', 'Zhou_uWUE_mean', 'Perez-Priego_mean', 'BF', 'Optimal']]

# Save to CSV
out_csv = os.path.join(out_dir, "five_method_comparison.csv")
df_final.to_csv(out_csv, index=False)
print(f"Saved 5-way comparison for {len(df_final)} sites to {out_csv}")

# Generate a bar plot
import matplotlib.pyplot as plt

sites = df_final['site'].tolist()
x = np.arange(len(sites))
width = 0.15

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x - 2*width, df_final['TEA_mean'], width, label='TEA')
rects2 = ax.bar(x - width, df_final['Zhou_uWUE_mean'], width, label='Zhou (uWUE)')
rects3 = ax.bar(x, df_final['Perez-Priego_mean'], width, label='Perez-Priego')
rects4 = ax.bar(x + width, df_final['BF'], width, label='BF', color='gray')
rects5 = ax.bar(x + 2*width, df_final['Optimal'], width, label='Optimal', color='salmon')

ax.set_ylabel('Mean T/ET Fraction')
ax.set_title('Comparison of T/ET Partitioning Across 5 Methodologies')
ax.set_xticks(x)
ax.set_xticklabels(sites, rotation=45, ha='right')
ax.legend()
plt.tight_layout()

plot_out = os.path.join(out_dir, "five_method_comparison.png")
plt.savefig(plot_out, dpi=300)
print(f"Saved plot to {plot_out}")
