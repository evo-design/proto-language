import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Agg')

# Set up font - use Liberation Sans (metrically equivalent to Arial)
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 6
# Tighten spacing between tick labels and ticks
plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 1.5
# Tighten axis label padding
plt.rcParams['axes.labelpad'] = 2

# Load data
tsv_df = pd.read_csv('analyze_af3_signaling_with_mmseqs.tsv', sep='\t')
with open('examples/scripts/human_systems_claude/outputs/vi_signaling_pathways__b2ar_to_tf_pathway__b2ar_to_tf_pathway/run_20260122_213114_848609/results.json', 'r') as f:
    native_data = json.load(f)

# Extract native values for each complex
native_metrics = {}
for complex_info in native_data['complex_scores']:
    cid = complex_info['complex_id']
    native_metrics[cid] = {
        'plddt': complex_info['af3_confidence']['plddt'],
        'iptm': complex_info['af3_confidence']['iptm'],
        'ptm': complex_info['af3_confidence']['ptm'],
    }
    # Get RMSD from pdb_comparisons (use first reference)
    pdb_comps = complex_info['pdb_comparisons']
    if pdb_comps:
        first_pdb = list(pdb_comps.keys())[0]
        native_metrics[cid]['rmsd'] = pdb_comps[first_pdb]['rmsd']

print("Native metrics:", native_metrics)
print("\nUnique complex_ids in TSV:", tsv_df['complex_id'].unique())

# Get unique complex IDs (excluding header)
complex_ids = tsv_df['complex_id'].unique()

# Define colors
slate_blue = '#6A7FDB'  # Main slate blue
slate_blue_edge = '#4A5FAB'  # Darker edge
pastel_red = '#FF9B9B'  # Pastel red for native
pastel_red_edge = '#E87979'  # Darker edge

# Convert mm to inches (1 mm = 0.03937 inches)
width_mm = 27
height_mm = 22
width_in = width_mm * 0.03937
height_in = height_mm * 0.03937

# Create figures for each complex and metric
for metric, ylabel in [('plddt', 'pLDDT'), ('rmsd', 'RMSD (Å)')]:
    for cid in complex_ids:
        # Filter data for this complex
        df_complex = tsv_df[tsv_df['complex_id'] == cid].copy()

        if len(df_complex) == 0:
            continue

        # Get native value if available
        native_x = 1.0  # 100% identity
        native_y = native_metrics.get(cid, {}).get(metric, None)

        # Create figure
        fig, ax = plt.subplots(figsize=(width_in, height_in))

        # Plot design points
        x = df_complex['mmseqs_weighted_score']
        y = df_complex[metric]

        ax.scatter(x, y, c=slate_blue, edgecolors=slate_blue_edge,
                   s=15, alpha=0.7, linewidths=0.3, zorder=2)

        # Plot native point if available
        if native_y is not None:
            ax.scatter([native_x], [native_y], c=pastel_red, edgecolors=pastel_red_edge,
                       s=25, alpha=0.9, linewidths=0.5, zorder=3, marker='D',
                       label='Native')

        # Labels and formatting
        ax.set_xlabel('MMseqs Weighted Score', fontsize=6)
        ax.set_ylabel(ylabel, fontsize=6)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Set x-axis to go from 0 to 1
        ax.set_xlim(0, 1.05)

        # Tight layout with minimal padding
        plt.tight_layout(pad=0.1)

        # Save figure
        safe_cid = cid.replace('::', '_').replace(' ', '_')
        filename = f'/home/claude/{safe_cid}_{metric}_scatter.svg'
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")

print("\nDone!")
