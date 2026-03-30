import re

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

# Load TSV with mmseqs scores
tsv_df = pd.read_csv('/mnt/user-data/uploads/analyze_af3_signaling_with_mmseqs.tsv', sep='\t')

# Extract run_id from run_dir in TSV
def extract_run_id(path):
    match = re.search(r'(run_\d+_\d+)', str(path))
    return match.group(1) if match else None

tsv_df['run_id'] = tsv_df['run_dir'].apply(extract_run_id)

# Load ligand CSVs
ac_atp_df = pd.read_csv('/mnt/user-data/uploads/pocket_ligand_AC_ATP.csv')
b2ar_ale_df = pd.read_csv('/mnt/user-data/uploads/pocket_ligand_b2AR_ALE.csv')

# Extract run_id from design path in CSVs
ac_atp_df['run_id'] = ac_atp_df['design'].apply(extract_run_id)
b2ar_ale_df['run_id'] = b2ar_ale_df['design'].apply(extract_run_id)

# Filter TSV for each complex type
tsv_ac = tsv_df[tsv_df['complex_id'] == 'MONOMER::Adenylyl_cyclase'][['run_id', 'mmseqs_weighted_score']].copy()
tsv_gpcr = tsv_df[tsv_df['complex_id'] == 'MONOMER::GPCR'][['run_id', 'mmseqs_weighted_score']].copy()

# Merge on run_id
merged_ac = ac_atp_df.merge(tsv_ac, on='run_id', how='inner')
merged_gpcr = b2ar_ale_df.merge(tsv_gpcr, on='run_id', how='inner')

print(f"AC_ATP: {len(merged_ac)} matched rows")
print(f"b2AR_ALE: {len(merged_gpcr)} matched rows")

# Define colors
slate_blue = '#6A7FDB'
slate_blue_edge = '#4A5FAB'
pastel_red = '#FF9B9B'
pastel_red_edge = '#E87979'

# Convert mm to inches
width_mm = 27
height_mm = 22
width_in = width_mm * 0.03937
height_in = height_mm * 0.03937

# Native values
native_values = {
    'AC_ATP': {'ligand_rmsd': 3.0, 'mmseqs_weighted_score': 1.0},
    'b2AR_ALE': {'ligand_rmsd': 0.23, 'mmseqs_weighted_score': 1.0}
}

# Plot function
def make_scatter(df, name, native_rmsd):
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Plot design points
    x = df['mmseqs_weighted_score']
    y = df['ligand_rmsd']

    ax.scatter(x, y, c=slate_blue, edgecolors=slate_blue_edge,
               s=15, alpha=0.7, linewidths=0.3, zorder=2)

    # Plot native point
    ax.scatter([1.0], [native_rmsd], c=pastel_red, edgecolors=pastel_red_edge,
               s=25, alpha=0.9, linewidths=0.5, zorder=3, marker='D',
               label='Native')

    # Labels and formatting
    ax.set_xlabel('MMseqs Weighted Score', fontsize=6)
    ax.set_ylabel('Ligand RMSD (Å)', fontsize=6)

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
    filename = f'/home/claude/{name}_ligand_rmsd_scatter.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")

# Generate plots
make_scatter(merged_ac, 'AC_ATP', native_values['AC_ATP']['ligand_rmsd'])
make_scatter(merged_gpcr, 'b2AR_ALE', native_values['b2AR_ALE']['ligand_rmsd'])

print("\nDone!")
