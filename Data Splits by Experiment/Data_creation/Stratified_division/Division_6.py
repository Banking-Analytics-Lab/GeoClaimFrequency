import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(
    '/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/With_osm14_corine2000/all/Data_all_osm14_clc2000_ALL.csv'
)

# Target
y = df['nclaims']

# -------------------------------------------------
# 1. Create quantile bins for stratification
#    - q=10 gives deciles of nclaims
# -------------------------------------------------
n_bins = 10
df['nclaims_bin'] = pd.qcut(y, q=n_bins, duplicates='drop')
df['nclaims_bin_code'] = df['nclaims_bin'].cat.codes
# -------------------------------------------------
# 2. Create 6 disjoint stratified folds
# -------------------------------------------------
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=611)

df['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(X=np.zeros(len(df)), y=df['nclaims_bin_code'])):
    df.loc[val_idx, 'fold'] = fold

# Sanity check: did every row get a fold?
assert (df['fold'] >= 0).all(), "Some rows did not get assigned to a fold."

# Optional: inspect fold sizes
print("Fold sizes:")
print(df['fold'].value_counts().sort_index())
# save all the data set with the number of the folds
df.to_csv("alldata_with_fold_id.csv")
# -------------------------------------------------
# 3. Plot distributions across folds for numeric features
# -------------------------------------------------

# choose numeric columns
numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()

# helper/diagnostic columns we DON'T want to judge model by
helper_cols = ['fold', 'nclaims_bin']
plot_feats = [f for f in numeric_feats if f not in helper_cols]

# make an output directory for plots
plot_dir = "fold_diagnostics_plots"
os.makedirs(plot_dir, exist_ok=True)

for feat in plot_feats:
    plt.figure(figsize=(7,4))
    sns.boxplot(x='fold', y=feat,hue='fold', data=df, palette='Set3')
    plt.title(f'{feat} distribution across 6 stratified folds')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{feat}_by_fold.jpg'), dpi=300)
    plt.close()  # close instead of show so we don't spam windows

# -------------------------------------------------
# 4. Numeric summary table by fold
# -------------------------------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['fold', 'nclaims_bin']]

# Compute per-fold summary statistics
summary_all = (
    df.groupby('fold')[numeric_cols]
      .agg(['mean', 'std', 'min', 'max'])
)

# Display rounded summary
print(summary_all.round(3))

# Optional: save to CSV for inspection
summary_all.to_csv("fold_balance_summary_all_numeric.csv")

# -------------------------------------------------
# 5. Save each fold subset D1...D6
# -------------------------------------------------
for fold in range(6):
    fold_df = df[df.fold == fold].drop(columns=['nclaims_bin'])  # keep it clean
    fold_df.to_csv(f'D{fold+1}.csv', index=False)
