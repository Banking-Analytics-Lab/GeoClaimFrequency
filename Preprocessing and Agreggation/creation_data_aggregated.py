import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

BelgMTPL_fil = pd.read_csv('C:\\Users\\salfo\\OneDrive - The University of Western Ontario\\PHD\\GNNActuary\\MTPL\\Belgian\\Preprocessing\\beMTPL97_filtered.csv', index_col=0)

print(BelgMTPL_fil.info())

print(BelgMTPL_fil.columns)

num_predictors = ['ageph', 'bm', 'power', 'agec']
categ_predictors = ['coverage', 'sex', 'fuel', 'use', 'fleet']
sum_predictors = ['expo', 'nclaims']

print(f'numeric predictors:{num_predictors}')
print(f'categorical predictors:{categ_predictors}')
print(f'Predictors to sum:{sum_predictors}')

for i in categ_predictors:
    print(BelgMTPL_fil[i].value_counts())

print('Making a copy of the data to aggregate the values')
BeMTPL_Agg = BelgMTPL_fil.copy()
 
### Aggregate the data 
def aggregate_by_zone(df, num_predictors, cate_predictors, group_col='postcode', drop_first=True):
    df = df.copy()
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=cate_predictors, drop_first=False)
    # Identify one-hot encoded columns for each categorical variable
    onehot_cols = []
    cat_col_map = {}
    for col in cate_predictors:
        cols = [c for c in df_encoded.columns if c.startswith(col + '_')] # identify the name of the columns
        cat_col_map[col] = sorted(cols)
        onehot_cols.extend(cols)

    # Drop one level from each categorical variable (if drop_first)
    if drop_first:
        for col, cols in cat_col_map.items():
            if len(cols) > 1:
                one_to_drop = cols[-1]  # Drop last one alphabetically
                onehot_cols.remove(one_to_drop)

    # Prepare aggregation dictionary for numeric variables
    agg_funcs = ['mean', 'median', 'std']
    agg_dict = {}

    for col in num_predictors:
        for func in agg_funcs:
            agg_dict[col + f'_{func}'] = (col, func)

    # Add categorical proportions (mean of dummies)
    for col in onehot_cols:
        agg_dict[col + '_prop'] = (col, 'mean')

    # Add target and exposure
    if 'nclaims' in df_encoded.columns:
        agg_dict['nclaims'] = ('nclaims', 'sum')
    if 'expo' in df_encoded.columns:
        agg_dict['expo'] = ('expo', 'sum')

    # Apply aggregation using named aggregation
    df_agg = df_encoded.groupby(group_col).agg(**agg_dict).reset_index()

    return df_agg

print(f'Creating the aggregation of the columns by postcode')
BeMTPL_Agg = aggregate_by_zone(BelgMTPL_fil, num_predictors, categ_predictors, group_col=['postcode', 'lat', 'long'], drop_first=True)

print(f'Printing the info() for the created data set')
print(BeMTPL_Agg.info())


df_check = BelgMTPL_fil.groupby('postcode')[['lat', 'long']].nunique()

# Flag cases where either lat or lon has more than one unique value
violations = df_check[(df_check['lat'] > 1) | (df_check['long'] > 1)]

# Output result
if violations.empty:
    print("Each postcode maps to exactly one (lat, long). You can safely group by either.")
else:
    print("Warning: Some postcodes have multiple lat/lon combinations:")
    print(violations)

df_check.shape

#define the variable postcode_2 as the first two digits of the postcode and look if many rows share the same postcode_2
print(f'Defining the postcode2 variable which is the first 2 digits of the postcode')
BeMTPL_Agg['postcode_2'] = BeMTPL_Agg['postcode'].astype(str).str[:2] ### To see 
print(f'Print the unique values of the postcode2')
print(BeMTPL_Agg['postcode_2'].nunique())

### Look the aggregate values for the numeric predictors
print(BeMTPL_Agg.columns)
BeMTPL_Agg.to_csv('BeMTPL_Agg.csv', index=True)
print('Script has finished!')
