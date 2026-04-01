import os, argparse, time, functools, json, hashlib
import numpy as np, pandas as pd
from math import sqrt
from scipy.special import gammaln
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import ast

print = functools.partial(print, flush=True)

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params or {}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll(y_true, y_pred, eps=1e-8):
    return np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1))


def build_design(X_train, X_val, num_vars, offset_col="log_Exposure"):
    off_tr = X_train[offset_col].to_numpy()
    off_va = X_val[offset_col].to_numpy()
    Xtr = X_train.drop(columns=[offset_col])
    Xva = X_val.drop(columns=[offset_col])

    preproc = ColumnTransformer([("num", Pipeline([("scaler", StandardScaler())]), num_vars)])
    Xtr_z = preproc.fit_transform(Xtr)
    Xva_z = preproc.transform(Xva)

    Xtr_df = pd.DataFrame(Xtr_z, columns=num_vars, index=Xtr.index)
    Xva_df = pd.DataFrame(Xva_z, columns=num_vars, index=Xva.index)
    Xtr_df = sm.add_constant(Xtr_df, has_constant='add')
    Xva_df = sm.add_constant(Xva_df, has_constant='add')
    return Xtr_df, Xva_df, off_tr, off_va

def build_design_with_cats(X_train, X_val, num_vars, cat_vars, offset_col="log_Exposure"):
    # offsets
    off_tr = X_train[offset_col].to_numpy()
    off_va = X_val[offset_col].to_numpy()

    # drop offset from features
    Xtr = X_train.drop(columns=[offset_col])
    Xva = X_val.drop(columns=[offset_col])

    # scale numerics on train only
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preproc = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_vars),
        ('cat', categorical_transformer, cat_vars)
    ])
    Xtr_z = preproc.fit_transform(Xtr)
    Xva_z = preproc.transform(Xva)

    cat_features = preproc.named_transformers_['cat']['onehot'].get_feature_names_out(cat_vars)
    all_columns = np.concatenate([num_vars, cat_features])

    # statsmodels DataFrames (+ intercept)
    Xtr_df = pd.DataFrame(Xtr_z, columns=all_columns, index=Xtr.index)
    Xva_df = pd.DataFrame(Xva_z, columns=all_columns, index=Xva.index)
    Xtr_df = sm.add_constant(Xtr_df, has_constant='add')
    Xva_df = sm.add_constant(Xva_df, has_constant='add')

    return Xtr_df, Xva_df, off_tr, off_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_withfolds_id", required=True,
                    help="CSV with all rows (features, expo, nclaims, and 'fold' column 0..5).")
    ap.add_argument("--outer_fold", type=int, required=True,
                    help="Which fold is OUTER TEST, e.g. 0..5.")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write results for this outer fold.")
    ap.add_argument("--alpha_list", default="1e-4,1e-3,1e-2,1e-1,0.15,0.2,0.25,0.5,1.0")
    ap.add_argument("--l1_wt_list", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--num_vars", type=str, required=True,
                    help="Python-style list of numeric var names (excluding expo).")
    ap.add_argument("--cat_vars", default="None",
                    help="Python-style list of categorical var names, or 'None'.")
    ap.add_argument("--has_osm14", default="no",
                    help="'yes' to auto-drop constant OSM features.")
    ap.add_argument("--fold_col", default="fold",
                    help="Name of the column with 0..5 fold IDs.")
    args = ap.parse_args()

    t0 = time.time()


    # Load data and define outer/inner splits
    df_full = pd.read_csv(args.data_withfolds_id)

    fold_col = args.fold_col
    outer_id = args.outer_fold

    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy()
    df_outer_tr   = df_full[~is_outer_test].copy()

    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())  # should be 5 ids

    # parse feature lists
    base_num_vars = ast.literal_eval(args.num_vars)
    use_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    if use_cats:
        base_cat_vars = ast.literal_eval(args.cat_vars)
    else:
        base_cat_vars = []

    alpha_list = [float(x) for x in args.alpha_list.split(",")]
    l1_wt_list = [float(x) for x in args.l1_wt_list.split(",")]

    # storage
    inner_details_rows = []   # each val_fold, each (alpha,L1_wt)
    inner_summary_rows = []   # mean across val_folds for each (alpha,L1_wt)

    # Inner CV hyperparameter tuning
    for l1_wt in l1_wt_list:
        for alpha in alpha_list:

            fold_val_rmse = []
            fold_val_nll  = []

            for val_fold in inner_fold_ids:
                # masks
                is_val   = (df_outer_tr[fold_col] == val_fold)
                is_train = (df_outer_tr[fold_col] != val_fold)

                df_inner_train = df_outer_tr[is_train].copy()
                df_inner_val   = df_outer_tr[is_val].copy()

                # offsets
                df_inner_train["log_Exposure"] = np.log(df_inner_train["expo"])
                df_inner_val["log_Exposure"]   = np.log(df_inner_val["expo"])

                # local copies of var lists so we don't mutate the global lists
                num_vars = list(base_num_vars)
                cat_vars = list(base_cat_vars)

                # build X/y
                X_train = df_inner_train.drop(columns=["nclaims"]).copy()
                y_train = df_inner_train["nclaims"].copy()

                X_val   = df_inner_val.drop(columns=["nclaims"]).copy()
                y_val   = df_inner_val["nclaims"].copy()

                # reset indices
                for tmp in (X_train, X_val, y_train, y_val):
                    tmp.reset_index(drop=True, inplace=True)

                # design matrices
                if use_cats and len(cat_vars) > 0:
                    Xtr_df, Xva_df, off_tr, off_va = build_design_with_cats(
                        X_train, X_val,
                        num_vars=num_vars,
                        cat_vars=cat_vars,
                        offset_col="log_Exposure"
                    )
                else:
                    Xtr_df, Xva_df, off_tr, off_va = build_design(
                        X_train, X_val,
                        num_vars=num_vars,
                        offset_col="log_Exposure"
                    )

                # fit penalized Poisson GLM with THIS alpha/l1_wt
                param_names = Xtr_df.columns.tolist()
                const_idx = param_names.index('const')

                base_model = sm.GLM(
                    y_train.to_numpy(),
                    Xtr_df,
                    family=sm.families.Poisson(),
                    offset=off_tr
                )

                alpha_vec = np.full(len(param_names), alpha, dtype=float)
                alpha_vec[const_idx] = 0.0  # do not penalize intercept

                try:
                    res = base_model.fit_regularized(
                        method='elastic_net',
                        alpha=alpha_vec,
                        L1_wt=l1_wt,
                        maxiter=1000,
                        cnvrg_tol=1e-6
                    )
                    mu_val = res.predict(Xva_df, offset=off_va)

                    rmse_val = sqrt(mean_squared_error(y_val.to_numpy(), mu_val))
                    nll_val  = poisson_nll(y_val.to_numpy(), mu_val)
                except Exception as e:
                    print(f"[warn] alpha={alpha}, L1_wt={l1_wt}, val_fold={val_fold} failed: {e}")
                    rmse_val = np.nan
                    nll_val  = np.nan

                fold_val_rmse.append(rmse_val)
                fold_val_nll.append(nll_val)

                inner_details_rows.append({
                    "outer_fold": outer_id,
                    "val_fold":   int(val_fold),
                    "alpha":      alpha,
                    "L1_wt":      l1_wt,
                    "rmse_val":   rmse_val,
                    "nll_val":    nll_val,
                    "n_train":    len(df_inner_train),
                    "n_val":      len(df_inner_val),
                })

            # summarize across the 5 inner folds for this hyperparam combo
            inner_summary_rows.append({
                "outer_fold":     outer_id,
                "alpha":          alpha,
                "L1_wt":          l1_wt,
                "rmse_val_mean":  float(np.nanmean(fold_val_rmse)),
                "rmse_val_std":   float(np.nanstd(fold_val_rmse)),
                "nll_val_mean":   float(np.nanmean(fold_val_nll)),
                "nll_val_std":    float(np.nanstd(fold_val_nll)),
            })

    # turn logs into DataFrames
    inner_details_df = pd.DataFrame(inner_details_rows)
    inner_summary_df = pd.DataFrame(inner_summary_rows)
    inner_summary_df=inner_summary_df.sort_values("rmse_val_mean", ascending=True)
    # pick best hyperparameters by lowest mean NLL from inner_summary_df
    best_row = inner_summary_df.sort_values("rmse_val_mean", ascending=True).iloc[0]
    best_alpha = best_row["alpha"]
    best_l1wt  = best_row["L1_wt"]

  
    # Retrain on all 5 inner folds combined,
    # then evaluate on outer test
    df_tr_full  = df_outer_tr.copy()
    df_test_full = df_outer_test.copy()

    df_tr_full["log_Exposure"]   = np.log(df_tr_full["expo"])
    df_test_full["log_Exposure"] = np.log(df_test_full["expo"])

    num_vars_final = list(base_num_vars)
    cat_vars_final = list(base_cat_vars)

    X_train_full = df_tr_full.drop(columns=["nclaims"]).copy()
    y_train_full = df_tr_full["nclaims"].copy()

    X_test_full  = df_test_full.drop(columns=["nclaims"]).copy()
    y_test_true  = df_test_full["nclaims"].copy()

    if use_cats and len(cat_vars_final) > 0:
        Xtr_full_df, Xte_full_df, off_tr_full, off_te_full = build_design_with_cats(
            X_train_full, X_test_full,
            num_vars=num_vars_final,
            cat_vars=cat_vars_final,
            offset_col="log_Exposure"
        )
    else:
        Xtr_full_df, Xte_full_df, off_tr_full, off_te_full = build_design(
            X_train_full, X_test_full,
            num_vars=num_vars_final,
            offset_col="log_Exposure"
        )

    param_names_full = Xtr_full_df.columns.tolist()
    const_idx_full = param_names_full.index('const')

    base_model_full = sm.GLM(
        y_train_full.to_numpy(),
        Xtr_full_df,
        family=sm.families.Poisson(),
        offset=off_tr_full
    )

    alpha_vec_full = np.full(len(param_names_full), best_alpha, dtype=float)
    alpha_vec_full[const_idx_full] = 0.0

    res_full = base_model_full.fit_regularized(
        method='elastic_net',
        alpha=alpha_vec_full,
        L1_wt=best_l1wt,
        maxiter=1000,
        cnvrg_tol=1e-6
    )

    mu_test = res_full.predict(Xte_full_df, offset=off_te_full)

    rmse_test = sqrt(mean_squared_error(y_test_true.to_numpy(), mu_test))
    nll_test  = poisson_nll(y_test_true.to_numpy(), mu_test)

    outer_eval_row = {
        "outer_fold": outer_id,
        "alpha":      best_alpha,
        "L1_wt":      best_l1wt,
        "rmse_test":  rmse_test,
        "nll_test":   nll_test,
        "n_train_rows_outer": len(df_tr_full),
        "n_test_rows_outer":  len(df_test_full),
        "seconds":    time.time() - t0
    }
    outer_eval_df = pd.DataFrame([outer_eval_row])

    # predictions for bootstrap CI later
    outer_preds_df = pd.DataFrame({
        "outer_fold": outer_id,
        "idx":        df_test_full.index,
        "y_true":     y_test_true.to_numpy(),
        "y_pred":     mu_test
    })

    coefs = res_full.params
    mini_summary = pd.DataFrame({
        "coef": coefs,
        "abs_coef": coefs.abs(),
        "non_zero": coefs != 0
    }).sort_values("abs_coef", ascending=False)

    print(mini_summary[mini_summary["non_zero"]])

    # Variables with coefficient == 0
    zero_coefs = coefs[coefs == 0]
    print("Variables with zero coefficients:")
    print(zero_coefs)

    print(f'Mean of the y_test {y_test_true.to_numpy().mean()} and std deviation {y_test_true.to_numpy().std(ddof=1)}')
    print(f'Mean of the y_test_pred {mu_test.mean()}')
  
    # Save everything
    os.makedirs(args.out_dir, exist_ok=True)

    #inner_details_path = os.path.join(args.out_dir, f"inner_details_outer{outer_id}.csv")
    inner_summary_path = os.path.join(args.out_dir, f"inner_summary_outer{outer_id}.csv")
    outer_eval_path    = os.path.join(args.out_dir, f"outer_eval_outer{outer_id}.csv")
    outer_preds_path   = os.path.join(args.out_dir, f"outer_predictions_outer{outer_id}.csv")

    #inner_details_df.to_csv(inner_details_path, index=False)
    inner_summary_df.to_csv(inner_summary_path, index=False)
    outer_eval_df.to_csv(outer_eval_path, index=False)
    outer_preds_df.to_csv(outer_preds_path, index=False)

    print(f"[outer_fold {outer_id}] best alpha={best_alpha}, L1_wt={best_l1wt}")
    print(f"[outer_fold {outer_id}] rmse_test={rmse_test:.6f}, nll_test={nll_test:.6f}")
    print("wrote:")
   # print(" ", inner_details_path)
    print(" ", inner_summary_path)
    print(" ", outer_eval_path)
    print(" ", outer_preds_path)


if __name__ == "__main__":
    main()
