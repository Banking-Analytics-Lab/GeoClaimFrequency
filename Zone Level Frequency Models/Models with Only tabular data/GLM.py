import os, argparse, functools, time, json, hashlib, ast
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import sqrt
from scipy.special import gammaln
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print = functools.partial(print, flush=True)

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params or {}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll(y_true, y_pred, eps=1e-8):
    return np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1))

def run_poisson_glm_with_dummies(
    X_train, y_train,
    X_val, y_val,
    offset_col='log_Exposure',
    num_vars=None,
    model_name="Poisson GLM"
):
    offset_train = X_train[offset_col].values
    offset_val   = X_val[offset_col].values
    X_train = X_train.drop(columns=[offset_col])
    X_val   = X_val.drop(columns=[offset_col])

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_vars)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)

    X_train_df = pd.DataFrame(X_train_proc, columns=num_vars, index=X_train.index)
    X_val_df   = pd.DataFrame(X_val_proc,   columns=num_vars, index=X_val.index)

    X_train_df = sm.add_constant(X_train_df, has_constant='add')
    X_val_df   = sm.add_constant(X_val_df, has_constant='add')

    model = sm.GLM(
        y_train,
        X_train_df,
        family=sm.families.Poisson(),
        offset=offset_train
    ).fit()

    y_val_pred = model.predict(X_val_df, offset=offset_val)
    val_pnll = poisson_nll(y_val, y_val_pred)

    return model, y_val_pred, val_pnll


def run_poisson_glm_with_dummies_with_cats(
    X_train, y_train,
    X_val, y_val,
    offset_col='log_Exposure',
    num_vars=None, cat_vars=None,
    model_name="Poisson GLM"
):
    offset_train = X_train[offset_col].values
    offset_val   = X_val[offset_col].values
    X_train = X_train.drop(columns=[offset_col])
    X_val   = X_val.drop(columns=[offset_col])

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_vars),
        ('cat', categorical_transformer, cat_vars)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)

    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_vars)
    all_columns  = np.concatenate([num_vars, cat_features])

    X_train_df = pd.DataFrame(X_train_proc, columns=all_columns, index=X_train.index)
    X_val_df   = pd.DataFrame(X_val_proc,   columns=all_columns, index=X_val.index)

    X_train_df = sm.add_constant(X_train_df, has_constant='add')
    X_val_df   = sm.add_constant(X_val_df, has_constant='add')

    model = sm.GLM(
        y_train,
        X_train_df,
        family=sm.families.Poisson(),
        offset=offset_train
    ).fit()

    y_val_pred = model.predict(X_val_df, offset=offset_val)
    val_pnll = poisson_nll(y_val, y_val_pred)

    return model, y_val_pred, val_pnll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_withfolds_id", required=True,
                    help="CSV with all rows (features, expo, nclaims, and 'fold' column 0..5).")
    ap.add_argument("--outer_fold", type=int, required=True,
                    help="Which fold is OUTER TEST, e.g. 0..5.")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write results for this outer fold.")
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

    # Load data and create offset
    df_full = pd.read_csv(args.data_withfolds_id).copy()
    df_full["log_Exposure"] = np.log(df_full["expo"])
    df_full["postcode_2"] = df_full["postcode_2"].astype(str) # adding this
    print("Unique postcode2:", df_full["postcode_2"].nunique())
    print("Smallest postcode counts:\n", df_full["postcode_2"].value_counts().tail(10))


    fold_col = args.fold_col
    outer_id = args.outer_fold

    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy().reset_index(drop=True)
    df_outer_tr   = df_full[~is_outer_test].copy().reset_index(drop=True)

    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())

    base_num_vars = ast.literal_eval(args.num_vars)
    use_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    base_cat_vars = ast.literal_eval(args.cat_vars) if use_cats else []


    # Inner CV on df_outer_tr
    cv_rows = []

    for val_fold in inner_fold_ids:
        is_val   = (df_outer_tr[fold_col] == val_fold)
        is_train = (df_outer_tr[fold_col] != val_fold)

        df_inner_train = df_outer_tr[is_train].copy().reset_index(drop=True)
        df_inner_val   = df_outer_tr[is_val].copy().reset_index(drop=True)

        num_vars = list(base_num_vars)
        cat_vars = list(base_cat_vars)

        # build X/y
        y_train = df_inner_train["nclaims"].copy().reset_index(drop=True)
        y_val   = df_inner_val["nclaims"].copy().reset_index(drop=True)

        X_train = df_inner_train.drop(columns=["nclaims"]).copy().reset_index(drop=True)
        X_val   = df_inner_val.drop(columns=["nclaims"]).copy().reset_index(drop=True)

        # columns actually present
        num_vars = [v for v in num_vars if v in X_train.columns]
        cat_vars = [v for v in cat_vars if v in X_train.columns]

        # fit model for this inner fold
        if use_cats and len(cat_vars) > 0:
            model, y_val_pred, val_pnll = run_poisson_glm_with_dummies_with_cats(
                X_train, y_train,
                X_val, y_val,
                offset_col='log_Exposure',
                num_vars=num_vars,
                cat_vars=cat_vars,
                model_name="Poisson GLM"
            )
        else:
            model, y_val_pred, val_pnll = run_poisson_glm_with_dummies(
                X_train, y_train,
                X_val, y_val,
                offset_col='log_Exposure',
                num_vars=num_vars,
                model_name="Poisson GLM"
            )

        val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))

        cv_rows.append({
            "outer_fold": outer_id,
            "val_fold": int(val_fold),
            "val_pnll": val_pnll,
            "val_rmse": val_rmse,
            "n_val": len(y_val)
        })

    # summarize CV after the loop
    cv_df = pd.DataFrame(cv_rows)
    cv_mean_pnll = cv_df["val_pnll"].mean()
    cv_std_pnll  = cv_df["val_pnll"].std(ddof=1)
    cv_mean_rmse = cv_df["val_rmse"].mean()
    cv_std_rmse  = cv_df["val_rmse"].std(ddof=1)


    # Final refit on all df_outer_tr, test on df_outer_test
    num_vars_full = list(base_num_vars)
    cat_vars_full = list(base_cat_vars)

    # ensure offset column exists (it does from earlier) and include it
    keep_cols_full = num_vars_full + cat_vars_full + ["expo", "log_Exposure"]

    df_outer_tr   = df_outer_tr.reset_index(drop=True)
    df_outer_test = df_outer_test.reset_index(drop=True)

    y_tr_full = df_outer_tr["nclaims"].copy().reset_index(drop=True)
    y_test    = df_outer_test["nclaims"].copy().reset_index(drop=True)

    X_tr_full = df_outer_tr[keep_cols_full].copy().reset_index(drop=True)
    X_test    = df_outer_test[keep_cols_full].copy().reset_index(drop=True)

    # train final model on ALL training rows, evaluate on outer test
    if use_cats and len(cat_vars_full) > 0:
        final_model, y_test_pred, test_pnll = run_poisson_glm_with_dummies_with_cats(
            X_tr_full, y_tr_full,
            X_test, y_test,
            offset_col='log_Exposure',
            num_vars=num_vars_full,
            cat_vars=cat_vars_full,
            model_name="FINAL GLM (refit on outer_tr)"
        )
    else:
        final_model, y_test_pred, test_pnll = run_poisson_glm_with_dummies(
            X_tr_full, y_tr_full,
            X_test, y_test,
            offset_col='log_Exposure',
            num_vars=num_vars_full,
            model_name="FINAL GLM (refit on outer_tr)"
        )

    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))


    # Save results
    os.makedirs(args.out_dir, exist_ok=True)

    cv_summary_row = {
        "outer_fold": outer_id,
        "cv_mean_pnll": cv_mean_pnll,
        "cv_std_pnll":  cv_std_pnll,
        "cv_mean_rmse": cv_mean_rmse,
        "cv_std_rmse":  cv_std_rmse,
        "n_inner_folds": len(inner_fold_ids),
        "runtime_sec": time.time() - t0,
        "config_id": stable_config_id({
            "model": "GLM",
            "num_vars": base_num_vars,
            "cat_vars": base_cat_vars,
            "has_osm14": args.has_osm14.lower()
        })
    }

    cv_df_out = cv_df.copy()
    for k,v in cv_summary_row.items():
        cv_df_out[k] = v

    cv_path = os.path.join(args.out_dir, f"cv_metrics_outer{outer_id}.csv")
    cv_df_out.to_csv(cv_path, index=False)

    test_row = pd.DataFrame([{
        "outer_fold": outer_id,
        "test_pnll": test_pnll,
        "test_rmse": test_rmse,
        "n_test": len(y_test),
        "runtime_sec": time.time() - t0,
        "config_id": stable_config_id({
            "model": "GLM",
            "num_vars": num_vars_full,
            "cat_vars": cat_vars_full,
            "has_osm14": args.has_osm14.lower()
        })
    }])

    test_path = os.path.join(args.out_dir, f"outer_test_metrics_outer{outer_id}.csv")
    test_row.to_csv(test_path, index=False)

   
    # Print summary
    print("=== INNER CV SUMMARY (outer train only) ===")
    print(cv_df[["val_fold","val_pnll","val_rmse"]])
    print(f"CV mean PNLL = {cv_mean_pnll:.6f}  (std {cv_std_pnll:.6f})")
    print(f"CV mean RMSE = {cv_mean_rmse:.6f}  (std {cv_std_rmse:.6f})")

    print("\n=== OUTER TEST EVAL ===")
    print(f"Outer fold {outer_id} Test PNLL = {test_pnll:.6f}")
    print(f"Outer fold {outer_id} Test RMSE = {test_rmse:.6f}")
    print(f"Wrote CV metrics to {cv_path}")
    print(f"Wrote outer test metrics to {test_path}")
    print(f"Total runtime {time.time() - t0:.1f} sec")


if __name__ == "__main__":
    main()
