import os, argparse, time, functools, json, hashlib, ast
import numpy as np
import pandas as pd
from math import sqrt
from itertools import product
from scipy.special import gammaln
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

print = functools.partial(print, flush=True)

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params or {}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll(y_true, y_pred, eps=1e-12):
    # mean negative log likelihood under Poisson
    return np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1))

# Custom regressor wrapper with offset
class XGBPoissonRegressorWithOffset(BaseEstimator, RegressorMixin):
    def __init__(self,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 offset_index=-1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.offset_index = offset_index
        self.model = None

    def fit(self, X, y):
        X = np.array(X)
        offset = X[:, self.offset_index]
        X_core = np.delete(X, self.offset_index, axis=1)

        dtrain = xgb.DMatrix(X_core, label=y)
        dtrain.set_base_margin(offset)

        params = {
            "objective": "count:poisson",
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "eval_metric": "poisson-nloglik",
            "verbosity": 0,
        }
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators
        )
        return self

    def predict(self, X):
        X = np.array(X)
        offset = X[:, self.offset_index]
        X_core = np.delete(X, self.offset_index, axis=1)

        dtest = xgb.DMatrix(X_core)
        dtest.set_base_margin(offset)
        return self.model.predict(dtest)


# helper to build processed matrices for one train/val split
def build_fold_design(
    df_tr_split,
    df_val_split,
    num_vars,
    cat_vars,
    has_cats,
    drop_constant="no"
):
    """
    Returns:
        Xtr_proc, Xva_proc, y_tr, y_val
    where X*_proc are numpy arrays whose LAST column is the offset.
    """
    # Copy so we don't mutate caller
    df_tr = df_tr_split.copy().reset_index(drop=True)
    df_va = df_val_split.copy().reset_index(drop=True)

    # add offset
    df_tr["log_Exposure"] = np.log(np.clip(df_tr["expo"].to_numpy(), 1e-12, None))
    df_va["log_Exposure"] = np.log(np.clip(df_va["expo"].to_numpy(), 1e-12, None))

    # separate y
    y_tr = df_tr["nclaims"].to_numpy()
    y_va = df_va["nclaims"].to_numpy()

    # keep only model inputs + offset
    # We include offset column 'log_Exposure' explicitly.
    base_cols = num_vars + (cat_vars if has_cats else []) + ["log_Exposure"]
    X_tr_raw = df_tr[base_cols].copy()
    X_va_raw = df_va[base_cols].copy()

    # ensure lists match columns actually present after const-drop
    num_vars = [v for v in num_vars if v in X_tr_raw.columns]
    cat_vars = [v for v in cat_vars if v in X_tr_raw.columns]

    # scale nums, onehot cats, passthrough offset as last column
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    if has_cats and len(cat_vars) > 0:
        categorical_transformer = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_vars),
            ("cat", categorical_transformer, cat_vars),
            ("offset", "passthrough", ["log_Exposure"])
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_vars),
            ("offset", "passthrough", ["log_Exposure"])
        ])

    X_tr_proc = preprocessor.fit_transform(X_tr_raw)
    X_va_proc = preprocessor.transform(X_va_raw)

    # by construction, offset is the LAST column in both arrays,
    # because we put ("offset", ...) last in ColumnTransformer.
    offset_index = X_tr_proc.shape[1] - 1

    return X_tr_proc, X_va_proc, y_tr, y_va, offset_index, num_vars, cat_vars



# helper to build final train/test design for refit
def build_final_design(
    df_tr_full,
    df_te_full,
    num_vars,
    cat_vars,
    has_cats,
    drop_constant="no"
):
    df_tr = df_tr_full.copy().reset_index(drop=True)
    df_te = df_te_full.copy().reset_index(drop=True)

    df_tr["log_Exposure"] = np.log(np.clip(df_tr["expo"].to_numpy(), 1e-12, None))
    df_te["log_Exposure"] = np.log(np.clip(df_te["expo"].to_numpy(), 1e-12, None))

    y_tr = df_tr["nclaims"].to_numpy()
    y_te = df_te["nclaims"].to_numpy()

    base_cols = num_vars + (cat_vars if has_cats else []) + ["log_Exposure"]
    X_tr_raw = df_tr[base_cols].copy()
    X_te_raw = df_te[base_cols].copy()

    # safety filter
    num_vars = [v for v in num_vars if v in X_tr_raw.columns]
    cat_vars = [v for v in cat_vars if v in X_tr_raw.columns]

    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    if has_cats and len(cat_vars) > 0:
        categorical_transformer = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_vars),
            ("cat", categorical_transformer, cat_vars),
            ("offset", "passthrough", ["log_Exposure"])
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_vars),
            ("offset", "passthrough", ["log_Exposure"])
        ])

    X_tr_proc = preprocessor.fit_transform(X_tr_raw)
    X_te_proc = preprocessor.transform(X_te_raw)

    offset_index = X_tr_proc.shape[1] - 1

    return X_tr_proc, X_te_proc, y_tr, y_te, offset_index, num_vars, cat_vars


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
                    help="Name of the column with fold IDs.")
    args = ap.parse_args()

    t0 = time.time()

    # Load full dataset
    df_full = pd.read_csv(args.data_withfolds_id).copy()

    fold_col = args.fold_col
    outer_id = args.outer_fold

    # outer split
    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy().reset_index(drop=True)
    df_outer_tr   = df_full[~is_outer_test].copy().reset_index(drop=True)

    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())

    # parse feature lists
    base_num_vars = ast.literal_eval(args.num_vars)
    has_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    base_cat_vars = ast.literal_eval(args.cat_vars) if has_cats else []

    # Hyperparameter grid (tune via inner CV)
    max_depth_grid     = [2, 3, 4]
    learning_rate_grid = [0.2, 0.15, 0.1, 0.05]
    n_estimators_grid  = [75, 100, 125, 150]
    reg_alpha_grid     = [0.0, 0.05, 0.1, 0.2]
    reg_lambda_grid    = [10, 20, 30, 50]

    combos = list(product(
        max_depth_grid,
        learning_rate_grid,
        n_estimators_grid,
        reg_alpha_grid,
        reg_lambda_grid
    ))

    inner_details_rows = []
    inner_summary_rows = []

    # For each hyperparam combo, perform inner CV across inner_fold_ids
    for (md, lr, ne, ra, rl) in combos:
        fold_val_pnll = []
        fold_val_rmse = []

        for val_fold in inner_fold_ids:
            # inner split: val_fold is validation, rest is train
            mask_val   = (df_outer_tr[fold_col] == val_fold)
            mask_train = ~mask_val

            df_inner_tr  = df_outer_tr[mask_train]
            df_inner_val = df_outer_tr[mask_val]

            # build preprocessed matrices for this split
            Xtr_proc, Xva_proc, y_tr, y_va, offset_idx, used_num_vars, used_cat_vars = build_fold_design(
                df_inner_tr,
                df_inner_val,
                num_vars=list(base_num_vars),
                cat_vars=list(base_cat_vars),
                has_cats=has_cats,
                drop_constant=args.has_osm14
            )

            # fit model with these hyperparams
            model = XGBPoissonRegressorWithOffset(
                max_depth=md,
                learning_rate=lr,
                n_estimators=ne,
                reg_alpha=ra,
                reg_lambda=rl,
                offset_index=offset_idx
            )

            try:
                model.fit(Xtr_proc, y_tr)
                y_pred = model.predict(Xva_proc)

                pnll = poisson_nll(y_va, y_pred)
                rmse = sqrt(mean_squared_error(y_va, y_pred))

            except Exception as e:
                print(f"[warn] combo md={md},lr={lr},ne={ne},ra={ra},rl={rl}, val_fold={val_fold} failed: {e}")
                pnll = np.nan
                rmse = np.nan

            fold_val_pnll.append(pnll)
            fold_val_rmse.append(rmse)

            inner_details_rows.append({
                "outer_fold": outer_id,
                "val_fold": int(val_fold),
                "max_depth": md,
                "learning_rate": lr,
                "n_estimators": ne,
                "reg_alpha": ra,
                "reg_lambda": rl,
                "val_pnll": float(pnll),
                "val_rmse": float(rmse),
                "n_train": len(df_inner_tr),
                "n_val": len(df_inner_val)
            })

        # summarize this hyperparam combo over all inner folds
        inner_summary_rows.append({
            "outer_fold": outer_id,
            "max_depth": md,
            "learning_rate": lr,
            "n_estimators": ne,
            "reg_alpha": ra,
            "reg_lambda": rl,
            "pnll_mean": float(np.nanmean(fold_val_pnll)),
            "pnll_std":  float(np.nanstd(fold_val_pnll)),
            "rmse_mean": float(np.nanmean(fold_val_rmse)),
            "rmse_std":  float(np.nanstd(fold_val_rmse)),
            "config_id": stable_config_id({
                "max_depth": md,
                "learning_rate": lr,
                "n_estimators": ne,
                "reg_alpha": ra,
                "reg_lambda": rl
            })
        })

    inner_details_df = pd.DataFrame(inner_details_rows)
    inner_summary_df = pd.DataFrame(inner_summary_rows)

    # sort by mean RMSE ascending (best = smallest cv RMSE)
    inner_summary_df = inner_summary_df.sort_values("rmse_mean", ascending=True).reset_index(drop=True)

    # pick best hyperparams from the top row (lowest rmse_mean)
    best_row = inner_summary_df.iloc[0]
    best_md  = best_row["max_depth"]
    best_lr  = best_row["learning_rate"]
    best_ne  = best_row["n_estimators"]
    best_ra  = best_row["reg_alpha"]
    best_rl  = best_row["reg_lambda"]
  
    # Final refit on ALL df_outer_tr, evaluate on df_outer_test
    Xtr_full_proc, Xte_full_proc, y_tr_full, y_te, offset_idx_full, used_num_final, used_cat_final = build_final_design(
        df_outer_tr,
        df_outer_test,
        num_vars=list(base_num_vars),
        cat_vars=list(base_cat_vars),
        has_cats=has_cats,
        drop_constant=args.has_osm14
    )

    final_model = XGBPoissonRegressorWithOffset(
        max_depth=int(best_md),
        learning_rate=float(best_lr),
        n_estimators=int(best_ne),
        reg_alpha=float(best_ra),
        reg_lambda=float(best_rl),
        offset_index=offset_idx_full
    )
    final_model.fit(Xtr_full_proc, y_tr_full)
    y_pred_test = final_model.predict(Xte_full_proc)

    test_pnll = poisson_nll(y_te, y_pred_test)
    test_rmse = sqrt(mean_squared_error(y_te, y_pred_test))

    outer_eval_df = pd.DataFrame([{
        "outer_fold": outer_id,
        "max_depth": best_md,
        "learning_rate": best_lr,
        "n_estimators": best_ne,
        "reg_alpha": best_ra,
        "reg_lambda": best_rl,
        "test_pnll": float(test_pnll),
        "test_rmse": float(test_rmse),
        "n_train_rows_outer": len(df_outer_tr),
        "n_test_rows_outer": len(df_outer_test),
        "runtime_sec": time.time() - t0,
        "config_id": best_row["config_id"]
    }])

    outer_preds_df = pd.DataFrame({
        "outer_fold": outer_id,
        "row_idx": df_outer_test.index,
        "y_true": y_te,
        "y_pred": y_pred_test
    })

  
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    inner_details_path = os.path.join(args.out_dir, f"inner_details_outer{outer_id}.csv")
    inner_summary_path = os.path.join(args.out_dir, f"inner_summary_outer{outer_id}.csv")
    outer_eval_path    = os.path.join(args.out_dir, f"outer_eval_outer{outer_id}.csv")
    outer_preds_path   = os.path.join(args.out_dir, f"outer_predictions_outer{outer_id}.csv")

    inner_details_df.to_csv(inner_details_path, index=False)
    inner_summary_df.to_csv(inner_summary_path, index=False)
    outer_eval_df.to_csv(outer_eval_path, index=False)
    outer_preds_df.to_csv(outer_preds_path, index=False)

    # Print final summary
    print(f"[outer_fold {outer_id}] best md={best_md}, lr={best_lr}, ne={best_ne}, ra={best_ra}, rl={best_rl}")
    print(f"[outer_fold {outer_id}] test_pnll={test_pnll:.6f}, test_rmse={test_rmse:.6f}")
    print("wrote:")
    print(" ", inner_details_path)
    print(" ", inner_summary_path)
    print(" ", outer_eval_path)
    print(" ", outer_preds_path)
    print(f"runtime_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
