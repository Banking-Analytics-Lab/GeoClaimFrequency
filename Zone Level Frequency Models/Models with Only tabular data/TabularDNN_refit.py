# Refit the tabular-only DNN on outer-train with fixed best hyperparameters, then evaluate on the outer-test.
import os, argparse, time, json, functools
import numpy as np
import pandas as pd
from math import sqrt
from scipy.special import gammaln

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump as joblib_dump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

#  helpers 
def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def poisson_nll_torch(y_true, y_pred, eps=1e-8):
    # y_pred is Poisson mean μ
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps) + torch.lgamma(y_true + 1))

def poisson_nll_numpy(y_true, y_pred, eps=1e-8):
    return float(np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1)))

def preprocess_tabular_data(X_tr, X_te, num_vars, cat_vars=None):
    """
    Fit preprocessor on TRAIN ONLY (outer-train), transform both.
    Returns dense numpy arrays + fitted preprocessor.
    """
    if not cat_vars:
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", StandardScaler())]), num_vars)],
            remainder="drop"
        )
    else:
        categorical = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        pre = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), num_vars),
            ("cat", categorical, cat_vars),
        ], remainder="drop")

    Xtr = pre.fit_transform(X_tr)
    Xte = pre.transform(X_te)
    return Xtr, Xte, pre

def save_learning_curves(tr, va, out_png, title=None):
    fig = plt.figure()
    plt.plot(tr, label="Train PNLL")
    plt.plot(va, label="Test PNLL (tracked)")
    plt.xlabel("Epoch"); plt.ylabel("Poisson NLL"); plt.grid(True); plt.legend()
    if title: plt.title(title)
    fig.savefig(out_png, bbox_inches="tight", dpi=200); plt.close(fig)

#  dataset 
class TabularPoissonDataset(Dataset):
    """
    Returns: tab (tab_dim,), off (scalar log_expo), y (scalar nclaims)
    """
    def __init__(self, tab_array, log_offsets, y):
        self.tab = torch.as_tensor(tab_array, dtype=torch.float32)
        self.off = torch.as_tensor(np.asarray(log_offsets), dtype=torch.float32).view(-1)
        self.y   = torch.as_tensor(np.asarray(y), dtype=torch.float32).view(-1)

    def __len__(self):
        return self.tab.shape[0]

    def __getitem__(self, idx):
        return self.tab[idx], self.off[idx], self.y[idx]

#  model 
class TabularPoissonNet(nn.Module):
    """
    Simple MLP for tabular-only Poisson regression with offset:
      μ = exp( f(tab) + log_offset )
    """
    def __init__(self, tabular_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tabular, log_offset=None):
        eta = self.net(tabular).squeeze(1)  
        if log_offset is not None:
            eta = eta + log_offset
        return torch.exp(eta)  # mu

#  train / eval 
def train_fixed_epochs(model, train_loader, val_loader, optimizer, device, scaler, epochs=50):
    tr_losses, va_losses = [], []
    for ep in range(epochs):
        # train
        model.train()
        tot = 0.0
        for tab, off, y in train_loader:
            tab, off, y = tab.to(device), off.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(tab, off)
                loss = poisson_nll_torch(y, pred)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item() * len(y)
        tr_epoch = tot / len(train_loader.dataset)
        tr_losses.append(tr_epoch)

        # "val" tracked on outer-test to monitor generalization
        model.eval()
        totv = 0.0
        with torch.no_grad():
            for tab, off, y in val_loader:
                tab, off, y = tab.to(device), off.to(device), y.to(device)
                with autocast():
                    pred = model(tab, off)
                    loss = poisson_nll_torch(y, pred)
                totv += loss.item() * len(y)
        va_epoch = totv / len(val_loader.dataset)
        va_losses.append(va_epoch)
        print(f"[Epoch {ep+1}/{epochs}] Train PNLL={tr_epoch:.4f} | Test PNLL={va_epoch:.4f}")
    return tr_losses, va_losses

def eval_loader_metrics(model, data_loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for tab, off, y in data_loader:
            tab, off = tab.to(device), off.to(device)
            with autocast():
                p = model(tab, off)
            preds.append(p.cpu().numpy())
            trues.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    pnll = poisson_nll_numpy(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return pnll, rmse, float(y_pred.mean()), y_true, y_pred

# main
def main():
    ap = argparse.ArgumentParser()

    # data / folds
    ap.add_argument("--data_withfolds_id", required=True,
                    help="CSV with rows incl. expo, nclaims, 'fold' 0..5, and features.")
    ap.add_argument("--outer_fold", type=int, required=True,
                    help="Which fold is OUTER TEST (0..5).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write refit artifacts for this outer fold.")

    # features
    ap.add_argument("--num_vars", type=str, required=True,
                    help="Python-style list of numeric/tabular var names.")
    ap.add_argument("--cat_vars", default="None",
                    help="Python-style list of categorical var names, or 'None'.")

    # optional: drop constant numeric columns (measured on outer-train only)
    ap.add_argument("--drop_constant_numeric", type=str, default="yes",
                    choices=["yes","no"])

    # fixed best hyperparameters
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--optimizer", type=str, required=True, choices=["adam","adamw"])
    ap.add_argument("--weight_decay", type=float, required=True)

    args = ap.parse_args()
    t0 = time.time()
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load data
    df_full = pd.read_csv(args.data_withfolds_id)
    num_vars = eval(args.num_vars)
    use_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    cat_vars = eval(args.cat_vars) if use_cats else []
    fold_col = "fold"
    outer_id = args.outer_fold

    # split outer train/test
    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy()
    df_outer_tr   = df_full[~is_outer_test].copy()

    # offsets
    df_outer_tr["log_Exposure"]   = np.log(df_outer_tr["expo"])
    df_outer_test["log_Exposure"] = np.log(df_outer_test["expo"])

    # target
    y_outer_tr   = df_outer_tr["nclaims"].reset_index(drop=True)
    y_outer_test = df_outer_test["nclaims"].reset_index(drop=True)

    # optionally drop constant NUMERIC columns measured on outer-train only
    dropped_numeric = []
    if args.drop_constant_numeric == "yes":
        for col in list(num_vars):
            if df_outer_tr[col].nunique(dropna=False) <= 1:
                dropped_numeric.append(col)
        if dropped_numeric:
            print(f"Dropping constant numeric columns (outer-train based): {dropped_numeric}")
            num_vars = [c for c in num_vars if c not in dropped_numeric]

    # preprocess on ALL outer-train, transform both
    Xtr_proc, Xte_proc, preproc = preprocess_tabular_data(
        df_outer_tr[num_vars + cat_vars] if use_cats else df_outer_tr[num_vars],
        df_outer_test[num_vars + cat_vars] if use_cats else df_outer_test[num_vars],
        num_vars=num_vars,
        cat_vars=cat_vars if use_cats else None
    )

    # build datasets/loaders
    train_ds = TabularPoissonDataset(
        tab_array=Xtr_proc,
        log_offsets=df_outer_tr["log_Exposure"],
        y=y_outer_tr
    )
    test_ds = TabularPoissonDataset(
        tab_array=Xte_proc,
        log_offsets=df_outer_test["log_Exposure"],
        y=y_outer_test
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=1, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    tab_dim = train_ds.tab.shape[1]
    model = TabularPoissonNet(
        tabular_dim=tab_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    # optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # train on ALL outer-train (track outer-test as "val")
    print(f"[outer_fold {outer_id}] Refit: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
          f"dropout={args.dropout}, opt={args.optimizer}, wd={args.weight_decay}, hd={args.hidden_dim}")
    tr_losses, va_losses = train_fixed_epochs(
        model, train_loader, test_loader, optimizer, device, scaler, epochs=args.epochs
    )

    # evaluate on outer-test
    pnll_te, rmse_te, mean_pred_te, y_true_te, y_pred_te = eval_loader_metrics(model, test_loader, device)

    # ---- saves ----
    # learning curves
    curve_png = os.path.join(args.out_dir, f"learning_curve_outer{outer_id}.png")
    save_learning_curves(tr_losses, va_losses, curve_png, title=f"Outer {outer_id} refit (tabular)")

    # metrics and predictions
    eval_df = pd.DataFrame([{
        "outer_fold": outer_id,
        "rmse_test": rmse_te,
        "pnll_test": pnll_te,
        "mean_pred_test": mean_pred_te,
        "n_train_rows_outer": len(df_outer_tr),
        "n_test_rows_outer": len(df_outer_test),
        "seconds": time.time() - t0,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "dropped_numeric_cols": ";".join(dropped_numeric) if dropped_numeric else ""
    }])
    eval_path = os.path.join(args.out_dir, f"outer_eval_outer{outer_id}.csv")
    eval_df.to_csv(eval_path, index=False)

    preds_df = pd.DataFrame({
        "outer_fold": outer_id,
        "idx": df_outer_test.index.to_numpy(),
        "y_true": y_true_te,
        "y_pred": y_pred_te
    })
    preds_path = os.path.join(args.out_dir, f"outer_predictions_outer{outer_id}.csv")
    preds_df.to_csv(preds_path, index=False)

    # save model + preprocessor + feature names used by preprocessor
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"refit_model_outer{outer_id}.pth"))
    joblib_dump(preproc, os.path.join(args.out_dir, f"preprocessor_outer{outer_id}.joblib"))
    try:
        feat_names = preproc.get_feature_names_out().tolist()
    except Exception:
        feat_names = []
    with open(os.path.join(args.out_dir, f"used_features_outer{outer_id}.json"), "w") as f:
        json.dump({
            "num_vars_after_drop": num_vars,
            "cat_vars": cat_vars,
            "preprocessor_feature_names_out": feat_names
        }, f)

    print(f"[outer_fold {outer_id}] DONE. rmse_test={rmse_te:.6f}, pnll_test={pnll_te:.6f}")
    print("Saved to:")
    print(" ", curve_png)
    print(" ", eval_path)
    print(" ", preds_path)

if __name__ == "__main__":
    main()
