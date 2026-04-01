# DNN (tabular-only) with extended cross validation (one config per job).
# Outputs in --out_dir:
#   inner_details_outer{outer}.csv (one row per inner fold)
#   inner_summary_outer{outer}.csv (one row per config)

import os, argparse, time, json, hashlib, functools, csv, fcntl
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

import matplotlib
matplotlib.use("Agg")

print = functools.partial(print, flush=True)

# helpers 
def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params or {}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll_torch(y_true, y_pred, eps=1e-8):
    # y_pred = Poisson mean μ
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps) + torch.lgamma(y_true + 1))

def poisson_nll_numpy(y_true, y_pred, eps=1e-8):
    return float(np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1)))

def preprocess_tabular_data(X_train, X_val, num_vars, cat_vars=None):
    """
    Fit StandardScaler / OneHot on TRAIN ONLY, transform both.
    Returns dense numpy arrays + fitted preprocessor.
    """
    if not cat_vars:
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", StandardScaler())]), num_vars)],
            remainder="drop"
        )
    else:
        categorical_transformer = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        pre = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), num_vars),
            ("cat", categorical_transformer, cat_vars),
        ], remainder="drop")

    X_tr = pre.fit_transform(X_train)
    X_va = pre.transform(X_val)
    return X_tr, X_va, pre

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
        eta = self.net(tabular).squeeze(1)  # [B]
        if log_offset is not None:
            eta = eta + log_offset
        return torch.exp(eta)  # μ

# train / eval 
def train_fixed_epochs(model, train_loader, val_loader,
                       optimizer, device, scaler, epochs=50):
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

        # val
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
        print(f"[Epoch {ep+1}/{epochs}] Train PNLL={tr_epoch:.4f} | Val PNLL={va_epoch:.4f}")
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

#  CSV helpers 
FIELDNAMES_SUMMARY = [
    "outer_fold","config_id","batch_size","hidden_dim","epochs","lr",
    "dropout","optimizer","weight_decay","seed",
    "rmse_val_mean","rmse_val_std","pnll_val_mean","pnll_val_std",
    "n_inner_folds","timestamp"
]
FIELDNAMES_DETAILS = [
    "outer_fold","config_id","val_fold","rmse_val","pnll_val",
    "n_train","n_val","seed","epochs","batch_size","hidden_dim","lr",
    "dropout","optimizer","weight_decay"
]

def append_result_row(csv_path: str, row: dict, fieldnames: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        write_header = (f.tell() == 0)
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            wr.writeheader()
        wr.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    # data / folds
    ap.add_argument("--data_withfolds_id", required=True,
        help="CSV with rows incl. expo, nclaims, 'fold' 0..5, and features.")
    ap.add_argument("--outer_fold", type=int, required=True,
        help="Which fold is OUTER TEST (0..5).")
    ap.add_argument("--out_dir", required=True,
        help="Directory to write results for this outer fold.")
    # features
    ap.add_argument("--num_vars", type=str, required=True,
        help="Python-style list of numeric/tabular var names.")
    ap.add_argument("--cat_vars", default="None",
        help="Python-style list of categorical var names, or 'None'.")
    # training / hp
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

    # load and split
    df_full = pd.read_csv(args.data_withfolds_id)
    num_vars = eval(args.num_vars)
    use_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    cat_vars = eval(args.cat_vars) if use_cats else []
    fold_col = "fold"
    outer_id = args.outer_fold

    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy()
    df_outer_tr   = df_full[~is_outer_test].copy()

    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())  # expect 5

    # offsets
    df_outer_tr["log_Exposure"]   = np.log(df_outer_tr["expo"])
    df_outer_test["log_Exposure"] = np.log(df_outer_test["expo"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # collect metrics across inner folds
    fold_val_rmse, fold_val_pnll = [], []
    detail_rows_local = []

    for val_fold in inner_fold_ids:
        is_val   = (df_outer_tr[fold_col] == val_fold)
        is_train = (df_outer_tr[fold_col] != val_fold)

        df_tr = df_outer_tr[is_train].copy()
        df_va = df_outer_tr[is_val].copy()

        y_tr = df_tr["nclaims"].reset_index(drop=True)
        y_va = df_va["nclaims"].reset_index(drop=True)

        # tabular preprocess (fit on train only)
        X_tr, X_va, preproc = preprocess_tabular_data(
            df_tr[num_vars + cat_vars] if use_cats else df_tr[num_vars],
            df_va[num_vars + cat_vars] if use_cats else df_va[num_vars],
            num_vars,
            cat_vars=cat_vars if use_cats else None
        )

        # datasets/loaders
        train_ds = TabularPoissonDataset(
            tab_array=X_tr,
            log_offsets=df_tr["log_Exposure"],
            y=y_tr
        )
        val_ds = TabularPoissonDataset(
            tab_array=X_va,
            log_offsets=df_va["log_Exposure"],
            y=y_va
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=1, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=1, pin_memory=True)

        tab_dim = train_ds.tab.shape[1]
        model = TabularPoissonNet(
            tabular_dim=tab_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)

        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
        scaler = GradScaler()

        print(f"\n[outer_fold {outer_id} | inner_val_fold {val_fold}] "
              f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
              f"dropout={args.dropout}, opt={args.optimizer}, wd={args.weight_decay}, "
              f"hidden_dim={args.hidden_dim}")
        print("-"*80)

        # train
        _ = train_fixed_epochs(model, train_loader, val_loader,
                               optimizer, device, scaler, epochs=args.epochs)

        # evaluate inner val
        pnll_val, rmse_val, mean_pred_val, y_true_val, y_pred_val = eval_loader_metrics(
            model, val_loader, device
        )
        fold_val_rmse.append(rmse_val)
        fold_val_pnll.append(pnll_val)

        detail_rows_local.append({
            "outer_fold": outer_id,
            "val_fold": int(val_fold),
            "rmse_val": rmse_val,
            "pnll_val": pnll_val,
            "n_train": len(df_tr),
            "n_val": len(df_va),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay
        })

    # summarize
    rmse_mean = float(np.nanmean(fold_val_rmse))
    rmse_std  = float(np.nanstd(fold_val_rmse))
    pnll_mean = float(np.nanmean(fold_val_pnll))
    pnll_std  = float(np.nanstd(fold_val_pnll))

    cfg = {
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "seed": args.seed
    }
    cfg_id = stable_config_id(cfg)
    timestamp = time.time()

    # write rows
    details_csv_path = os.path.join(args.out_dir, f"inner_details_outer{outer_id}.csv")
    for d in detail_rows_local:
        d["config_id"] = cfg_id
        append_result_row(details_csv_path, d, FIELDNAMES_DETAILS)

    summary_row = {
        "outer_fold": outer_id,
        "config_id": cfg_id,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "rmse_val_mean": rmse_mean,
        "rmse_val_std": rmse_std,
        "pnll_val_mean": pnll_mean,
        "pnll_val_std": pnll_std,
        "n_inner_folds": len(set(inner_fold_ids)),
        "timestamp": timestamp
    }
    summary_csv_path = os.path.join(args.out_dir, f"inner_summary_outer{outer_id}.csv")
    append_result_row(summary_csv_path, summary_row, FIELDNAMES_SUMMARY)

    print(f"[outer_fold {outer_id}] DONE single-config inner CV.")
    print(f"config_id={cfg_id}")
    print(f"rmse_val_mean={rmse_mean:.6f}, pnll_val_mean={pnll_mean:.6f}")
    print(f"Appended to:\n  {summary_csv_path}\n  {details_csv_path}")

if __name__ == "__main__":
    main()

