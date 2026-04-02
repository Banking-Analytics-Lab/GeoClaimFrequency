import os, argparse, time, json, hashlib, functools, csv, fcntl
import numpy as np
import pandas as pd
from math import sqrt
from scipy.special import gammaln

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

# Utility helpers

def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll_torch(y_true, y_pred, eps=1e-8):
    # (E[Poisson]==y_pred is model output)
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps)+ torch.lgamma(y_true + 1))


def poisson_nll_numpy(y_true, y_pred, eps=1e-8):
    # numpy arrays
    return float(np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1)))


def preprocess_tabular_data(X_train, X_val, num_vars, cat_vars=None):
    """
    Fit StandardScaler on numeric columns of X_train,
    transform both X_train and X_val,
    return arrays + the fitted preprocessor.
    """
    if cat_vars is None or len(cat_vars) == 0:
        # numeric-only case
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", StandardScaler())]), num_vars)],
            remainder="drop")
        X_tr = pre.fit_transform(X_train)
        X_va = pre.transform(X_val)
        return X_tr, X_va, pre

    # numeric + categorical
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), num_vars),
            ("cat", categorical_transformer, cat_vars),
        ],remainder="drop")
    X_tr = pre.fit_transform(X_train)
    X_va = pre.transform(X_val)
    return X_tr, X_va, pre

def empty_tabular_like(n_rows: int):
    """Return a (n_rows x 0) numpy array for images-only mode."""
    return np.zeros((n_rows, 0), dtype=np.float32)

def compute_gray_stats(df, unique_loc, img_root, radius_km, limit=2000):
    """
    Estimate mean/std of grayscale pixel intensities on TRAIN rows only,
    so we can normalize consistently.
    """
    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    vals = []
    for i, row in enumerate(df.itertuples(index=False)):
        if i >= limit:
            break
        postcode = getattr(row, "postcode")

        lat = unique_loc.at[postcode, "lat"]
        lon = unique_loc.at[postcode, "long"]

        p = os.path.join(
            img_root,
            f"squares_R{radius_km}km",
            f"Orth95_{lat}_{lon}_R{radius_km}.jpg"
        )
        try:
            x = tfm(Image.open(p).convert("L")).view(-1)
            vals.append(x)
        except Exception:
            pass

    if not vals:
        return 0.5, 0.5  # fallback
    x = torch.cat(vals)
    return float(x.mean()), float(x.std(unbiased=False))


class ClaimMultimodalDataset(Dataset):
    """
    Returns:
      img (C,H,W), tab (tab_dim,), off (scalar log_expo), y (scalar nclaims)
    """
    def __init__(self, df, y, unique_loc, img_root, radius_km, img_tfm, tab_array, log_offsets):
        self.df = df.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.unique_loc = unique_loc
        self.img_root = img_root
        self.radius_km = radius_km
        self.tfm = img_tfm
        self.tab_array = tab_array
        self.log_offsets = log_offsets.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        postcode = self.df.iloc[idx]['postcode']
        lat = self.unique_loc.at[postcode, "lat"]
        lon = self.unique_loc.at[postcode, "long"]

        p = os.path.join(
            self.img_root,
            f"squares_R{self.radius_km}km",
            f"Orth95_{lat}_{lon}_R{self.radius_km}.jpg"
        )
        img = self.tfm(Image.open(p).convert("L"))

        tab = torch.tensor(self.tab_array[idx], dtype=torch.float32)
        y   = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        off = torch.tensor(self.log_offsets.iloc[idx], dtype=torch.float32)
        return img, tab, off, y


class MultimodalPoissonNet(nn.Module):
    """
    Image encoder: ResNet18 up to fc (Identity head), adapted to 1 channel.
    Fusion head: concat(image_features, tabular) to MLP and then linear predictor.
    Output is exp(eta + log_offset), which is Poisson mean.
    """
    def __init__(self, tabular_dim, device, weights_path,
                 image_embed_dim=512, hidden_dim=128, dropout=0.1):
        super().__init__()

        base_model = resnet18(weights=None)

        # load pretrained weights 
        state_dict = torch.load(weights_path, map_location=device)
        base_model.load_state_dict(state_dict)

        # adapt first conv from 3ch->1ch by averaging
        with torch.no_grad():
            old = base_model.conv1  # [64,3,7,7]
            new = nn.Conv2d(
                in_channels=1,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False
            )
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        base_model.conv1 = new

        base_model.fc = nn.Identity()  # take penultimate features
        self.image_backbone = base_model

        self.fc = nn.Sequential(
            nn.Linear(tabular_dim + image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, tabular, log_offset=None):
        img_feat = self.image_backbone(image)  # [B,512]
        x = torch.cat([img_feat, tabular], dim=1)
        eta = self.fc(x).squeeze(1)  # [B]
        if log_offset is not None:
            eta = eta + log_offset
        return torch.exp(eta)  # Poisson mean μ = exp(eta + log_offset)


def train_fixed_epochs(model, train_loader, val_loader,
                       optimizer, device, scaler, epochs=50):
    """
    val_loader is saved here only so we can track val loss per epoch
    """
    tr_losses, va_losses = [], []
    for ep in range(epochs):
        # ---------------- train ----------------
        model.train()
        tot = 0.0
        for img, tab, off, y in train_loader:
            img, tab, off, y = img.to(device), tab.to(device), off.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(img, tab, off)
                loss = poisson_nll_torch(y, pred)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot += loss.item() * len(y)

        tr_epoch = tot / len(train_loader.dataset)
        tr_losses.append(tr_epoch)

        # ---------------- val ----------------
        model.eval()
        totv = 0.0
        with torch.no_grad():
            for img, tab, off, y in val_loader:
                img, tab, off, y = img.to(device), tab.to(device), off.to(device), y.to(device)
                with autocast():
                    pred = model(img, tab, off)
                    loss = poisson_nll_torch(y, pred)
                totv += loss.item() * len(y)

        va_epoch = totv / len(val_loader.dataset)
        va_losses.append(va_epoch)

        print(f"[Epoch {ep+1}/{epochs}] Train PNLL={tr_epoch:.4f} | Val PNLL={va_epoch:.4f}")

    return tr_losses, va_losses


def eval_loader_metrics(model, data_loader, device):
    """
    Run model on a loader, return:
      pnll, rmse, mean_pred, y_true[], y_pred[]
    """
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for img, tab, off, y in data_loader:
            img, tab, off = img.to(device), tab.to(device), off.to(device)
            with autocast():
                p = model(img, tab, off)
            preds_list.append(p.cpu().numpy())
            trues_list.append(y.numpy())

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(trues_list)

    pnll = poisson_nll_numpy(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mean_pred = float(y_pred.mean())
    return pnll, rmse, mean_pred, y_true, y_pred


# Safe append to shared CSV
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

    # data / structure
    ap.add_argument("--data_withfolds_id", required=True,
        help="CSV with all rows: features incl. expo, postcode, nclaims, and 'fold' column 0..5.")
    ap.add_argument("--unique_loc_csv", required=True,
        help="CSV with columns [postcode,lat,long]. Will be set_index('postcode').")
    ap.add_argument("--outer_fold", type=int, required=True,
        help="Which fold is OUTER TEST, e.g. 0..5.")
    ap.add_argument("--out_dir", required=True,
        help="Directory to write results for this outer fold.")
    ap.add_argument("--img_root", required=True,
        help="Root dir containing squares_R{radius}km/Orth95_{lat}_{lon}_R{radius}.jpg.")
    ap.add_argument("--radius_km", required=True,
        help="Radius string used in image filename, e.g. '0.5','1','3','5'.")

    # tabular vars / training setup
    # in main() ArgParser
    ap.add_argument("--images_only", action="store_true", help="If set, ignore tabular vars and train on images only.")

    ap.add_argument("--num_vars", type=str, required=True,
        help="Python-style list of numeric/tabular var names given to the network.")
    ap.add_argument("--cat_vars", default="None", help="Python-style list of categorical var names, or 'None'.")
    ap.add_argument("--seed", type=int, default=42,
        help="Global random seed (single-seed protocol).")
    ap.add_argument("--weights_path", required=True,
        help="Path to resnet18_weights.pth (pretrained weights already saved).")

    # hyperparameter grid
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--optimizer", type=str, required=True,
                    choices=["adam","adamw"])
    ap.add_argument("--weight_decay", type=float, required=True)

    args = ap.parse_args()
    t0 = time.time()

    # reproducibility / deterministic seed
    set_global_seed(args.seed)

    # prep output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # load full dataset with folds
    df_full = pd.read_csv(args.data_withfolds_id)
    num_vars = eval(args.num_vars)  # parse the python-style list string
    use_cats = (args.cat_vars is not None) and (args.cat_vars != "None")
    if use_cats:
        cat_vars = eval(args.cat_vars)
    else:
        cat_vars = []
    fold_col = "fold"
    outer_id = args.outer_fold

    # define outer splits
    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy()
    df_outer_tr   = df_full[~is_outer_test].copy()

    # inner folds are whatever fold IDs remain in df_outer_tr
    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())  # expect 5 IDs

    # log_Exposure
    df_outer_tr["log_Exposure"]   = np.log(df_outer_tr["expo"])
    df_outer_test["log_Exposure"] = np.log(df_outer_test["expo"])

    # postcode -> lat/long lookup
    unique_loc = pd.read_csv(args.unique_loc_csv).set_index("postcode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # metrics across inner folds for the given config
    fold_val_rmse = []
    fold_val_pnll = []
    detail_rows_local = []

    for val_fold in inner_fold_ids:
        is_val   = (df_outer_tr[fold_col] == val_fold)
        is_train = (df_outer_tr[fold_col] != val_fold)

        df_inner_train = df_outer_tr[is_train].copy()
        df_inner_val   = df_outer_tr[is_val].copy()

        y_train = df_inner_train["nclaims"].reset_index(drop=True)
        y_val   = df_inner_val["nclaims"].reset_index(drop=True)

        # tabular preprocess df_inner_train is the only fitted
        if args.images_only:
            # No tabular features at all
            X_tr_proc = empty_tabular_like(len(df_inner_train))
            X_va_proc = empty_tabular_like(len(df_inner_val))
            tab_dim = 0
        else:
            # numeric/categorical 
            X_tr_proc, X_va_proc, preproc = preprocess_tabular_data(
                df_inner_train[num_vars + cat_vars] if use_cats else df_inner_train[num_vars],
                df_inner_val[num_vars + cat_vars]   if use_cats else df_inner_val[num_vars],
                num_vars,
                cat_vars=cat_vars if use_cats else None
            )
            tab_dim = X_tr_proc.shape[1]

        # grayscale stats from df_inner_train only
        gray_mean, gray_std = compute_gray_stats(
            df_inner_train,
            unique_loc,
            args.img_root,
            args.radius_km
        )

        tfm_train = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[gray_mean], std=[gray_std])
        ])
        tfm_eval = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[gray_mean], std=[gray_std])
        ])

        train_ds = ClaimMultimodalDataset(
            df_inner_train,
            y_train,
            unique_loc,
            args.img_root,
            args.radius_km,
            tfm_train,
            X_tr_proc,
            df_inner_train["log_Exposure"]
        )
        val_ds = ClaimMultimodalDataset(
            df_inner_val,
            y_val,
            unique_loc,
            args.img_root,
            args.radius_km,
            tfm_eval,
            X_va_proc,
            df_inner_val["log_Exposure"]
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        # build model
        #tabular_dim = train_ds.tab_array.shape[1]
        tabular_dim = X_tr_proc.shape[1]  # 0 in images-only mode
        model = MultimodalPoissonNet(
            tabular_dim=tabular_dim,
            device=device,
            weights_path=args.weights_path,
            image_embed_dim=512,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)

        # optimizer
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        scaler = GradScaler()

        print(f"\n[outer_fold {outer_id} | inner_val_fold {val_fold}] "
              f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
              f"dropout={args.dropout}, opt={args.optimizer}, wd={args.weight_decay}, "
              f"hidden_dim={args.hidden_dim}")
        print("-"*80)

        # train
        tr_losses, va_losses = train_fixed_epochs(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            scaler,
            epochs=args.epochs
        )

        # evaluate this inner val fold
        pnll_val, rmse_val, mean_pred_val, y_true_val, y_pred_val = eval_loader_metrics(
            model,
            val_loader,
            device
        )

        fold_val_rmse.append(rmse_val)
        fold_val_pnll.append(pnll_val)

        detail_rows_local.append({
            "outer_fold": outer_id,
            "val_fold": int(val_fold),
            "rmse_val": rmse_val,
            "pnll_val": pnll_val,
            "n_train": len(df_inner_train),
            "n_val": len(df_inner_val),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay
        })

    # summarize this config across inner folds
    rmse_mean = float(np.nanmean(fold_val_rmse))
    rmse_std  = float(np.nanstd(fold_val_rmse))
    pnll_mean = float(np.nanmean(fold_val_pnll))
    pnll_std  = float(np.nanstd(fold_val_pnll))

    # build config dict (for stable id)
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

    # write per-fold detail rows
    details_csv_path = os.path.join(args.out_dir, f"inner_details_outer{outer_id}.csv")
    for drow in detail_rows_local:
        drow["config_id"] = cfg_id
        append_result_row(details_csv_path, drow, FIELDNAMES_DETAILS)

    # write summary row for this config
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
        "n_inner_folds": len(inner_fold_ids),
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
