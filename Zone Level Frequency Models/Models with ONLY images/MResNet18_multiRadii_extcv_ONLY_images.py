# Extended cross-validation (one config per job) using multiple image radii per postcode.
# Uses Poisson NLL loss for training and RMSE for evaluation 
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
from PIL import Image

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

def stable_config_id(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def poisson_nll_torch(y_true, y_pred, eps=1e-8):
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps) + torch.lgamma(y_true + 1))

def poisson_nll_numpy(y_true, y_pred, eps=1e-8):
    return float(np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1)))

def compute_gray_stats_all_radii(df, unique_loc, img_root, radii_list, limit_per_radius=2000):
    tfm = transforms.Compose([transforms.Grayscale(1),
                              transforms.Resize((224,224)),
                              transforms.ToTensor()])
    sums, sums2, count, sampled = 0.0, 0.0, 0, 0
    for r in radii_list:
        per_r = 0
        subdir = f"squares_R{r}km"
        for row in df.itertuples(index=False):
            if per_r >= limit_per_radius:
                break
            pc = getattr(row, "postcode")
            lat = unique_loc.at[pc, "lat"]
            lon = unique_loc.at[pc, "long"]
            p = os.path.join(img_root, subdir, f"Orth95_{lat}_{lon}_R{r}.jpg")
            try:
                v = tfm(Image.open(p).convert("L")).view(-1)
                sums  += float(v.sum()); sums2 += float((v*v).sum())
                count += v.numel(); per_r += 1; sampled += 1
            except Exception:
                continue
    if count == 0:
        return 0.5, 0.5
    mean = sums / count
    var  = (sums2 / count) - mean*mean
    std  = (var if var > 1e-12 else 1e-12) ** 0.5
    print(f"[gray stats ALL radii] sampled={sampled}, mean={mean:.4f}, std={std:.4f}")
    return float(mean), float(std)

#  dataset 

class MultiRadiiDataset(Dataset):
    """Each sample loads one grayscale image per radius -> stacked [V,1,224,224]."""
    def __init__(self, df, y, unique_loc, img_root, radii_list, img_tfm, log_offsets):
        self.df = df.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.unique_loc = unique_loc
        self.img_root = img_root
        self.radii_list = [str(r) for r in radii_list]
        self.tfm = img_tfm
        self.log_offsets = log_offsets.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        postcode = self.df.iloc[idx]['postcode']
        lat = self.unique_loc.at[postcode, "lat"]
        lon = self.unique_loc.at[postcode, "long"]
        views = []
        for r in self.radii_list:
            p = os.path.join(self.img_root, f"squares_R{r}km", f"Orth95_{lat}_{lon}_R{r}.jpg")
            try:
                img = self.tfm(Image.open(p).convert("L"))
            except Exception as e:
                raise FileNotFoundError(f"Missing image: {p}") from e
            views.append(img)
        images = torch.stack(views, dim=0)  # [V,1,224,224]
        y   = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        off = torch.tensor(self.log_offsets.iloc[idx], dtype=torch.float32)
        return images, off, y

# model 

class MultiRadiiPoissonNet(nn.Module):
    """ResNet18 1-ch backbone, mean-pool features across radii -> MLP -> Poisson mean."""
    def __init__(self, device, weights_path, image_embed_dim=512, hidden_dim=128, dropout=0.1):
        super().__init__()
        base_model = resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location=device)
        base_model.load_state_dict(state_dict)
        # Adapt to 1ch
        with torch.no_grad():
            old = base_model.conv1
            new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        base_model.conv1 = new
        base_model.fc = nn.Identity()
        self.image_backbone = base_model
        self.fc = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images, log_offset=None):
        B, V, C, H, W = images.size()
        x = images.view(B*V, C, H, W)
        feats = self.image_backbone(x).view(B, V, -1).mean(dim=1)
        eta = self.fc(feats).squeeze(1)
        if log_offset is not None:
            eta = eta + log_offset
        return torch.exp(eta)

# training & eval 

def train_fixed_epochs(model, train_loader, val_loader, optimizer, device, scaler, epochs=50):
    tr_losses, va_losses = [], []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for imgs, off, y in train_loader:
            imgs, off, y = imgs.to(device), off.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(imgs, off)
                loss = poisson_nll_torch(y, pred)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tot += loss.item() * len(y)
        tr_losses.append(tot / len(train_loader.dataset))

        model.eval(); totv = 0.0
        with torch.no_grad():
            for imgs, off, y in val_loader:
                imgs, off, y = imgs.to(device), off.to(device), y.to(device)
                with autocast():
                    pred = model(imgs, off)
                    loss = poisson_nll_torch(y, pred)
                totv += loss.item() * len(y)
        va_losses.append(totv / len(val_loader.dataset))
        print(f"[Epoch {ep+1}/{epochs}] Train PNLL={tr_losses[-1]:.4f} | Val PNLL={va_losses[-1]:.4f}")
    return tr_losses, va_losses

def eval_loader_metrics(model, data_loader, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for imgs, off, y in data_loader:
            imgs, off = imgs.to(device), off.to(device)
            with autocast():
                p = model(imgs, off)
            preds.append(p.cpu().numpy()); trues.append(y.numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
    pnll = poisson_nll_numpy(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mean_pred = float(y_pred.mean())
    return pnll, rmse, mean_pred, y_true, y_pred

#  CSV with the results

FIELDNAMES_SUMMARY = [
    "outer_fold","config_id","batch_size","hidden_dim","epochs","lr",
    "dropout","optimizer","weight_decay","seed","radii",
    "rmse_val_mean","rmse_val_std","pnll_val_mean","pnll_val_std",
    "n_inner_folds","timestamp"
]

FIELDNAMES_DETAILS = [
    "outer_fold","config_id","val_fold","rmse_val","pnll_val",
    "n_train","n_val","seed","epochs","batch_size","hidden_dim","lr",
    "dropout","optimizer","weight_decay","radii"
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
        f.flush(); os.fsync(f.fileno()); fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_withfolds_id", required=True)
    ap.add_argument("--unique_loc_csv", required=True)
    ap.add_argument("--outer_fold", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--radii", type=str, required=True,
                    help='Comma-separated list of radii, e.g. "0.5,1,3"')
    ap.add_argument("--weights_path", required=True)

    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--optimizer", choices=["adam","adamw"], required=True)
    ap.add_argument("--weight_decay", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_global_seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)

    df_full = pd.read_csv(args.data_withfolds_id)
    radii_list = [s.strip() for s in args.radii.split(",") if s.strip()]
    if len(radii_list) == 0:
        raise ValueError("--radii provided but empty after parsing.")

    fold_col = "fold"; outer_id = args.outer_fold
    df_outer_test = df_full[df_full[fold_col] == outer_id].copy()
    df_outer_tr   = df_full[df_full[fold_col] != outer_id].copy()
    inner_fold_ids = sorted(df_outer_tr[fold_col].unique())

    df_outer_tr["log_Exposure"]   = np.log(df_outer_tr["expo"])
    df_outer_test["log_Exposure"] = np.log(df_outer_test["expo"])

    unique_loc = pd.read_csv(args.unique_loc_csv).set_index("postcode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_val_rmse, fold_val_pnll, detail_rows_local = [], [], []

    for val_fold in inner_fold_ids:
        is_val = (df_outer_tr[fold_col] == val_fold)
        df_train, df_val = df_outer_tr[~is_val].copy(), df_outer_tr[is_val].copy()
        y_tr, y_va = df_train["nclaims"].reset_index(drop=True), df_val["nclaims"].reset_index(drop=True)

        gray_mean, gray_std = compute_gray_stats_all_radii(df_train, unique_loc, args.img_root, radii_list)

        tfm_train = transforms.Compose([
            transforms.Grayscale(1), transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15), transforms.ToTensor(),
            transforms.Normalize(mean=[gray_mean], std=[gray_std])
        ])
        tfm_eval = transforms.Compose([
            transforms.Grayscale(1), transforms.Resize((224,224)),
            transforms.ToTensor(), transforms.Normalize(mean=[gray_mean], std=[gray_std])
        ])

        train_ds = MultiRadiiDataset(df_train, y_tr, unique_loc, args.img_root, radii_list, tfm_train, df_train["log_Exposure"])
        val_ds   = MultiRadiiDataset(df_val,   y_va, unique_loc, args.img_root, radii_list, tfm_eval,   df_val["log_Exposure"])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=1, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

        model = MultiRadiiPoissonNet(device=device, weights_path=args.weights_path,
                                     image_embed_dim=512, hidden_dim=args.hidden_dim,
                                     dropout=args.dropout).to(device)

        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scaler = GradScaler()

        print(f"\n[outer {outer_id} | inner_val {val_fold}] epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
              f"drop={args.dropout}, opt={args.optimizer}, wd={args.weight_decay}, hd={args.hidden_dim}, radii={radii_list}")
        print("-"*80)

        tr_losses, va_losses = train_fixed_epochs(model, train_loader, val_loader, optimizer, device, scaler, epochs=args.epochs)

        pnll_val, rmse_val, _, _, _ = eval_loader_metrics(model, val_loader, device)
        fold_val_rmse.append(rmse_val); fold_val_pnll.append(pnll_val)

        detail_rows_local.append({
            "outer_fold": outer_id, "val_fold": int(val_fold),
            "rmse_val": rmse_val, "pnll_val": pnll_val,
            "n_train": len(df_train), "n_val": len(df_val),
            "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "hidden_dim": args.hidden_dim,
            "lr": args.lr, "dropout": args.dropout, "optimizer": args.optimizer,
            "weight_decay": args.weight_decay, "radii": ",".join(radii_list)
        })

    # summary across inner folds
    rmse_mean, rmse_std = float(np.mean(fold_val_rmse)), float(np.std(fold_val_rmse))
    pnll_mean, pnll_std = float(np.mean(fold_val_pnll)), float(np.std(fold_val_pnll))

    cfg = {"batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "epochs": args.epochs,
           "lr": args.lr, "dropout": args.dropout, "optimizer": args.optimizer,
           "weight_decay": args.weight_decay, "seed": args.seed, "radii": ",".join(radii_list)}
    cfg_id = stable_config_id(cfg); timestamp = time.time()

    details_csv = os.path.join(args.out_dir, f"inner_details_outer{outer_id}.csv")
    for row in detail_rows_local:
        row["config_id"] = cfg_id
        append_result_row(details_csv, row, FIELDNAMES_DETAILS)

    summary_row = {**cfg, "outer_fold": outer_id, "config_id": cfg_id,
                   "rmse_val_mean": rmse_mean, "rmse_val_std": rmse_std,
                   "pnll_val_mean": pnll_mean, "pnll_val_std": pnll_std,
                   "n_inner_folds": len(inner_fold_ids), "timestamp": timestamp}
    summary_csv = os.path.join(args.out_dir, f"inner_summary_outer{outer_id}.csv")
    append_result_row(summary_csv, summary_row, FIELDNAMES_SUMMARY)

    print(f"[outer {outer_id}] DONE multi-radii images-only inner CV.")
    print(f"config_id={cfg_id} | RMSE={rmse_mean:.4f} | PNLL={pnll_mean:.4f}")
    print(f"Wrote:\n  {summary_csv}\n  {details_csv}")

if __name__ == "__main__":
    main()
