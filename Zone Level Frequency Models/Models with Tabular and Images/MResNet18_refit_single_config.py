import os, argparse, time, json, functools
import numpy as np
import pandas as pd
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
from joblib import dump as joblib_dump
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

# utils 
def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def poisson_nll_torch(y_true, y_pred, eps=1e-8):
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps) + torch.lgamma(y_true + 1))

def poisson_nll_numpy(y_true, y_pred, eps=1e-8):
    return float(np.mean(y_pred - y_true * np.log(y_pred + eps) + gammaln(y_true + 1)))

def preprocess_tabular_data(X_tr, X_te, num_vars, cat_vars=None):
    if cat_vars is None or len(cat_vars) == 0:
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", StandardScaler())]), num_vars)],
            remainder="drop")
    else:
        categorical = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([
            ("num", Pipeline([("scaler", StandardScaler())]), num_vars),
            ("cat", categorical, cat_vars),
        ], remainder="drop")

    Xtr = pre.fit_transform(X_tr)
    Xte = pre.transform(X_te)
    return Xtr, Xte, pre

def compute_gray_stats(df_tr, unique_loc, img_root, radius_km, limit=2000):
    tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor()])
    vals = []
    for i, row in enumerate(df_tr.itertuples(index=False)):
        if i >= limit: break
        postcode = getattr(row, "postcode")
        lat = unique_loc.at[postcode, "lat"]
        lon = unique_loc.at[postcode, "long"]
        p = os.path.join(img_root, f"squares_R{radius_km}km", f"Orth95_{lat}_{lon}_R{radius_km}.jpg")
        try:
            x = tfm(Image.open(p).convert("L")).view(-1)
            vals.append(x)
        except Exception:
            continue
    if not vals:
        return 0.5, 0.5
    x = torch.cat(vals)
    return float(x.mean()), float(x.std(unbiased=False))

class ClaimMultimodalDataset(Dataset):
    def __init__(self, df, y, unique_loc, img_root, radius_km, img_tfm, tab_array, log_offsets):
        self.df = df.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.unique_loc = unique_loc
        self.img_root = img_root
        self.radius_km = radius_km
        self.tfm = img_tfm
        self.tab_array = tab_array
        self.log_offsets = log_offsets.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        postcode = self.df.iloc[idx]["postcode"]
        lat = self.unique_loc.at[postcode, "lat"]
        lon = self.unique_loc.at[postcode, "long"]
        p = os.path.join(self.img_root, f"squares_R{self.radius_km}km", f"Orth95_{lat}_{lon}_R{self.radius_km}.jpg")
        img = self.tfm(Image.open(p).convert("L"))
        tab = torch.tensor(self.tab_array[idx], dtype=torch.float32)
        y   = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        off = torch.tensor(self.log_offsets.iloc[idx], dtype=torch.float32)
        return img, tab, off, y

class MultimodalPoissonNet(nn.Module):
    def __init__(self, tabular_dim, device, weights_path,
                 image_embed_dim=512, hidden_dim=128, dropout=0.1):
        super().__init__()
        base = resnet18(weights=None)

        # load pretrained resnet18 weights (ImageNet)
        state_dict = torch.load(weights_path, map_location=device)
        base.load_state_dict(state_dict)

        # adapt 3ch->1ch
        with torch.no_grad():
            old = base.conv1
            new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        base.conv1 = new
        base.fc = nn.Identity()
        self.image_backbone = base

        self.fc = nn.Sequential(
            nn.Linear(tabular_dim + image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, tabular, log_offset=None):
        img_feat = self.image_backbone(image)   # [B,512]
        x = torch.cat([img_feat, tabular], dim=1)
        eta = self.fc(x).squeeze(1)
        if log_offset is not None:
            eta = eta + log_offset
        return torch.exp(eta)

def train_fixed_epochs(model, train_loader, val_loader, optimizer, device, scaler, epochs=50):
    tr_losses, va_losses = [], []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for img, tab, off, y in train_loader:
            img, tab, off, y = img.to(device), tab.to(device), off.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(img, tab, off)
                loss = poisson_nll_torch(y, pred)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tot += loss.item() * len(y)
        tr_losses.append(tot / len(train_loader.dataset))

        model.eval()
        totv = 0.0
        with torch.no_grad():
            for img, tab, off, y in val_loader:
                img, tab, off, y = img.to(device), tab.to(device), off.to(device), y.to(device)
                with autocast():
                    pred = model(img, tab, off)
                    loss = poisson_nll_torch(y, pred)
                totv += loss.item() * len(y)
        va_losses.append(totv / len(val_loader.dataset))
        print(f"[Epoch {ep+1}/{epochs}] Train PNLL={tr_losses[-1]:.4f} | Val PNLL={va_losses[-1]:.4f}")
    return tr_losses, va_losses

def eval_loader_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for img, tab, off, y in loader:
            img, tab, off = img.to(device), tab.to(device), off.to(device)
            with autocast():
                p = model(img, tab, off)
            preds.append(p.cpu().numpy())
            trues.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    pnll = poisson_nll_numpy(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return pnll, rmse, float(y_pred.mean()), y_true, y_pred

def save_learning_curves(tr, va, out_png, title=None):
    fig = plt.figure()
    plt.plot(tr, label="Train PNLL"); plt.plot(va, label="Val PNLL")
    plt.xlabel("Epoch"); plt.ylabel("Poisson NLL"); plt.grid(True); plt.legend()
    if title: plt.title(title)
    fig.savefig(out_png, bbox_inches="tight", dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()

    # data / structure
    ap.add_argument("--data_withfolds_id", required=True)
    ap.add_argument("--unique_loc_csv", required=True)
    ap.add_argument("--outer_fold", type=int, required=True)
    ap.add_argument("--out_dir", required=True)

    # images
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--radius_km", required=True)

    # features
    ap.add_argument("--num_vars", type=str, required=True,
                    help="Python list string of numeric feature names")
    ap.add_argument("--cat_vars", default="None",
                    help="Python list string of categorical names or 'None'")

    # pretrained backbone
    ap.add_argument("--weights_path", required=True)

    # best hyperparameters (fixed)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--optimizer", choices=["adam","adamw"], required=True)
    ap.add_argument("--weight_decay", type=float, required=True)

    ap.add_argument("--seed", type=int, default=42)

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

    is_outer_test = (df_full[fold_col] == outer_id)
    df_outer_test = df_full[is_outer_test].copy()
    df_outer_tr   = df_full[~is_outer_test].copy()

    # offsets
    df_outer_tr["log_Exposure"]   = np.log(df_outer_tr["expo"])
    df_outer_test["log_Exposure"] = np.log(df_outer_test["expo"])

    unique_loc = pd.read_csv(args.unique_loc_csv).set_index("postcode")

    # targets
    y_outer_tr   = df_outer_tr["nclaims"].reset_index(drop=True)
    y_outer_test = df_outer_test["nclaims"].reset_index(drop=True)

    # preprocess (fit on all the outer-train)
    Xtr_proc, Xte_proc, preproc = preprocess_tabular_data(
        df_outer_tr[num_vars + cat_vars] if use_cats else df_outer_tr[num_vars],
        df_outer_test[num_vars + cat_vars] if use_cats else df_outer_test[num_vars],
        num_vars=num_vars,
        cat_vars=cat_vars if use_cats else None
    )

    # image stats on all outer-train
    gray_mean, gray_std = compute_gray_stats(df_outer_tr, unique_loc, args.img_root, args.radius_km)

    tfm_train = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[gray_mean], std=[gray_std]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[gray_mean], std=[gray_std]),
    ])

    # datasets / loaders
    train_ds = ClaimMultimodalDataset(
        df_outer_tr, y_outer_tr, unique_loc, args.img_root, args.radius_km,
        tfm_train, Xtr_proc, df_outer_tr["log_Exposure"]
    )
    test_ds = ClaimMultimodalDataset(
        df_outer_test, y_outer_test, unique_loc, args.img_root, args.radius_km,
        tfm_eval, Xte_proc, df_outer_test["log_Exposure"]
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=1, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    tabular_dim = train_ds.tab_array.shape[1]
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler()

    # train on ALL outer-train (track test as "val" curve)
    print(f"[outer_fold {outer_id}] Refit: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
          f"dropout={args.dropout}, opt={args.optimizer}, wd={args.weight_decay}, hd={args.hidden_dim}")
    tr_losses, va_losses = train_fixed_epochs(model, train_loader, test_loader, optimizer, device, scaler, epochs=args.epochs)

    # evaluate on outer-test
    pnll_te, rmse_te, mean_pred_te, y_true_te, y_pred_te = eval_loader_metrics(model, test_loader, device)

    # save the output
    # learning curves
    curve_png = os.path.join(args.out_dir, f"learning_curve_outer{outer_id}.png")
    save_learning_curves(tr_losses, va_losses, curve_png, title=f"Outer {outer_id} refit")

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
        "seed": args.seed
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

    # saving model and preprocessor and image stats
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"refit_model_outer{outer_id}.pth"))
    joblib_dump(preproc, os.path.join(args.out_dir, f"preprocessor_outer{outer_id}.joblib"))
    with open(os.path.join(args.out_dir, f"image_norm_outer{outer_id}.json"), "w") as f:
        json.dump({"gray_mean": gray_mean, "gray_std": gray_std}, f)

    print(f"[outer_fold {outer_id}] DONE. rmse_test={rmse_te:.6f}, pnll_test={pnll_te:.6f}")
    print("Saved to:")
    print(" ", curve_png)
    print(" ", eval_path)
    print(" ", preds_path)

if __name__ == "__main__":
    main()
