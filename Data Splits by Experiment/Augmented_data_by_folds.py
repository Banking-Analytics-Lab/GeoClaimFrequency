import os, argparse, json, functools
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
# This is the final since we require to have one per each data fold!
print = functools.partial(print, flush=True)

# helpers
def set_seed(s=611):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def compute_gray_stats_outer_train(df_outer_train, unique_loc, img_root, radius_km, limit=2000):
    """Compute mean/std from outer-train images only.
      Concatenates all sampled pixels, returns their mean and population std (unbiased=False). These become your per-fold normalization stats."""
    tfm = transforms.Compose([transforms.Grayscale(1),
                              transforms.Resize((224,224)),
                              transforms.ToTensor()])
    vals = []
    for i, row in enumerate(df_outer_train.itertuples(index=False)):
        if i >= limit: break
        pc = getattr(row, "postcode")
        try:
            lat, lon = unique_loc.at[str(pc), "lat"], unique_loc.at[str(pc), "long"]
            path = os.path.join(img_root, f"squares_R{radius_km}km",
                                f"Orth95_{lat}_{lon}_R{radius_km}.jpg")
            vals.append(tfm(Image.open(path).convert("L")).view(-1))
        except Exception:
            continue
    if not vals:
        return 0.5, 0.5
    x = torch.cat(vals, dim=0)
    return float(x.mean()), float(x.std(unbiased=False))

class EmbedDataset(Dataset):
    """Return (idx, postcode, image tensor)."""
    def __init__(self, df, unique_loc, img_root, radius_km, tfm):
        # for keeping the row identity to map embeddings with idx and postcode
        self.df = df[["idx","postcode"]].copy()
        self.df["postcode"] = self.df["postcode"].astype(str) # saving postcode as str
        self.unique_loc = unique_loc; self.img_root = img_root
        self.radius_km = radius_km; self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        # identify the row
        r = self.df.iloc[i]; idx = int(r["idx"]); pc = r["postcode"]
        # select the lat and long for the given postcode
        lat, lon = self.unique_loc.at[pc,"lat"], self.unique_loc.at[pc,"long"]
        #identify the path
        p = os.path.join(self.img_root, f"squares_R{self.radius_km}km",
                         f"Orth95_{lat}_{lon}_R{self.radius_km}.jpg")
        x = self.tfm(Image.open(p).convert("L"))  
        # link the idx the postcode and the image (transformed)
        return idx, pc, x

def build_resnet18_1ch(weights_path, device):
    #constructing a vanilla ResNet18, this is without weights
    m = resnet18(weights=None)
    with torch.no_grad():
        # changing the first conv vector to accept the 1 channel instead of 3
        old = m.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)
    m.conv1 = new; m.fc = nn.Identity() # reemplace the final output to be the 512 D penultimate features
    sd = torch.load(weights_path, map_location="cpu") # loading the check point weights
    if any(k.startswith("image_backbone.") for k in sd.keys()): # Eliminate the prefixes for matching the plain ResNets module names
        sd = {k[len("image_backbone."):]: v for k,v in sd.items()
              if k.startswith("image_backbone.")}
    m.load_state_dict(sd, strict=False) # This is since my model is a modification of the resnet18
    return m.eval().to(device) # desable training

@torch.no_grad() # disabel gradients
def extract_embeddings(backbone, loader, device, prefix="img_embR3km"):
    ids, pcs, vecs = [], [], []
    for idxs, postcodes, x in loader: # iterates batches from the dataloader
        x = x.to(device, non_blocking=True) # moves images to the device
        z = backbone(x) # calculate the embeddings, these ones should be [B,512] or [B,512,1,1]
        if z.ndim == 4: z = z.view(z.size(0), -1) # if dimension 4 flatten it
        vecs.append(z.cpu().numpy()) # accumulate embeddings into vecs
        ids.extend([int(i) for i in idxs]); pcs.extend([str(p) for p in postcodes]) # Also accumulate idx and postcodes
    V = np.concatenate(vecs, axis=0) #Stacks all [B,512] into [N,512], creates column names like img_embR3km_0..511
    cols = [f"{prefix}_{i}" for i in range(V.shape[1])]
    out = pd.DataFrame(V, columns=cols)
    out.insert(0,"idx",ids); out.insert(1,"postcode",pcs) # create a data frame with those columns first idx, second the postcode and following the 512 embeddings
    return out

def main():
    ap = argparse.ArgumentParser("Generate augmented CSV (3 km embeddings) per outer fold.")
    ap.add_argument("--data_withfolds_id", required=True)
    ap.add_argument("--unique_loc_csv", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--weights_path", required=True,
                    help="ResNet checkpoint trained on outer-train (3 km).")
    ap.add_argument("--outer_fold", type=int, required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--norm_json", default=None, help="Optional JSON with {'gray_mean','gray_std'} from the training set (in this case ALL the training!).")
    ap.add_argument("--radius_km", type=str, required=True, help="e.g., '3'.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=611)
    args = ap.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(args.data_withfolds_id)
    if "idx" not in df.columns:
        df = df.reset_index().rename(columns={"index":"idx"})
    df["idx"]=df["idx"].astype(np.int64); df["postcode"]=df["postcode"].astype(str) # convert also the postcode as etr for the original dt

    fold_col="fold"; outer=args.outer_fold ; radius = float(args.radius_km)
    is_test = (df[fold_col]==outer)
    df_train = df.loc[~is_test,["idx","postcode"]] # This is for calculating the norm stats for the images! (Remember this procedure is equivalent to the use of validation in early stopping and also use cv for model selection!, intoduces a little of bias in the cv estimage but the test scores are still valid!)
    df_test  = df.loc[ is_test,["idx","postcode"]]

    loc = pd.read_csv(args.unique_loc_csv).set_index("postcode")
    loc.index = loc.index.astype(str)

    # normalization (prefer JSON if provided; else compute it)
    if args.norm_json and os.path.exists(args.norm_json):
        with open(args.norm_json, "r") as f:
            stats = json.load(f)
        gmean, gstd = float(stats["gray_mean"]), float(stats["gray_std"])
        print(f"[outer {outer}] loaded norm from {args.norm_json}: mean={gmean:.4f}, std={gstd:.4f}")
    else:
        gmean, gstd = compute_gray_stats_outer_train(df_train, loc, args.img_root, radius)
        print(f"[outer {outer}] computed norm on outer-train: mean={gmean:.4f}, std={gstd:.4f}")

    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[gmean], std=[gstd]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_resnet18_1ch(args.weights_path,device)

    # extract embeddings for outer-train + outer-test
    ds_tr = EmbedDataset(df_train,loc,args.img_root,3,tfm)
    ds_te = EmbedDataset(df_test,loc,args.img_root,3,tfm)
    dl_tr = DataLoader(ds_tr,batch_size=args.batch_size,shuffle=False,
                       num_workers=args.num_workers,pin_memory=True)
    dl_te = DataLoader(ds_te,batch_size=args.batch_size,shuffle=False,
                       num_workers=args.num_workers,pin_memory=True)

    emb_tr = extract_embeddings(backbone,dl_tr,device)
    emb_te = extract_embeddings(backbone,dl_te,device)
    emb_all = pd.concat([emb_tr,emb_te],ignore_index=True)

    #save 
    df_aug = df.merge(emb_all,on=["idx","postcode"],how="left")
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    df_aug.to_csv(args.out_path,index=False)
    print("Saved:", args.out_path)

if __name__ == "__main__":
    main()
