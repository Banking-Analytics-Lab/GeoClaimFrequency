import os, argparse, functools
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

print = functools.partial(print, flush=True)

def as_rgb(img):  #Since our tiles are grayscale; most encoders expect RGB
    return img.convert("RGB") if img.mode != "RGB" else img

class PC3km(Dataset):
    # unique_loc indexed by 'postcode' (string) with columns: lat,long
    # Files at: {img_root}/squares_R3km/Orth95_{lat}_{lon}_R3.jpg
    def __init__(self, postcodes, loc_df, img_root):
        self.pcs = pd.Index(postcodes.astype(str).unique()) # list of unique postcodes
        self.loc = loc_df; self.img_root = img_root 
        self.paths = []
        for pc in self.pcs:
            lat = self.loc.at[pc, "lat"]; lon = self.loc.at[pc, "long"] # find the corresponding lat and long
            self.paths.append(os.path.join(img_root, "squares_R3km", f"Orth95_{lat}_{lon}_R3.jpg")) # Find the corresponding image
    def __len__(self): return len(self.pcs)
    def __getitem__(self, i): return self.pcs[i], self.paths[i] # return the postcode and the path of the corresponding image

def collate(batch):
    pcs, paths = zip(*batch)
    return list(pcs), list(paths)

@torch.no_grad()
def embed_batch(paths, processor, model, device, skip_missing=False):
    imgs, keep = [], []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            if skip_missing: continue
            raise FileNotFoundError(p)
        imgs.append(as_rgb(Image.open(p)))
        keep.append(i)
    if not imgs:  # all missing in this batch
        return [], np.zeros((0,1))
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    out = model(**inputs)
    z = getattr(out, "pooler_output", None)
    if z is None and hasattr(out, "last_hidden_state"):
        z = out.last_hidden_state[:, 0]
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return keep, z.detach().cpu().numpy()

def main():
    ap = argparse.ArgumentParser("Make one dataframe: original + Nomic v1.5 embeddings @ 3km")
    ap.add_argument("--data_withfolds_csv", required=True)   # has idx, postcode, fold, etc.
    ap.add_argument("--unique_loc_csv",     required=True)   # columns: postcode, lat, long
    ap.add_argument("--img_root",           required=True)
    ap.add_argument("--local_model_dir",    required=True)   # your saved HF model dir
    ap.add_argument("--batch_size",         type=int, default=128)
    ap.add_argument("--num_workers",        type=int, default=2)
    ap.add_argument("--skip_missing",       action="store_true", help="fill NaNs if tile missing")
    ap.add_argument("--out_parquet",        required=True)
    ap.add_argument("--also_csv",           action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.data_withfolds_csv)
    if "idx" not in df.columns:
        df = df.reset_index().rename(columns={"index":"idx"})
    df["idx"] = df["idx"].astype(np.int64)
    df["postcode"] = df["postcode"].astype(str)

    loc = pd.read_csv(args.unique_loc_csv).set_index("postcode")
    loc.index = loc.index.astype(str)

    ds = PC3km(df["postcode"], loc, args.img_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate)

    processor = AutoImageProcessor.from_pretrained(args.local_model_dir)
    model     = AutoModel.from_pretrained(args.local_model_dir, trust_remote_code=True).to(device).eval()

    # collect embeddings per postcode
    pc_list, vec_list = [], []
    for pcs, paths in loader:
        keep_idx, vec = embed_batch(paths, processor, model, device, args.skip_missing)
        if len(keep_idx) == 0:  # all missing in this batch
            continue
        pc_list.extend([pcs[i] for i in keep_idx])
        vec_list.append(vec)

    if vec_list:
        M = np.vstack(vec_list)
        emb_cols = [f"nomic_v15_R3km_{i:03d}" for i in range(M.shape[1])]
        emb_df = pd.DataFrame(M, columns=emb_cols)
        emb_df.insert(0, "postcode", pc_list)
    else:
        # no images found; create empty block with NaNs
        emb_df = pd.DataFrame({"postcode": df["postcode"].unique()})
        emb_cols = [f"nomic_v15_R3km_{i:03d}" for i in range(1)]
        for c in emb_cols: emb_df[c] = np.nan

    # if skipping missing, fill absent postcodes with NaNs and keep merge shape
    if args.skip_missing:
        full_pc = pd.DataFrame({"postcode": df["postcode"].astype(str).unique()})
        emb_df = full_pc.merge(emb_df, on="postcode", how="left")

    df_aug = df.merge(emb_df, on="postcode", how="left")

    outp = Path(args.out_parquet); outp.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_parquet(outp, index=False)
    print(f"Saved Parquet: {outp}")
    if args.also_csv:
        csv_p = outp.with_suffix(".csv"); df_aug.to_csv(csv_p, index=False)
        print(f"Saved CSV: {csv_p}")

if __name__ == "__main__":
    main()
