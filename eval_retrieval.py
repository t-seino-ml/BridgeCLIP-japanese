import argparse, os, json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from tqdm import tqdm
import re

class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return self.paths[idx], idx

def _load_img(path: str):
    return Image.open(path).convert("RGB")

@torch.no_grad()
def encode_images(model, preprocess, paths, device, batch_size, num_workers):
    ds = ImgDataset(paths)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda batch: ([b[0] for b in batch], torch.tensor([b[1] for b in batch], dtype=torch.long)),
    )
    feats = torch.empty((len(paths), model.text_projection.shape[1]), device=device, dtype=torch.float16)  # [N,d]
    for b_paths, b_idx in tqdm(dl, desc="Encode images"):
        imgs = torch.stack([preprocess(_load_img(p)) for p in b_paths]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            f = model.encode_image(imgs)
        f = F.normalize(f, dim=-1).to(torch.float16)
        feats[b_idx.to(device)] = f
    return feats

@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size):
    feats = torch.empty((len(texts), model.text_projection.shape[1]), device=device, dtype=torch.float16)
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts"):
        bt = texts[i:i+batch_size]
        tok = tokenizer(bt).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            f = model.encode_text(tok)
        f = F.normalize(f, dim=-1).to(torch.float16)
        feats[i:i+len(bt)] = f
    return feats

def recall_i2t(sim, ks=(1,5,10)):
    # sim: [N,N] on GPU
    N = sim.shape[0]
    maxk = max(ks)
    topk = torch.topk(sim, k=maxk, dim=1).indices  # [N,maxk]
    gt = torch.arange(N, device=sim.device).view(N, 1)
    m = (topk == gt)
    return {f"R@{k}": m[:, :k].any(dim=1).float().mean().item() for k in ks}

def recall_t2i(sim, ks=(1,5,10)):
    N = sim.shape[0]
    maxk = max(ks)
    topk = torch.topk(sim, k=maxk, dim=0).indices  # [maxk,N]
    gt = torch.arange(N, device=sim.device).view(1, N)
    m = (topk == gt)
    return {f"R@{k}": m[:k, :].any(dim=0).float().mean().item() for k in ks}

def recall_i2i_self_included(img_feats, ks=(1,5,10)):
    # self-included retrieval（要求に合わせる）
    sim = img_feats @ img_feats.T  # [N,N]
    N = sim.shape[0]
    maxk = max(ks)
    topk = torch.topk(sim, k=maxk, dim=1).indices
    gt = torch.arange(N, device=sim.device).view(N, 1)
    m = (topk == gt)
    out = {f"R@{k}": m[:, :k].any(dim=1).float().mean().item() for k in ks}
    out["note"] = "I2I is self-included (self should rank top)."
    return out

def load_model(ckpt_path, device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--img-key", default="image")
    ap.add_argument("--txt-key", default="text")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.val_csv)
    paths = df[args.img_key].tolist()
    texts = df[args.txt_key].tolist()
    assert len(paths) == len(texts)

    model, preprocess, tokenizer = load_model(args.ckpt, device)

    img_feats = encode_images(model, preprocess, paths, device, args.batch_size, args.num_workers)
    txt_feats = encode_texts(model, tokenizer, texts, device, args.batch_size)

    sim = img_feats @ txt_feats.T  # [N,N] on GPU

    res = {
        "ckpt": args.ckpt,
        "val_csv": args.val_csv,
        "N": len(paths),
        "i2t": recall_i2t(sim),
        "t2i": recall_t2i(sim),
        "i2i": recall_i2i_self_included(img_feats),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
