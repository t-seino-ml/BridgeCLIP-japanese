import argparse, os, re, glob, math

RE = re.compile(r"clip_val_loss:\s*([0-9.]+)\s+epoch:\s*([0-9.]+)")

def find_log(run_dir: str):
    # open_clip_train は run_dir/out.log にも出るが、環境により場所が違うので広めに探す
    cand = []
    cand += glob.glob(os.path.join(run_dir, "out.log"))
    cand += glob.glob(os.path.join(run_dir, "*.log"))
    cand += glob.glob(os.path.join(run_dir, "**", "out.log"), recursive=True)
    cand += glob.glob(os.path.join(run_dir, "**", "*.log"), recursive=True)
    cand = [p for p in cand if os.path.isfile(p)]
    # なるべく短いパス（直下）優先
    cand.sort(key=lambda p: (p.count(os.sep), -os.path.getsize(p)))
    return cand[0] if cand else None

def best_epoch_from_log(log_path: str):
    best = None  # (loss, epoch_int, epoch_float)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE.search(line)
            if not m:
                continue
            loss = float(m.group(1))
            epf = float(m.group(2))
            epi = int(round(epf))  # 1.0000 -> 1
            if best is None or loss < best[0]:
                best = (loss, epi, epf)
    return best  # or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"[ERR] checkpoints not found: {ckpt_dir}")

    log_path = find_log(run_dir)
    if log_path is None:
        # fallback
        best_ckpt = os.path.join(ckpt_dir, "epoch_latest.pt")
        print(f"{os.path.basename(run_dir)}\t{best_ckpt}\tNA\tNA")
        return

    best = best_epoch_from_log(log_path)
    if best is None:
        best_ckpt = os.path.join(ckpt_dir, "epoch_latest.pt")
        print(f"{os.path.basename(run_dir)}\t{best_ckpt}\tNA\tNA")
        return

    loss, epi, epf = best
    # open_clip_train の命名は epoch_{k}.pt が一般的
    ckpt = os.path.join(ckpt_dir, f"epoch_{epi}.pt")
    if not os.path.isfile(ckpt):
        # もし無ければ epoch_latest
        ckpt = os.path.join(ckpt_dir, "epoch_latest.pt")
    print(f"{os.path.basename(run_dir)}\t{ckpt}\t{epi}\t{loss}")

if __name__ == "__main__":
    main()
