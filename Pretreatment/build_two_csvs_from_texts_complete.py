# build_two_csvs_from_texts_complete.py
# -*- coding: utf-8 -*-
"""
texts/*.txt から2種類のCSV（image,text）を作る

(1) detail_only.csv:
  "<損傷位置、種類、原因>。<損傷の性状に関する見立て>。<損傷の進展予測>。<結論>。<診断文章>。"
  ※欠損項目はその項目だけ落として連結

(2) combined.csv:
  "{base文}。{detail文}。"
  ※base文もdetail文も「欠損箇所のみ落として」自然に連結

画像パスは必ず:
  /media/seino/HDD-UT3/Xroad/images/<filename>
にする（入力に ./images/... や save_path が入っていても上書き）。

実行例:
  python build_two_csvs_from_texts_complete.py \
    --texts_dir /media/seino/HDD-UT3/Xroad/CLIP_training_code/texts \
    --out_dir   /media/seino/HDD-UT3/Xroad/CLIP_training_code
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

KV_RE = re.compile(r"^\s*([^:]+)\s*:\s*(.*)\s*$")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_kv_text(raw: str) -> dict:
    d = {}
    for line in raw.splitlines():
        m = KV_RE.match(line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k:
            d[k] = v
    return d


def try_parse_record(txt_path: Path) -> Optional[dict]:
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    s = raw.strip()

    # JSON優先
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # key:value 形式
    obj = parse_kv_text(raw)
    return obj if obj else None


def _clean(s: str) -> str:
    s = (s or "").replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_base_sentence(obj: dict) -> Optional[str]:
    """
    欠損があっても、存在する要素だけで自然文にする。
    """
    part = _clean(obj.get("part_material_value") or obj.get("part_material") or "")
    damage = _clean(obj.get("damage_type") or obj.get("damage_type_value") or "")
    health = _clean(obj.get("当前健全度判断") or obj.get("current_health") or "")
    counter = _clean(obj.get("当前部材毎の対策区分判定") or obj.get("current_countermeasure") or "")

    clauses = []

    # 主語部（part/damage）
    if part and damage:
        clauses.append(f"{part}において{damage}が見られます")
    elif part:
        clauses.append(f"{part}に損傷が見られます")
    elif damage:
        clauses.append(f"{damage}が見られます")

    # 健全度
    if health:
        clauses.append(f"健全度判定は{health}です")

    # 対策区分
    if counter:
        clauses.append(f"部材毎の対策区分判定は{counter}です")

    if not clauses:
        return None

    # 読点で連結し、末尾句点を統一
    sent = "、".join(clauses).rstrip("。") + "。"
    return sent


def build_detail_sentence(obj: dict) -> Optional[str]:
    """
    欠損項目はその項目だけ削除して連結。
    """
    keys = [
        "損傷位置、種類、原因",
        "損傷の性状に関する見立て",
        "損傷の進展予測",
        "結論",
        "診断文章",
    ]
    parts = []
    for k in keys:
        v = _clean(obj.get(k) or "")
        if v:
            parts.append(v.rstrip("。"))

    if not parts:
        return None

    return "。".join(parts) + "。"


def build_combined_sentence(obj: dict) -> Optional[str]:
    """
    base/detail のうち存在するものだけを、句点重複なしで結合。
    """
    base = build_base_sentence(obj)
    detail = build_detail_sentence(obj)

    texts = []
    if base:
        texts.append(base.rstrip("。"))
    if detail:
        texts.append(detail.rstrip("。"))

    if not texts:
        return None

    return "。".join(texts) + "。"


def make_image_abs_path(obj: dict, images_root_abs: Path) -> Optional[str]:
    """
    画像ファイル名を推定して、必ず
      /media/seino/HDD-UT3/Xroad/images/<filename>
    の形式で返す。
    """
    # 1) path1 があるなら、その basename を採用
    p1 = obj.get("path1")
    if isinstance(p1, str) and p1.strip():
        filename = Path(p1.strip()).name
        return str(images_root_abs / filename)

    # 2) save_path があるなら basename
    sp = _clean(obj.get("save_path") or "")
    if sp:
        filename = Path(sp).name
        return str(images_root_abs / filename)

    # 3) photo_name があるならそれ
    pn = _clean(obj.get("photo_name") or "")
    if pn:
        return str(images_root_abs / pn)

    return None


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "text"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_dir", type=str, default="texts", help="texts/*.txt があるディレクトリ")
    ap.add_argument("--out_dir", type=str, default=".", help="CSV出力先ディレクトリ")
    ap.add_argument(
        "--images_root",
        type=str,
        default="/media/seino/HDD-UT3/Xroad/images",
        help="画像の絶対パスroot（必ずこの配下に寄せる）",
    )
    ap.add_argument("--max_txt_mb", type=float, default=0.0, help="巨大txtスキップ（0で無効）")
    ap.add_argument("--glob", type=str, default="*.txt", help="texts_dir 内を探索するglob（例: '*.txt'）")
    args = ap.parse_args()

    texts_dir = Path(args.texts_dir)
    out_dir = Path(args.out_dir)
    images_root_abs = Path(args.images_root)

    out_detail = out_dir / "detail_only.csv"
    out_combined = out_dir / "combined.csv"

    max_bytes = int(args.max_txt_mb * 1024 * 1024) if args.max_txt_mb and args.max_txt_mb > 0 else 0

    files_iter = texts_dir.glob(args.glob)

    detail_rows = []
    combined_rows = []
    skipped = 0
    parsed = 0

    t0 = time.time()

    iterator = files_iter
    if tqdm is not None:
        iterator = tqdm(iterator, desc="processing texts", unit="file")

    for p in iterator:
        # 巨大txtスキップ（必要なら --max_txt_mb で有効化）
        if max_bytes:
            try:
                if p.stat().st_size > max_bytes:
                    skipped += 1
                    continue
            except Exception:
                pass

        obj = try_parse_record(p)
        if not obj:
            skipped += 1
            continue
        parsed += 1

        image_abs = make_image_abs_path(obj, images_root_abs)
        if not image_abs:
            skipped += 1
            continue

        detail = build_detail_sentence(obj)
        combined = build_combined_sentence(obj)

        # (1) detail_only.csv は detail があるものだけ
        if detail:
            detail_rows.append({"image": image_abs, "text": detail})

        # (2) combined.csv は combined があるものだけ（baseのみ、detailのみ、両方、いずれもOK）
        if combined:
            combined_rows.append({"image": image_abs, "text": combined})

        # tqdmなしでも「止まって見える」を避けるためのログ
        if tqdm is None and parsed % 5000 == 0:
            dt = time.time() - t0
            rate = parsed / dt if dt > 0 else 0
            print(
                f"[parsed {parsed}] elapsed={dt:.1f}s rate={rate:.1f} files/s "
                f"detail={len(detail_rows)} combined={len(combined_rows)} skipped={skipped}"
            )

    write_csv(detail_rows, out_detail)
    write_csv(combined_rows, out_combined)

    dt = time.time() - t0
    rate = parsed / dt if dt > 0 else 0

    print("done.")
    print("texts_dir:", texts_dir)
    print("images_root:", images_root_abs)
    print("parsed:", parsed, "skipped:", skipped)
    print("detail_only rows:", len(detail_rows), "->", out_detail)
    print("combined rows:", len(combined_rows), "->", out_combined)
    print(f"elapsed={dt:.1f}s  avg_rate={rate:.2f} files/s")


if __name__ == "__main__":
    main()