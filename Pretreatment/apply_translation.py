"""
Apply the Japanese→English translation dictionary to all CSVs.

Reads:
  - train_subsets_item/*.csv  → writes train_subsets_item_en/*.csv
  - val_item_data/*.csv       → writes val_item_data_en/*.csv

Keeps the original 'image' column unchanged; replaces 'text' with English.
Rows whose Japanese text is not in the dictionary are reported and kept as JP
(so nothing is silently lost).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DICT = Path(__file__).resolve().parent / "translation_dict.json"
PAIRS = [
    (ROOT / "train_subsets_item", ROOT / "train_subsets_item_en"),
    (ROOT / "val_item_data", ROOT / "val_item_data_en"),
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp932"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default=str(DEFAULT_DICT))
    ap.add_argument("--report_missing", default=str(Path(__file__).resolve().parent / "missing_translations.txt"))
    args = ap.parse_args()

    dict_path = Path(args.dict)
    translations: dict[str, str] = json.loads(dict_path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(translations)} translations from {dict_path}")

    all_missing: set[str] = set()
    total_rows = 0
    total_missing = 0

    for in_dir, out_dir in PAIRS:
        if not in_dir.exists():
            print(f"[warn] input dir not found: {in_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(in_dir.glob("*.csv")):
            df = read_csv_robust(csv_path)
            if "text" not in df.columns:
                print(f"[skip] {csv_path.name} (no 'text' column)")
                continue

            original = df["text"].astype(str).map(str.strip)
            mapped = original.map(translations)
            missing_mask = mapped.isna()
            missing_n = int(missing_mask.sum())

            if missing_n:
                missing_texts = set(original[missing_mask].unique())
                all_missing |= missing_texts
                # keep JP for missing rows (don't silently drop data)
                mapped = mapped.where(~missing_mask, original)

            df_out = df.copy()
            df_out["text"] = mapped
            out_path = out_dir / csv_path.name
            df_out.to_csv(out_path, index=False, encoding="utf-8")

            total_rows += len(df_out)
            total_missing += missing_n
            print(
                f"[ok] {csv_path.name}"
                f" -> {out_path.relative_to(ROOT)}"
                f"  rows={len(df_out)}  missing={missing_n}"
            )

    print(f"\n[summary] rows={total_rows}  missing_rows={total_missing}  unique_missing_texts={len(all_missing)}")

    if all_missing:
        report = Path(args.report_missing)
        report.write_text("\n".join(sorted(all_missing)) + "\n", encoding="utf-8")
        print(f"[info] missing texts saved to {report}")


if __name__ == "__main__":
    main()
