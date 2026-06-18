# -*- coding: utf-8 -*-
"""
Convert JP label columns (kenzenudo / taisaku / damage_type / damage_loc) in a
labeled CSV to canonical English labels, writing a new CSV.

The label columns in the input are produced by extract_labels.py and are stored
as '|' -joined strings (e.g., "主桁|床版"). This script maps each JP label to
its canonical English counterpart via JP_TO_EN_BY_CATEGORY, drops unknown
tokens, and writes back the joined English strings.

Usage:
    python -m classification.data.convert_labels_to_en \
        --in  classification/results/labeled_val.csv \
        --out classification/results/labeled_val_en.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification.data.label_definitions_en import JP_TO_EN_BY_CATEGORY


def _convert_series(series: pd.Series, jp_to_en: dict[str, str]) -> pd.Series:
    def _one(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        tokens = str(val).split("|")
        out: list[str] = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            en = jp_to_en.get(tok)
            if en is None:
                # Unknown token (likely a non-canonical variant from extract_labels).
                # Drop it silently — same behavior as downstream multi-hot parsing.
                continue
            if en not in out:
                out.append(en)
        return "|".join(out)
    return series.map(_one)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JP label columns to English")
    ap.add_argument("--in",  dest="in_path",  required=True, help="input labeled CSV (JP)")
    ap.add_argument("--out", dest="out_path", required=True, help="output labeled CSV (EN)")
    ap.add_argument(
        "--keep_text",
        action="store_true",
        help="keep the original 'text' column unchanged (default); if unset, same.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    # Convert both GT columns (e.g., "kenzenudo") and prediction columns
    # (e.g., "pred_kenzenudo") so supervised prediction CSVs can be compared
    # against English VLM / CLIP predictions.
    for cat, jp_to_en in JP_TO_EN_BY_CATEGORY.items():
        for col in (cat, f"pred_{cat}"):
            if col not in df.columns:
                continue
            df[col] = _convert_series(df[col], jp_to_en)
            print(f"[ok] {col}: converted")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False, encoding="utf-8")
    print(f"saved: {args.out_path}  rows={len(df)}")


if __name__ == "__main__":
    main()
