"""
Build an English labeled CSV from a Japanese labeled CSV.

Input : a CSV that has image + text (JP) + label columns (JP)
        [image, text, kenzenudo, taisaku, damage_type, damage_loc]
Output: the same rows with
        - text column translated to English via translation_dict.json
        - label columns mapped to canonical English labels

Usage:
    python -m Pretreatment.build_labeled_en_csv \
        --in  classification/results/unified_val_user.csv \
        --out classification/results/unified_val_user_en.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from classification.data.label_definitions_en import JP_TO_EN_BY_CATEGORY

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DICT = ROOT / "Pretreatment" / "translation_dict.json"


def _convert_label_series(series: pd.Series, jp_to_en: dict[str, str]) -> pd.Series:
    def _one(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        out: list[str] = []
        for tok in str(val).split("|"):
            tok = tok.strip()
            if not tok:
                continue
            en = jp_to_en.get(tok)
            if en is not None and en not in out:
                out.append(en)
        return "|".join(out)
    return series.map(_one)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="JP labeled CSV")
    ap.add_argument("--out", dest="out_path", required=True, help="EN labeled CSV")
    ap.add_argument("--dict", default=str(DEFAULT_DICT), help="translation_dict.json path")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    print(f"[info] rows={len(df)}  columns={list(df.columns)}")

    translations: dict[str, str] = json.loads(Path(args.dict).read_text(encoding="utf-8"))
    print(f"[info] loaded {len(translations)} translations")

    # 1) translate text column
    if "text" in df.columns:
        jp = df["text"].astype(str).map(str.strip)
        en = jp.map(translations)
        missing = int(en.isna().sum())
        if missing:
            print(f"[warn] {missing} rows have untranslated text; keeping JP")
            en = en.where(~en.isna(), jp)
        df["text"] = en
        print(f"[ok] text column translated (missing={missing})")
    else:
        print("[warn] no 'text' column, skipping text translation")

    # 2) convert label columns
    for cat, jp_to_en in JP_TO_EN_BY_CATEGORY.items():
        if cat not in df.columns:
            print(f"[warn] column not found: {cat}")
            continue
        df[cat] = _convert_label_series(df[cat], jp_to_en)
        print(f"[ok] label column converted: {cat}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False, encoding="utf-8")
    print(f"[done] saved: {args.out_path}  rows={len(df)}")


if __name__ == "__main__":
    main()
