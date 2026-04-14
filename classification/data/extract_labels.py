# -*- coding: utf-8 -*-
"""
CSVテキストからラベルを抽出して labeled_dataset.csv を生成する。

使い方:
    python -m classification.data.extract_labels \
        --input data/val_clean.csv \
        --output classification/results/labeled_val.csv

    python -m classification.data.extract_labels \
        --input data/train_clean.csv \
        --output classification/results/labeled_train.csv \
        --report                        # ラベル分布レポートも出力

出力CSV列:
    image, text,
    kenzenudo,          # 健全度判定ラベル (例: "Ⅱ")
    taisaku,            # 対策区分ラベル   (例: "C1")
    damage_type,        # 損傷種類リスト   (例: "ひびわれ|漏水・遊離石灰")
    damage_loc,         # 損傷部位リスト   (例: "主桁|床版")
    kenzenudo_valid,    # 抽出成功フラグ
    taisaku_valid,
    damage_type_valid,
    damage_loc_valid
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from classification.data.label_definitions import (
    KENZENUDO_LABELS,
    TAISAKU_LABELS,
    DAMAGE_TYPE_LABELS,
    DAMAGE_LOC_LABELS,
    KENZENUDO_NORMALIZE,
    TAISAKU_NORMALIZE,
    DAMAGE_TYPE_NORMALIZE,
    DAMAGE_LOC_NORMALIZE,
)

# ────────────────────────────── 正規表現パターン ─────────────────────────────────

_RE_KENZENUDO = re.compile(r"健全度判定は\s*([^\s、。でですが]+)\s*です")
_RE_TAISAKU   = re.compile(r"対策区分判定は\s*([A-Za-zＡ-Ｚ１-９0-9１２ＳＣＥＭ][12１２]?)\s*(?:です|判定|$)")
_RE_DAMAGE    = re.compile(r"([^、。\s]+)において([^、。が]+?)が見られます")

# 構造化部分（第1文）のみを抽出するパターン
# 「<部位>において<損傷種類>が見られます、健全度判定は<X>です、部材毎の対策区分判定は<Y>です。」
_RE_STRUCTURED_PART = re.compile(
    r"^([^。]+(?:において|に)[^。]*(?:が見られます|損傷が見られます)[^。]*。)"
)


def extract_structured_text(text: str) -> str:
    """
    テキストから構造化部分（第1文）のみを返す。

    入力例:
        「側壁においてひびわれが見られます、健全度判定はⅠです、部材毎の対策区分判定はBです。
         側壁（A2）に、剥離（50×150×10mm他）が生じている。原因は...」
    出力例:
        「側壁においてひびわれが見られます、健全度判定はⅠです、部材毎の対策区分判定はBです。」

    構造化部分が見つからない場合はテキスト全体をそのまま返す。
    """
    m = _RE_STRUCTURED_PART.match(text.strip())
    return m.group(1) if m else text

# 補助パターン（メインパターンに引っかからない場合のフォールバック）
_RE_KENZENUDO_ALT = re.compile(r"健全度\s*[:：]\s*([ⅠⅡⅢⅣIVIVI1-4１-４一二三四]+)")
_RE_TAISAKU_ALT   = re.compile(r"対策区分\s*[:：]\s*([A-Za-zＡ-Ｚ][12１２]?)")


def _normalize_kenzenudo(raw: str) -> str | None:
    """健全度判定を正規化する。"""
    raw = raw.strip()
    if raw in KENZENUDO_LABELS:
        return raw
    # 正規化マップで変換
    normalized = KENZENUDO_NORMALIZE.get(raw)
    if normalized:
        return normalized
    # 大文字・全角変換後に再試行
    upper = raw.upper().strip()
    return KENZENUDO_NORMALIZE.get(upper)


def _normalize_taisaku(raw: str) -> str | None:
    """対策区分を正規化する。"""
    raw = raw.strip()
    if raw in TAISAKU_LABELS:
        return raw
    # 正規化マップ
    normalized = TAISAKU_NORMALIZE.get(raw)
    if normalized and normalized in TAISAKU_LABELS:
        return normalized
    # 大文字変換後に再試行
    upper = raw.upper().replace("　", "").replace(" ", "")
    if upper in TAISAKU_LABELS:
        return upper
    return TAISAKU_NORMALIZE.get(upper)


def _normalize_damage_type(raw: str) -> str:
    """損傷種類を正規化する。マッチしなければ 'その他' を返す。"""
    raw = raw.strip()
    if raw in DAMAGE_TYPE_LABELS:
        return raw
    # 正規化マップ
    for key, val in DAMAGE_TYPE_NORMALIZE.items():
        if raw.startswith(key) or raw == key:
            return val
    # 部分一致でチェック
    for label in DAMAGE_TYPE_LABELS:
        if label in raw or raw in label:
            return label
    return "その他"


def _normalize_damage_loc(raw: str) -> str:
    """損傷部位を正規化する。マッチしなければ 'その他' を返す。"""
    raw = raw.strip()
    if raw in DAMAGE_LOC_LABELS:
        return raw
    # 正規化マップ
    for key, val in DAMAGE_LOC_NORMALIZE.items():
        if raw == key or raw.startswith(key):
            return val
    # 部分一致でチェック
    for label in DAMAGE_LOC_LABELS:
        if label in raw:
            return label
    return "その他"


def extract_labels_from_text(text: str) -> dict:
    """
    テキストから4カテゴリのラベルを抽出して返す。

    Returns:
        {
          "kenzenudo":       str | None,   # 正規化済みラベル or None
          "taisaku":         str | None,
          "damage_type":     list[str],    # 1件以上（空の場合はその他）
          "damage_loc":      list[str],    # 1件以上（空の場合はその他）
          "kenzenudo_valid": bool,
          "taisaku_valid":   bool,
          "damage_type_valid": bool,
          "damage_loc_valid":  bool,
        }
    """
    result = {
        "kenzenudo": None, "taisaku": None,
        "damage_type": [], "damage_loc": [],
        "kenzenudo_valid": False, "taisaku_valid": False,
        "damage_type_valid": False, "damage_loc_valid": False,
    }

    # ── 健全度判定 ──
    m = _RE_KENZENUDO.search(text)
    if not m:
        m = _RE_KENZENUDO_ALT.search(text)
    if m:
        label = _normalize_kenzenudo(m.group(1))
        if label:
            result["kenzenudo"] = label
            result["kenzenudo_valid"] = True

    # ── 対策区分 ──
    m = _RE_TAISAKU.search(text)
    if not m:
        m = _RE_TAISAKU_ALT.search(text)
    if m:
        label = _normalize_taisaku(m.group(1))
        if label:
            result["taisaku"] = label
            result["taisaku_valid"] = True

    # ── 損傷部位 + 損傷種類 ──
    # パターン: "{部位}において{損傷種類}が見られます"
    damage_locs  = []
    damage_types = []
    for m in _RE_DAMAGE.finditer(text):
        loc_raw  = m.group(1).strip()
        type_raw = m.group(2).strip()
        # 部位が長すぎる場合（文章全体を拾っている）はスキップ
        if len(loc_raw) > 20:
            continue
        damage_locs.append(_normalize_damage_loc(loc_raw))
        # 損傷種類は "、" 区切りで複数ある場合があるので分割
        for t in re.split(r"[、,]", type_raw):
            t = t.strip()
            if t:
                damage_types.append(_normalize_damage_type(t))

    # 重複除去・順序維持
    seen = set()
    damage_locs  = [x for x in damage_locs  if not (x in seen or seen.add(x))]
    seen = set()
    damage_types = [x for x in damage_types if not (x in seen or seen.add(x))]

    if damage_locs:
        result["damage_loc"] = damage_locs
        result["damage_loc_valid"] = True
    if damage_types:
        result["damage_type"] = damage_types
        result["damage_type_valid"] = True

    return result


def process_csv(input_path: str, output_path: str, report: bool = False) -> pd.DataFrame:
    """
    入力CSVを読み込み、ラベルを抽出してoutput_pathに保存する。
    """
    df = pd.read_csv(input_path)
    assert "image" in df.columns and "text" in df.columns, \
        "CSVに 'image' と 'text' カラムが必要です"

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ラベル抽出"):
        # 構造化部分（第1文）のみ使用
        structured_text = extract_structured_text(str(row["text"]))
        labels = extract_labels_from_text(structured_text)
        rows.append({
            "image":             row["image"],
            "text":              structured_text,   # 第1文のみ保存
            "kenzenudo":         labels["kenzenudo"] or "",
            "taisaku":           labels["taisaku"] or "",
            "damage_type":       "|".join(labels["damage_type"]),
            "damage_loc":        "|".join(labels["damage_loc"]),
            "kenzenudo_valid":   labels["kenzenudo_valid"],
            "taisaku_valid":     labels["taisaku_valid"],
            "damage_type_valid": labels["damage_type_valid"],
            "damage_loc_valid":  labels["damage_loc_valid"],
        })

    out_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"保存完了: {output_path}  ({len(out_df)} 行)")

    # 抽出率の報告
    for col in ["kenzenudo_valid", "taisaku_valid", "damage_type_valid", "damage_loc_valid"]:
        n = out_df[col].sum()
        print(f"  {col}: {n}/{len(out_df)} ({100*n/len(out_df):.1f}%)")

    if report:
        _print_label_distribution(out_df)

    return out_df


def _print_label_distribution(df: pd.DataFrame) -> None:
    """ラベル分布をコンソールに出力する。"""
    print("\n── 健全度判定 分布 ──")
    print(df["kenzenudo"].value_counts().to_string())

    print("\n── 対策区分 分布 ──")
    print(df["taisaku"].value_counts().to_string())

    print("\n── 損傷種類 分布（上位20） ──")
    from collections import Counter
    dtype_counter: Counter = Counter()
    for val in df["damage_type"].dropna():
        for t in val.split("|"):
            if t:
                dtype_counter[t] += 1
    for k, v in dtype_counter.most_common(20):
        print(f"  {k}: {v}")

    print("\n── 損傷部位 分布（上位20） ──")
    dloc_counter: Counter = Counter()
    for val in df["damage_loc"].dropna():
        for t in val.split("|"):
            if t:
                dloc_counter[t] += 1
    for k, v in dloc_counter.most_common(20):
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="CSVテキストからラベルを抽出する")
    parser.add_argument("--input",  required=True, help="入力CSV (image, text)")
    parser.add_argument("--output", required=True, help="出力CSV（ラベル付き）")
    parser.add_argument("--report", action="store_true", help="ラベル分布レポートを表示")
    args = parser.parse_args()

    process_csv(args.input, args.output, report=args.report)


if __name__ == "__main__":
    main()
