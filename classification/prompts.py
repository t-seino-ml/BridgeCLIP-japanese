# -*- coding: utf-8 -*-
"""
全VLMモデル共通の分類プロンプト

GPT-4o / Llama / InternVL3.5 / Qwen3-VL で同一プロンプトを使用する。
比較実験のため、プロンプトはすべてこのファイルで一元管理する。
"""

from classification.data.label_definitions import (
    KENZENUDO_LABELS,
    TAISAKU_LABELS,
    DAMAGE_TYPE_LABELS,
    DAMAGE_LOC_LABELS,
)

# ────────────────────────────── ラベル選択肢文字列 ────────────────────────────────

_KENZENUDO_OPTIONS   = " / ".join(KENZENUDO_LABELS)
_TAISAKU_OPTIONS     = " / ".join(TAISAKU_LABELS)
_DAMAGE_TYPE_OPTIONS = "、".join(DAMAGE_TYPE_LABELS)
_DAMAGE_LOC_OPTIONS  = "、".join(DAMAGE_LOC_LABELS)

# ────────────────────────────── メインプロンプト ──────────────────────────────────

SYSTEM_PROMPT = "あなたは道路橋の点検に精通した専門家です。提示された橋梁点検写真を分析し、指定された形式で回答してください。"

USER_PROMPT = f"""以下の橋梁点検写真を見て、4つの項目を分類してください。
必ず下記のJSON形式のみで回答し、それ以外の文章は一切含めないでください。

```json
{{
  "健全度判定": "<選択肢から1つ>",
  "対策区分": "<選択肢から1つ>",
  "損傷種類": ["<選択肢から1つ以上>"],
  "損傷部位": ["<選択肢から1つ以上>"]
}}
```

【健全度判定の選択肢】（必ず1つ選択）
{_KENZENUDO_OPTIONS}
- Ⅰ: 損傷なし、または軽微な損傷
- Ⅱ: 予防保全段階（損傷はあるが緊急性なし）
- Ⅲ: 早期措置段階（損傷が進行中、早期対応が必要）
- Ⅳ: 緊急措置段階（即時対応が必要）

【対策区分の選択肢】（必ず1つ選択）
{_TAISAKU_OPTIONS}
- A: 措置不要
- B: 予防保全（次回点検までに維持管理）
- C1: 早期措置（概ね5年以内に補修）
- C2: 緊急措置（早急に補修）
- E1: 詳細調査が必要
- E2: 詳細調査が緊急に必要
- M: 維持工事で対処可能
- S1: 詳細調査（構造安全性調査）
- S2: 詳細調査（緊急）

【損傷種類の選択肢】（1つ以上選択、複数可）
{_DAMAGE_TYPE_OPTIONS}

【損傷部位の選択肢】（1つ以上選択、複数可）
{_DAMAGE_LOC_OPTIONS}

画像から判断できない項目は、最も可能性が高いものを推定して回答してください。
"""

# ────────────────────────────── Few-shot例（オプション） ─────────────────────────

FEW_SHOT_EXAMPLE = """
【出力例】
```json
{
  "健全度判定": "Ⅱ",
  "対策区分": "C1",
  "損傷種類": ["ひびわれ", "漏水・遊離石灰"],
  "損傷部位": ["主桁", "床版"]
}
```
"""

# Few-shotを含む完全プロンプト
USER_PROMPT_WITH_EXAMPLE = USER_PROMPT + FEW_SHOT_EXAMPLE

# ────────────────────────────── CLIP用テキストテンプレート ───────────────────────

# CLIP ゼロショット分類に使用するテキストテンプレート
# 各クラスに対して複数のテンプレートを用意し、平均を取る

CLIP_TEMPLATES = {
    "kenzenudo": {
        label: [
            f"健全度{label}の橋梁損傷写真",
            f"道路橋点検 健全度判定{label}",
            f"橋梁の損傷状態 健全度{label}",
        ]
        for label in KENZENUDO_LABELS
    },
    "taisaku": {
        label: [
            f"対策区分{label}の橋梁損傷写真",
            f"道路橋点検 対策区分{label}",
            f"橋梁損傷 対策区分判定{label}",
        ]
        for label in TAISAKU_LABELS
    },
    "damage_type": {
        label: [
            f"橋梁の{label}の写真",
            f"道路橋の{label}損傷",
            f"{label}が見られる橋梁点検写真",
        ]
        for label in DAMAGE_TYPE_LABELS
    },
    "damage_loc": {
        label: [
            f"橋梁の{label}の損傷写真",
            f"道路橋の{label}部分",
            f"{label}に損傷が見られる橋梁点検写真",
        ]
        for label in DAMAGE_LOC_LABELS
    },
}
