# -*- coding: utf-8 -*-
"""
English label definitions, paired 1:1 with label_definitions.py (Japanese).

Order matters: index i in the English list corresponds to index i in the Japanese
list. Use the JP_TO_EN / EN_TO_JP mappings below to convert between languages.

Soundness rating and measure classification use ASCII symbols (I/II/III/IV vs
Japanese Ⅰ/Ⅱ/Ⅲ/Ⅳ; letter codes are already ASCII).
"""

from classification.data.label_definitions import (
    KENZENUDO_LABELS as _KENZENUDO_LABELS_JP,
    TAISAKU_LABELS as _TAISAKU_LABELS_JP,
    DAMAGE_TYPE_LABELS as _DAMAGE_TYPE_LABELS_JP,
    DAMAGE_LOC_LABELS as _DAMAGE_LOC_LABELS_JP,
)

# ────────────────────────────── Canonical English labels ─────────────────────

KENZENUDO_LABELS = ["I", "II", "III", "IV"]

TAISAKU_LABELS = ["A", "B", "C1", "C2", "E1", "E2", "M", "S1", "S2"]

DAMAGE_TYPE_LABELS = [
    "cracking",                                 # ひびわれ
    "spalling / rebar exposure",                # 剥離・鉄筋露出
    "corrosion",                                # 腐食
    "delamination",                             # うき
    "water leakage / efflorescence",            # 漏水・遊離石灰
    "deformation / defect",                     # 変形・欠損
    "soil/debris clogging",                     # 土砂詰まり
    "loosening / dislodgement",                 # ゆるみ・脱落
    "deterioration of anti-corrosion function", # 防食機能の劣化
    "bearing dysfunction",                      # 支承の機能障害
    "fracture",                                 # 破断
    "fatigue cracking",                         # き裂
    "pavement defect",                          # 舗装の異常
    "scour",                                    # 洗掘
    "other",                                    # その他
]

DAMAGE_LOC_LABELS = [
    "main girder",                 # 主桁
    "cross girder",                # 横桁
    "deck slab",                   # 床版
    "vertical wall",               # 竪壁
    "wing wall",                   # 翼壁
    "parapet",                     # 胸壁
    "pier",                        # 橋脚
    "bearing",                     # 支承
    "expansion joint",             # 伸縮装置
    "pavement",                    # 舗装
    "drainage facility",           # 排水施設
    "railing",                     # 高欄
    "guardrail",                   # 防護柵
    "curb",                        # 地覆
    "attached equipment",          # 添架物
    "top slab",                    # 頂版
    "bottom slab",                 # 底版
    "side wall",                   # 側壁
    "unseating prevention system", # 落橋防止システム
    "other",                       # その他
]

# Sanity: parallel lists must stay 1:1 with the Japanese definitions.
assert len(KENZENUDO_LABELS) == len(_KENZENUDO_LABELS_JP)
assert len(TAISAKU_LABELS) == len(_TAISAKU_LABELS_JP)
assert len(DAMAGE_TYPE_LABELS) == len(_DAMAGE_TYPE_LABELS_JP)
assert len(DAMAGE_LOC_LABELS) == len(_DAMAGE_LOC_LABELS_JP)

ALL_LABEL_SETS = {
    "kenzenudo":   KENZENUDO_LABELS,
    "taisaku":     TAISAKU_LABELS,
    "damage_type": DAMAGE_TYPE_LABELS,
    "damage_loc":  DAMAGE_LOC_LABELS,
}

NUM_CLASSES = {k: len(v) for k, v in ALL_LABEL_SETS.items()}

# ────────────────────────────── JP ↔ EN label mappings ───────────────────────

KENZENUDO_JP_TO_EN   = dict(zip(_KENZENUDO_LABELS_JP,   KENZENUDO_LABELS))
TAISAKU_JP_TO_EN     = dict(zip(_TAISAKU_LABELS_JP,     TAISAKU_LABELS))
DAMAGE_TYPE_JP_TO_EN = dict(zip(_DAMAGE_TYPE_LABELS_JP, DAMAGE_TYPE_LABELS))
DAMAGE_LOC_JP_TO_EN  = dict(zip(_DAMAGE_LOC_LABELS_JP,  DAMAGE_LOC_LABELS))

KENZENUDO_EN_TO_JP   = {v: k for k, v in KENZENUDO_JP_TO_EN.items()}
TAISAKU_EN_TO_JP     = {v: k for k, v in TAISAKU_JP_TO_EN.items()}
DAMAGE_TYPE_EN_TO_JP = {v: k for k, v in DAMAGE_TYPE_JP_TO_EN.items()}
DAMAGE_LOC_EN_TO_JP  = {v: k for k, v in DAMAGE_LOC_JP_TO_EN.items()}

JP_TO_EN_BY_CATEGORY = {
    "kenzenudo":   KENZENUDO_JP_TO_EN,
    "taisaku":     TAISAKU_JP_TO_EN,
    "damage_type": DAMAGE_TYPE_JP_TO_EN,
    "damage_loc":  DAMAGE_LOC_JP_TO_EN,
}

# ────────────────────────────── Normalization maps ───────────────────────────
# VLMs often emit minor surface variations; normalize them before matching.
# Keys are looked up case-insensitively in run_on_csv / _parse_response.

KENZENUDO_NORMALIZE = {
    # Japanese Roman numerals → ASCII.
    "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV",
    # Arabic digits.
    "1": "I", "2": "II", "3": "III", "4": "IV",
    # Full-width digits.
    "１": "I", "２": "II", "３": "III", "４": "IV",
    # Lowercase ASCII.
    "i": "I", "ii": "II", "iii": "III", "iv": "IV",
    # Common VLM phrases.
    "Grade I": "I", "Grade II": "II", "Grade III": "III", "Grade IV": "IV",
    "Rating I": "I", "Rating II": "II", "Rating III": "III", "Rating IV": "IV",
    "Roman I": "I", "Roman II": "II", "Roman III": "III", "Roman IV": "IV",
}

TAISAKU_NORMALIZE = {
    # Lowercase.
    "a": "A", "b": "B", "m": "M",
    "c1": "C1", "c2": "C2",
    "e1": "E1", "e2": "E2",
    "s1": "S1", "s2": "S2",
    # Full-width.
    "Ａ": "A", "Ｂ": "B", "Ｍ": "M",
    "Ｃ１": "C1", "Ｃ２": "C2",
    "Ｅ１": "E1", "Ｅ２": "E2",
    "Ｓ１": "S1", "Ｓ２": "S2",
    # Legacy single-letter E → E1 (matches JP convention).
    "E": "E1",
    # Stray whitespace.
    "C 1": "C1", "C 2": "C2", "S 1": "S1", "S 2": "S2",
}

DAMAGE_TYPE_NORMALIZE = {
    # cracking
    "crack": "cracking",
    "cracks": "cracking",
    "slab cracking": "cracking",
    "deck slab cracking": "cracking",
    "joint cracking": "cracking",
    # fatigue cracking (separate category)
    "fatigue crack": "fatigue cracking",
    "fatigue cracks": "fatigue cracking",
    # spalling / rebar exposure
    "spalling": "spalling / rebar exposure",
    "rebar exposure": "spalling / rebar exposure",
    "spalling and rebar exposure": "spalling / rebar exposure",
    "spall": "spalling / rebar exposure",
    "concrete spalling": "spalling / rebar exposure",
    # delamination
    "hollow area": "delamination",
    "laminar separation": "delamination",
    # water leakage / efflorescence
    "water leakage": "water leakage / efflorescence",
    "leakage": "water leakage / efflorescence",
    "efflorescence": "water leakage / efflorescence",
    "free lime": "water leakage / efflorescence",
    "water leakage / pooling": "water leakage / efflorescence",
    "water leakage / free lime": "water leakage / efflorescence",
    "water leakage and efflorescence": "water leakage / efflorescence",
    "water seepage": "water leakage / efflorescence",
    # deformation / defect
    "deformation": "deformation / defect",
    "defect": "deformation / defect",
    "deformation and defect": "deformation / defect",
    "damage": "deformation / defect",
    # loosening / dislodgement
    "loosening": "loosening / dislodgement",
    "dislodgement": "loosening / dislodgement",
    "dislodge": "loosening / dislodgement",
    "loosening and dislodgement": "loosening / dislodgement",
    "loosening or dislodgement": "loosening / dislodgement",
    # deterioration of anti-corrosion function
    "anti-corrosion function deterioration": "deterioration of anti-corrosion function",
    "deterioration of corrosion protection": "deterioration of anti-corrosion function",
    "corrosion protection degradation": "deterioration of anti-corrosion function",
    # bearing dysfunction
    "bearing functional failure": "bearing dysfunction",
    "bearing malfunction": "bearing dysfunction",
    "bearing failure": "bearing dysfunction",
    # soil/debris clogging
    "soil clogging": "soil/debris clogging",
    "debris clogging": "soil/debris clogging",
    "clogging": "soil/debris clogging",
    # pavement defect
    "pavement abnormality": "pavement defect",
    "road surface unevenness": "pavement defect",
    "pavement damage": "pavement defect",
    # scour
    "scouring": "scour",
    "erosion": "scour",
    "scour and erosion": "scour",
    # other
    "soil accumulation": "other",
    "sediment accumulation": "other",
    "misc": "other",
}

DAMAGE_LOC_NORMALIZE = {
    # main girder
    "main beam": "main girder",
    "girder": "main girder",
    "stringer": "main girder",
    "pc girder": "main girder",
    "rc girder": "main girder",
    "steel girder": "main girder",
    "i-girder": "main girder",
    "i girder": "main girder",
    # cross girder
    "cross beam": "cross girder",
    "crossbeam": "cross girder",
    "cross-beam": "cross girder",
    "end cross girder": "cross girder",
    "intermediate cross girder": "cross girder",
    # deck slab
    "floor slab": "deck slab",
    "slab": "deck slab",
    "rc deck slab": "deck slab",
    "pc deck slab": "deck slab",
    "steel deck slab": "deck slab",
    # vertical wall
    "abutment vertical wall": "vertical wall",
    "vertical wall of the abutment": "vertical wall",
    "abutment": "vertical wall",
    # wing wall
    "wing retaining wall": "wing wall",
    # parapet
    "breast wall": "parapet",
    # pier
    "column section": "pier",
    "wall section": "pier",
    "beam section": "pier",
    "column / wall section": "pier",
    "pier body": "pier",
    "pier beam": "pier",
    "pier column": "pier",
    # bearing
    "bearing body": "bearing",
    "rubber bearing": "bearing",
    "steel bearing": "bearing",
    "bearing section": "bearing",
    # expansion joint
    "expansion gap": "expansion joint",
    # drainage facility
    "drainage basin": "drainage facility",
    "drainage pipe": "drainage facility",
    "drainage device": "drainage facility",
    "drainage ditch": "drainage facility",
    "drainage channel": "drainage facility",
    # railing / guardrail
    "handrail": "railing",
    "barrier": "guardrail",
    "guard rail": "guardrail",
    "guard pipe": "guardrail",
    "vehicle barrier": "guardrail",
    # curb
    "curb stone": "curb",
    "kerb": "curb",
    "wheel guard": "curb",
    # attached equipment
    "attached accessory": "attached equipment",
    "accessory": "attached equipment",
    "attached structure": "attached equipment",
    "attachment": "attached equipment",
    # unseating prevention system
    "unseating prevention device": "unseating prevention system",
    "fall prevention system": "unseating prevention system",
    "displacement limiter": "unseating prevention system",
    # other
    "anchor bolt": "other",
    "pc anchorage": "other",
    "bearing seat mortar": "other",
    "truss joint": "other",
    "arch rib": "other",
    "external cable": "other",
    "partition wall": "other",
    "pedestal concrete": "other",
}


def normalize_label(category: str, raw: str) -> str | None:
    """
    Normalize a VLM-emitted label to a canonical English label.

    Returns the canonical label if `raw` matches; otherwise returns None.
    Matching is case-insensitive and ignores surrounding whitespace.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    labels = ALL_LABEL_SETS.get(category, [])
    norm_map = {
        "kenzenudo":   KENZENUDO_NORMALIZE,
        "taisaku":     TAISAKU_NORMALIZE,
        "damage_type": DAMAGE_TYPE_NORMALIZE,
        "damage_loc":  DAMAGE_LOC_NORMALIZE,
    }.get(category, {})

    # Direct match (preserves casing for kenzenudo/taisaku).
    if s in labels:
        return s
    if s in norm_map:
        return norm_map[s]

    # Case-insensitive / lowercase fallback.
    s_lower = s.lower()
    for canonical in labels:
        if s_lower == canonical.lower():
            return canonical
    for k, v in norm_map.items():
        if s_lower == k.lower():
            return v
    return None
