"""
Translate Japanese bridge-inspection texts to English with GPT-4o.

- Collects unique texts from train_clean_base.csv and val_clean_base.csv (a superset
  of all subset CSVs), translates each with GPT-4o using a fixed MLIT-based
  terminology dictionary, and caches results in translation_dict.json.
- Resumable: re-running skips already-translated entries.
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DICT = Path(__file__).resolve().parent / "translation_dict.json"
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a professional translator specialized in Japanese→English translation of bridge inspection reports, following MLIT (Ministry of Land, Infrastructure, Transport and Tourism) Bridge Periodic Inspection Guidelines terminology.

## Output rule
Output ONLY a JSON object: {"translation": "<english sentence>"}. No extra text.

## Symbols to preserve EXACTLY as-is (do NOT translate)
- Soundness rating: Ⅰ, Ⅱ, Ⅲ, Ⅳ (keep the Roman numeral characters)
- Measure classification: A, B, C1, C2, E1, E2, M, S1, S2

## Sentence templates (strict)
- "<PART>において<DAMAGE>が見られます、健全度判定は<X>です、部材毎の対策区分判定は<Y>です。"
  → "<DAMAGE> is observed in the <PART>. The soundness rating is <X>. The component-wise measure classification is <Y>."
- "<PART>において<DAMAGE>が見られます。"
  → "<DAMAGE> is observed in the <PART>."

The damage phrase starts the English sentence and is capitalized. Use "in the" before the part name. Do not paraphrase the template; keep sentence structure consistent across all inputs.

## Multi-value handling (when parts or damages are joined by 、 or ,)
- Join multiple damages with " and " (e.g., "corrosion and deterioration of anti-corrosion function").
- Join multiple parts with " and " (e.g., "main girder and cross girder").
- If the item itself contains a slash-style compound (e.g., 剥離・鉄筋露出 → spalling / rebar exposure), keep the slash form — this is ONE damage item, not two.

## Subject–verb agreement (strict)
Count the distinct damage items listed in the input (items separated by 、 or , — NOT by ・ or / which are part of a single compound term).
- Exactly 1 damage item → "<Damage> is observed in the <part>."
- 2 or more damage items → "<Damages joined by 'and'> are observed in the <part>."
Examples:
- "腐食が見られます" (1 item) → "Corrosion is observed ..."
- "腐食、変形・欠損が見られます" (2 items: 腐食 and 変形・欠損) → "Corrosion and deformation / defect ARE observed ..."
- "うき、漏水が見られます" (2 items) → "Delamination and water leakage ARE observed ..."
- "ひびわれ、剥離、うきが見られます" (3 items) → "Cracking, spalling, and delamination are observed ..."
- "漏水・滞水が見られます" (1 item — slash-compound) → "Water leakage / pooling is observed ..."

## Position identifiers (strip them)
Japanese part names are often prefixed with position codes such as A1, A2, A1G2, P1, P2, G1, G2, M1, S1S, A1S, etc. (abutment/pier/girder/span indices).
Drop these position codes entirely — do NOT render them in English. The label-relevant information is the component type only.
Examples:
- "A1橋台竪壁" → "vertical wall of the abutment"  (drop "A1")
- "A1G2支承" → "bearing"  (drop "A1G2")
- "A1,A2伸縮装置" → "expansion joint"  (drop "A1,A2")
- "1排水ます" → "drainage basin"  (drop leading "1")
- "A1S台座コンクリート" → "pedestal concrete"  (drop "A1S")
However, Roman-numeral soundness ratings (Ⅰ, Ⅱ, Ⅲ, Ⅳ) and measure-classification codes (A, B, C1, C2, E1, E2, M, S1, S2) inside their dedicated clauses must still be preserved verbatim.

## Orthographic normalization
Normalize the following variations to the canonical English term (treat them as the same word):
- half-width vs full-width (PC / ＰＣ, ｱﾝｶｰﾎﾞﾙﾄ / アンカーボルト, ・ / ･, 、 / ,)
- typos: 堅壁 → 竪壁 (vertical wall); 添加物 → 添架物 (attached equipment); 床板 → 床版 (deck slab); 鋪装 → 舗装 (pavement); 沓座ﾓﾙﾀﾙ → 沓座モルタル (bearing seat mortar)
- code prefixes like "Ct_頂版", "Sw_側壁", "Ww_袖擁壁", "Fg_地覆", "Mg_主桁", "Pm_舗装", "Iw_隔壁", "Aw_翼壁" — strip the prefix and use the canonical translation.
- "橋台（竪壁）", "橋台 竪壁", "橋台竪壁", "A1竪壁" → "vertical wall of the abutment"
- Parenthetical clarifications like "その他（土砂堆積）" → "other (soil accumulation)".

## Canonical terminology

### Damage types (MLIT 16 categories + common variants)
- 腐食 → corrosion
- ひびわれ → cracking
- 亀裂 → cracking (fatigue crack when in steel context)
- 破断 → fracture
- 防食機能の劣化 → deterioration of anti-corrosion function
- 床版ひびわれ → deck slab cracking
- 漏水・滞水 → water leakage / pooling
- 漏水・遊離石灰 → water leakage / efflorescence
- 漏水 → water leakage
- 遊離石灰 → efflorescence
- 滞水 → water pooling
- うき → delamination
- 剥離・鉄筋露出 → spalling / rebar exposure
- 剥離 → spalling
- 鉄筋露出 → rebar exposure
- 変形・欠損 → deformation / defect
- 変形 → deformation
- 欠損 → defect
- ゆるみ・脱落 → loosening / dislodgement
- 脱落 → dislodgement
- 異常な音・振動 → abnormal noise / vibration
- 異常なたわみ → abnormal deflection
- 路面の凹凸 → road surface unevenness
- 舗装の異常 → pavement defect
- 舗装の異常、路面の凹凸 → pavement defect and road surface unevenness
- 定着部の異常 → anchorage defect
- 変色・劣化 → discoloration / deterioration
- 支承部の機能障害 → bearing dysfunction
- 沈下・移動・傾斜 → settlement / displacement / inclination
- 補修・補強材の損傷 → damage to repair/reinforcement material
- 土砂詰まり → soil/debris clogging
- 土砂堆積 → soil accumulation
- 遊間の異常 → expansion gap defect
- 異常な音・振動、路面の凹凸 → abnormal noise / vibration and road surface unevenness
- その他 → other

### Bridge components (canonical English)
- 主桁 → main girder
- 横桁 → cross girder
- 縦桁 → stringer
- 床版 → deck slab
- 伸縮装置 → expansion joint
- 支承 → bearing
- 支承本体 → bearing body
- 支承部 → bearing section
- アンカーボルト → anchor bolt
- 沓座モルタル → bearing seat mortar
- 高欄 → railing
- 防護柵 → guardrail
- 地覆 → curb
- 縁石 → curb stone
- 舗装 → pavement
- 排水管 → drainage pipe
- 排水ます → drainage basin
- 排水桝 → drainage basin
- 排水施設 → drainage facility
- 排水装置 → drainage device
- 橋台 → abutment
- 橋脚 → pier
- フーチング → footing
- 底版 → bottom slab
- 頂版 → top slab
- 竪壁 → vertical wall
- 堅壁 → vertical wall
- 胸壁 → parapet
- 側壁 → side wall
- 翼壁 → wing wall
- 袖擁壁 → wing retaining wall
- 落橋防止システム → unseating prevention system
- PC定着部 → PC anchorage
- 外ケーブル → external cable
- 添架物 → attached equipment
- 点検施設 → inspection facility
- 遮音施設 → sound barrier
- 遮音壁 → sound barrier wall
- 梁部 → beam section
- 梁 → beam
- 柱部 → column section
- 柱部・壁部 → column / wall section
- 対傾構 → sway bracing
- 下横構 → lower lateral bracing
- 上横構 → upper lateral bracing
- 中央分離帯 → median strip
- 格点 → truss joint
- アーチリブ → arch rib
- アーチ補剛桁 → arch stiffening girder
- アーチ支柱 → arch strut
- 斜材 → diagonal member
- 補剛桁 → stiffening girder
- 目地部 → joint section
- 台座コンクリート → pedestal concrete
- 隔壁 → partition wall
- 主桁ゲルバー部 → Gerber section of the main girder
- 橋脚梁部 → pier beam
- 縦断方向連結部 → longitudinal connection
- 支承部その他 → other bearing section
- 橋台その他 → other abutment part
- 橋脚その他 → other pier part
- 上部構造その他 → other superstructure part
- 上部工その他 → other superstructure part
- 基礎その他 → other substructure part
- 排水施設その他 → other drainage facility
- 溝橋その他 → other culvert part
- 路上 → road surface area

### Usage notes
- Use lowercase for generic English component names ("main girder", not "Main Girder"), except at the start of a sentence.
- Do not include the Japanese original in the output.
- Keep the exact English template wording so downstream CSVs are consistent.
"""


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp932"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def collect_unique_texts(root: Path, extra_csvs: list[Path] | None = None) -> list[str]:
    files = [
        root / "train_subsets_item" / "train_clean_base.csv",
        root / "val_item_data" / "val_clean_base.csv",
    ]
    if extra_csvs:
        files.extend(extra_csvs)
    uniq: set[str] = set()
    for f in files:
        if not f.exists():
            print(f"[warn] not found: {f}")
            continue
        df = read_csv_robust(f)
        if "text" not in df.columns:
            print(f"[warn] no text col: {f}")
            continue
        vals = df["text"].dropna().astype(str).map(str.strip)
        uniq |= set(vals.unique())
    uniq.discard("")
    return sorted(uniq)


async def translate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    text: str,
    max_retries: int = 6,
) -> str:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                data = json.loads(content)
                en = str(data.get("translation", "")).strip()
                if not en:
                    raise ValueError("empty translation")
                return en
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = min(60.0, (2 ** attempt) + random.random())
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")


def save_dict(path: Path, d: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def run(args) -> None:
    out_path = Path(args.out)
    extra = [Path(p) for p in (args.extra_csv or [])]
    texts = collect_unique_texts(ROOT, extra_csvs=extra)
    print(f"[info] unique texts: {len(texts)}")

    if out_path.exists():
        translations: dict[str, str] = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[info] loaded existing translations: {len(translations)}")
    else:
        translations = {}

    todo = [t for t in texts if t not in translations]
    print(f"[info] todo: {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"[info] limited to first {len(todo)}")

    if not todo:
        print("[info] nothing to translate.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    save_every = args.save_every
    done_since_save = 0

    async def task(t: str):
        en = await translate_one(client, sem, t)
        return t, en

    coros = [task(t) for t in todo]
    for fut in atqdm.as_completed(coros, total=len(coros), desc="translating"):
        try:
            jp, en = await fut
        except Exception as e:
            print(f"[error] {e}")
            continue
        translations[jp] = en
        done_since_save += 1
        if done_since_save >= save_every:
            save_dict(out_path, translations)
            done_since_save = 0

    save_dict(out_path, translations)
    print(f"[done] saved {len(translations)} translations to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_DICT))
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="translate only first N (for smoke test)")
    ap.add_argument("--extra_csv", action="append", default=None,
                    help="additional CSV(s) with a 'text' column to include as sources (repeatable)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
