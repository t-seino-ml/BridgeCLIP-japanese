# EN 版実験 実行手順書（サーバ用）

JA 版で完成した全実験を English 版で再生するための手順。**サーバ側 1.5 日 で完了予定**。

## 前提

- このリポジトリは JA 用パイプラインを `LABEL_LANG` 環境変数で EN にも切り替え可能になっている
- EN ラベル定義: `classification/data/label_definitions_en.py`
- EN プロンプト: `classification/prompts_en.py`
- EN ラベル付き CSV: `classification/results/unified_{train,val}_user_en.csv`（既存）
- 画像実体: `~/Flash_Storage/CLIP/images/`（共有）

## 0. CPU 上での事前確認（ローカル PC で実施済み）

```bash
# JA / EN 両方でラベルアダプタが動くかチェック
python -c "from classification.data.labels import ALL_LABEL_SETS; print(ALL_LABEL_SETS['kenzenudo'])"
LABEL_LANG=en python -c "from classification.data.labels import ALL_LABEL_SETS; print(ALL_LABEL_SETS['kenzenudo'])"
```

## 1. EN CLIP-FT 再学習（30〜60 分 / 1 GPU）

```bash
cd ~/Flash_Storage/CLIP
bash classification/retrain_clip_short_en.sh
# 出力: logs_classification_en/clip_en_v2/checkpoints/epoch_{1..10}.pt
```

ベストエポック確認：
```bash
uv run python - <<'PY'
import json
from pathlib import Path
ck = Path('logs_classification_en/clip_en_v2/checkpoints')
ent = []
with open(ck/'results.jsonl') as f:
    for l in f:
        d = json.loads(l)
        ent.append((d['clip_val_loss'], d['epoch']))
ent.sort()
for l, e in ent[:5]:
    print(f'epoch {e}: val_loss={l:.4f}')
PY
```

## 2. EN ResNet50/ViT × 6 構成（4 GPU 並列、約 3〜4 時間）

```bash
PYTHON="uv run python" \
IMAGE_ROOT=$(pwd)/images \
bash classification/run_all_classifiers_parallel_en.sh
# 出力: classification/results_en/{resnet50,vit}_*_en_preds.csv
#       classification/results_en/comparison_en.json
```

## 3. EN CLIP-FT 派生 + text-dot（4 GPU 並列、約 1.5〜2 時間 + 5 分）

```bash
# 線形ヘッド系（C/D/E/F）
PYTHON="uv run python" \
IMAGE_ROOT=$(pwd)/images \
CLIP_CKPT_DIR=logs_classification_en/clip_en_v2/checkpoints \
bash classification/run_clip_classifier_parallel_en.sh

# text-dot（A/B）
PYTHON="uv run python" \
IMAGE_ROOT=$(pwd)/images \
CLIP_CKPT_DIR=logs_classification_en/clip_en_v2/checkpoints \
GPU=0 \
bash classification/run_clip_textdot_en.sh
```

## 4. EN CLIP-FT kNN 分類（best ckpt 使用、約 10 分）

```bash
LABEL_LANG=en uv run python -m classification.models.clip_finetuned_knn \
  --train_csv classification/results/unified_train_user_en.csv \
  --val_csv   classification/results/unified_val_user_en.csv \
  --ckpt_dir  logs_classification_en/clip_en_v2/checkpoints \
  --out       classification/results_en/clip_finetuned_knn_best_en_preds.csv \
  --k 10
```

評価：
```bash
LABEL_LANG=en uv run python -m classification.evaluate_en \
  --pred classification/results_en/clip_finetuned_knn_best_en_preds.csv \
  --out  classification/results_en/clip_finetuned_knn_best_en_metrics.json
```

## 5. EN 属性ベース検索評価 §2.4 相当（約 1 分）

```bash
# CLIP-base
LABEL_LANG=en uv run python -m classification.retrieval_eval \
  --val_csv classification/results/unified_val_user_en.csv \
  --out     classification/results_en/clip_base_retrieval_v2_en.json

# CLIP-FT (best epoch)
LABEL_LANG=en uv run python -m classification.retrieval_eval \
  --val_csv  classification/results/unified_val_user_en.csv \
  --ckpt_dir logs_classification_en/clip_en_v2/checkpoints \
  --out      classification/results_en/clip_finetuned_retrieval_v2_en.json
```

## 6. EN VLM zero-shot（既存 EN 予測 CSV を使用、評価のみ）

```bash
# 既存の `classification/results_en/{gpt4o,qwen3vl,internvl,llama}_en_preds.csv` は健全。
# evaluate_en で metrics JSON を再生成
LABEL_LANG=en uv run python - <<'PY'
import json
from pathlib import Path
from classification.evaluate_en import evaluate_model
for m in ['gpt4o','qwen3vl','internvl','llama']:
    p = f'classification/results_en/{m}_en_preds.csv'
    res = evaluate_model(p)
    Path(f'classification/results_en/{m}_en_metrics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved {m}_en_metrics.json')
PY
```

## 7. EN データスケーリング sweep §2.5 相当（4 GPU 並列、約 1.5〜2 時間）

```bash
# 7a. EN 用サブセット作成（10k〜120k + clean、約 1 分）
bash classification/make_subsets_en.sh

# 7b. sweep 学習（10 epoch × 13 サブセット、4 GPU 並列）
bash run_sweep_4gpu_queue_en.sh

# 7c. 各 best ckpt で評価
bash run_eval_best_4gpu_queue_en.sh
# 出力: eval_results_best_en_sweep/roadclip_*_*.json
```

## 8. EN 混同行列（CPU、約 1 分）

```bash
uv run python -m classification.plot_confusion_matrices_en \
  --out_dir classification/results_en/confusion_matrices
```

## 9. EN HuggingFace アップロード（10 分）

```bash
# best ckpt のエポック番号を確認してアップロードスクリプトに渡す
cd 000/huggingface_en

# 例: best が epoch 5 だった場合
CKPT_PATH=../../logs_classification_en/clip_en_v2/checkpoints/epoch_5.pt \
bash upload.sh Seino404 bridge-inspection-clip-en
```

---

## 合計所要時間

| Phase | 内容 | GPU | 時間 |
|---|---|---|---|
| 1 | EN CLIP-FT 再学習 (10 epoch) | 1 | 30〜60 分 |
| 2 | EN ResNet50/ViT × 6 構成 | 4 並列 | 3〜4 時間 |
| 3 | EN CLIP-FT 派生 + text-dot | 4 並列 | 1.5〜2 時間 |
| 4 | EN CLIP-FT kNN 分類 | 1 | 10 分 |
| 5 | EN 属性ベース検索 | 1 | 1 分 |
| 6 | EN VLM 評価（再評価のみ） | 0 | 1 分 |
| 7 | EN データスケーリング sweep | 4 並列 | 1.5〜2 時間 |
| 8 | EN 混同行列 | CPU | 1 分 |
| 9 | EN HF アップロード | 0 | 10 分 |
| | **合計** | | **約 7〜10 時間 GPU** |

---

## トラブルシューティング

- `LABEL_LANG=en` を必ず付ける（実行スクリプトは内部で `export LABEL_LANG=en` 済み）
- EN VLM 予測スクリプトを再実行したい場合は `classification/models/{gpt4o,llama,internvl,qwen3vl}_en.py` を使う（JA 版とは別ファイル）
- 万一 EN CLIP-FT ckpt が再度削除された場合の備えとして、HF にアップロード（Phase 9）を **学習完了直後** に実行することを推奨
