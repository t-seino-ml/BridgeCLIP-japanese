# BridgeCLIP: Fine-tuned CLIP for Bridge Inspection Image Classification and Retrieval

A fine-tuned [OpenCLIP ViT-B/32](https://github.com/mlfoundations/open_clip) model for bridge inspection image classification and image-text retrieval, trained on the Japanese national road facility inspection database (xROAD). Both Japanese-caption and English-caption variants are supported.

## Overview

- **Proposed method**: CLIP fine-tuning + k-NN classification (k=10, cosine similarity over the fine-tuned image features)
- **Additional ablations**: CLIP-FT + supervised linear / finetune heads (C/D/E/F), text–image dot product (A/B), top-1 fair comparison for weighted-BCE variants
- **Baselines**: ResNet50, ViT-B/16 (supervised: linear probe / full finetune / weighted BCE), GPT-4o, Qwen3-VL-8B, InternVL3-8B, Llama3.2-Vision (zero-shot VLMs)
- **Tasks**:
  - 4-category multi-label classification:
    - Soundness rating (Ⅰ/Ⅱ/Ⅲ/Ⅳ, single-label)
    - Measure classification (A/B/C1/C2/E1/E2/M/S1/S2, single-label)
    - Damage type (15 classes, multi-label)
    - Damage location (20 classes, multi-label)
  - Image–text retrieval (I2T / T2I) — paired by row in the validation set
  - Attribute-based retrieval — counts a gallery item as a hit if it shares at least one label with the query (AttrMatch@k, attribute-based mAP, NDCG@10)

## Performance (Japanese captions, val=2,679)

Macro-F1 per category, plus the mean. The proposed CLIP fine-tuned k-NN reaches a competitive 0.4151 while keeping the highest macro-precision on damage-location.

| Model | Soundness | Measure | Damage type | Damage loc. | **Mean macro-F1** |
|---|---|---|---|---|---|
| ResNet50 weighted finetune | **0.6106** | 0.3293 | **0.5379** | 0.4640 | **0.4855** |
| ViT weighted finetune | 0.5502 | **0.3593** | 0.5310 | 0.4368 | 0.4693 |
| ResNet50 finetune | 0.5515 | 0.3416 | 0.5005 | 0.3793 | 0.4432 |
| ViT finetune | 0.5519 | 0.3502 | 0.4777 | 0.3592 | 0.4347 |
| **Proposed: CLIP fine-tuned + k-NN** | 0.4170 | 0.2415 | 0.5115 | **0.4904** | **0.4151** |
| Proposed: CLIP fine-tuned + linear classifier (D) | 0.3879 | 0.2273 | 0.4939 | 0.5021 | 0.4028 |
| Proposed: CLIP fine-tuned + supervised + weighted BCE (F) | 0.4617 | 0.3082 | 0.5363 | 0.4721 | 0.4446 |
| GPT-4o (zero-shot) | 0.1550 | 0.1593 | 0.2967 | 0.2783 | 0.2223 |
| Qwen3-VL-8B (zero-shot) | 0.1102 | 0.1394 | 0.2195 | 0.1610 | 0.1575 |
| InternVL3-8B (zero-shot) | 0.0668 | 0.0235 | 0.2043 | 0.1291 | 0.1059 |
| Llama3.2-Vision (zero-shot) | 0.0042 | 0.0005 | 0.0004 | 0.0011 | 0.0015 |

### Feature-only comparison: CLIP-FT contrastive features vs ImageNet/LAION baselines

Same linear-probe head, only the backbone pre-training differs:

| Backbone | Linear-probe Mean macro-F1 |
|---|---|
| CLIP-base (LAION-2B) | 0.3304 |
| **CLIP-FT (proposed)** | **0.4028 (+21.9% relative)** |
| ResNet50 (ImageNet) | 0.3278 |
| ViT-B/16 (ImageNet) | 0.3496 |

→ Domain contrastive learning gives a clear +21.9% relative gain over LAION-2B base CLIP at the linear-probe level.

### Retrieval (val=2,679)

| Direction | Base CLIP | **CLIP fine-tuned (epoch 5)** |
|---|---|---|
| I→T R@1  | 0.0004 | **0.0370** |
| I→T R@10 | 0.0037 | **0.2251** |
| T→I R@1  | 0.0011 | **0.0493** |
| T→I R@10 | 0.0045 | **0.2437** |

### Attribute-based retrieval (Text → Image, AttrMatch@1)

A hit is counted if the retrieved gallery image shares at least one label with the query text.

| Category | Base CLIP | **CLIP fine-tuned** | Relative gain |
|---|---|---|---|
| Soundness | 0.3502 | **0.7043** | ×2.01 |
| Measure | 0.3101 | **0.6281** | ×2.03 |
| Damage type | 0.1248 | **0.6667** | **×5.34** |
| Damage loc. | 0.1252 | **0.6397** | **×5.11** |

Damage type / location see a 5× gain that is impossible for the ResNet/ViT baselines (no text–image alignment).

## Model checkpoints

Fine-tuned checkpoints are released on the Hugging Face Hub:

- **Japanese captions**: [Seino404/bridge-inspection-clip](https://huggingface.co/Seino404/bridge-inspection-clip)
- **English captions**: [Seino404/bridge-inspection-clip-en](https://huggingface.co/Seino404/bridge-inspection-clip-en) (uploaded after the EN sweep finishes)

## Dataset

Image-text pairs from the Japanese national road facility inspection database (xROAD).

| Split | Samples |
|---|---|
| Train | 130,930 |
| Validation | 2,679 |
| k-NN database (all 4 categories valid) | 90,987 |

Image paths are removed from the redistributed CSVs; the image files themselves are not redistributed because of dataset licensing. Code is provided to rebuild CSVs from the inspection database; see `Pretreatment/`.

## Quick start

### Installation

```bash
uv sync                 # or: pip install -e .
```

### Reproducing the proposed CLIP-FT k-NN

```bash
# 1. Fine-tune CLIP (or download the checkpoint from HF Hub)
bash classification/train_clip.sh

# 2. k-NN classification with the FT checkpoint
python -m classification.models.clip_finetuned_knn \
  --train_csv classification/results/unified_train_user.csv \
  --val_csv   classification/results/unified_val_user.csv \
  --ckpt_dir  logs_classification/bridgeclip_vitb32_unified/checkpoints \
  --out       classification/results/clip_finetuned_knn_preds.csv \
  --k 10

# 3. Evaluate
python -m classification.evaluate \
  --pred classification/results/clip_finetuned_knn_preds.csv \
  --out  classification/results/clip_finetuned_knn_metrics.json
```

### Re-running all baselines (4 GPU parallel)

```bash
# Japanese
IMAGE_ROOT=/path/to/images bash classification/run_all_classifiers_parallel.sh

# English
IMAGE_ROOT=/path/to/images bash classification/run_all_classifiers_parallel_en.sh
```

### Attribute-based retrieval (used in §3 of the paper)

```bash
python -m classification.retrieval_eval \
  --val_csv  classification/results/unified_val_user.csv \
  --ckpt_dir logs_classification/bridgeclip_vitb32_unified/checkpoints \
  --out      classification/results/clip_finetuned_retrieval_v2.json
```

Outputs `AttrMatch@k`, attribute-based `mAP`, and `NDCG@10` for both I2I and T2I, across all four categories.

## Layout

```
.
├── classification/
│   ├── data/
│   │   ├── dataset.py              # multi-label dataset with --image_root remap
│   │   ├── label_definitions.py    # Japanese labels (Ⅰ/Ⅱ/Ⅲ/Ⅳ, etc.)
│   │   ├── label_definitions_en.py # English labels (I/II/III/IV, etc.)
│   │   └── labels.py               # adapter: LABEL_LANG=ja/en switches the two
│   ├── models/
│   │   ├── clip_finetuned_knn.py   # proposed method (k-NN over CLIP-FT features)
│   │   ├── clip_classifier.py      # CLIP backbone + supervised heads (C/D/E/F)
│   │   ├── clip_zeroshot.py        # text-dot product (A/B), supports --ckpt for FT
│   │   ├── resnet50.py, vit.py, ...
│   │   ├── gpt4o.py / qwen3vl.py / internvl.py / llama.py  # VLM zero-shot
│   │   └── *_en.py                 # English-prompt variants for the VLMs
│   ├── train.py                    # supervised training entry point
│   ├── train_clip.sh               # CLIP fine-tuning launcher
│   ├── evaluate.py / evaluate_en.py
│   ├── retrieval_eval.py           # standard + attribute-based retrieval
│   ├── plot_confusion_matrices.py / *_en.py
│   ├── predict_top1.py             # top-1 fair comparison for weighted variants
│   └── run_*.sh                    # batch run scripts (single GPU / 3 GPU / 4 GPU)
├── Pretreatment/                   # text & label preprocessing for xROAD data
├── eval_retrieval.py               # standalone retrieval eval (used by sweep)
├── pick_best_ckpt.py               # picks best epoch from results.jsonl
├── run_sweep_*.sh                  # data-scaling sweep entry points
├── pyproject.toml                  # uv project file
└── README.md
```

## Caveats noted in our experiments

- The validation set shares **89.4% of bridges with the training set**; the current split does not isolate bridge-level generalization. Claims about unseen-bridge generalization require a leave-one-bridge-out (LOBO) re-split.
- The maintained CLIP-FT checkpoint is `logs_classification/bridgeclip_vitb32_unified/checkpoints/epoch_5.pt` (val_loss = 2.5242, downloaded from Hugging Face after the original was deleted to free disk).
- Older buggy supervised baselines (mean exact-match accuracy ≈ 0.38) that appeared in earlier drafts were caused by (i) a silent black-image fallback in `BridgeInspectionDataset` when image paths were wrong, and (ii) BCE-on-one-hot loss for single-label categories. Both bugs have been fixed; the numbers in this README are the post-fix re-trained values.

## Citation

Paper under preparation.

## License

Code: MIT. The dataset itself is not redistributed (see Dataset section).
