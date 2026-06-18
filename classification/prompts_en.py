# -*- coding: utf-8 -*-
"""
English classification prompts (parallel to prompts.py).

Used by the *_en.py VLM model files to prompt VLMs in English and parse English
JSON keys in the response. CLIP zero-shot also uses CLIP_TEMPLATES from here.
"""

from classification.data.label_definitions_en import (
    KENZENUDO_LABELS,
    TAISAKU_LABELS,
    DAMAGE_TYPE_LABELS,
    DAMAGE_LOC_LABELS,
)

# ────────────────────────────── Label option strings ─────────────────────────

_KENZENUDO_OPTIONS   = " / ".join(KENZENUDO_LABELS)
_TAISAKU_OPTIONS     = " / ".join(TAISAKU_LABELS)
_DAMAGE_TYPE_OPTIONS = ", ".join(DAMAGE_TYPE_LABELS)
_DAMAGE_LOC_OPTIONS  = ", ".join(DAMAGE_LOC_LABELS)

# ────────────────────────────── Main prompts ─────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in road-bridge inspection. Analyze the provided bridge "
    "inspection photograph and respond in the specified format."
)

USER_PROMPT = f"""Examine the following bridge inspection photograph and classify it in four categories.
Respond ONLY in the JSON format below — do not include any other text.

```json
{{
  "soundness_rating": "<one option>",
  "measure_classification": "<one option>",
  "damage_type": ["<one or more options>"],
  "damage_location": ["<one or more options>"]
}}
```

[Soundness rating options] (choose exactly one)
{_KENZENUDO_OPTIONS}
- I:   No damage, or only minor damage.
- II:  Preventive maintenance stage (damage exists but is not urgent).
- III: Early-action stage (damage is progressing; early response required).
- IV:  Emergency-action stage (immediate response required).

[Measure classification options] (choose exactly one)
{_TAISAKU_OPTIONS}
- A:  No action required.
- B:  Preventive maintenance (ordinary upkeep before the next inspection).
- C1: Early action (repair within approximately 5 years).
- C2: Urgent action (repair as soon as possible).
- E1: Detailed investigation required.
- E2: Urgent detailed investigation required.
- M:  Can be addressed by maintenance work.
- S1: Detailed investigation (structural safety).
- S2: Detailed investigation (urgent).

[Damage type options] (choose one or more)
{_DAMAGE_TYPE_OPTIONS}

[Damage location options] (choose one or more)
{_DAMAGE_LOC_OPTIONS}

For any category that cannot be determined from the image, provide the most likely estimate.
"""

# ────────────────────────────── Few-shot example (optional) ──────────────────

FEW_SHOT_EXAMPLE = """
[Example output]
```json
{
  "soundness_rating": "II",
  "measure_classification": "C1",
  "damage_type": ["cracking", "water leakage / efflorescence"],
  "damage_location": ["main girder", "deck slab"]
}
```
"""

USER_PROMPT_WITH_EXAMPLE = USER_PROMPT + FEW_SHOT_EXAMPLE

# ────────────────────────────── JSON key mapping ─────────────────────────────
# Internal category id → English JSON key expected in VLM responses.

JSON_KEYS = {
    "kenzenudo":   "soundness_rating",
    "taisaku":     "measure_classification",
    "damage_type": "damage_type",
    "damage_loc":  "damage_location",
}

# ────────────────────────────── CLIP text templates ──────────────────────────
# Ensemble-of-prompts: multiple templates per class, averaged at encoding time.

CLIP_TEMPLATES = {
    "kenzenudo": {
        label: [
            f"a bridge inspection photograph with soundness rating {label}",
            f"a road bridge with soundness rating {label}",
            f"bridge damage image, soundness rating {label}",
        ]
        for label in KENZENUDO_LABELS
    },
    "taisaku": {
        label: [
            f"a bridge inspection photograph with measure classification {label}",
            f"a road bridge with measure classification {label}",
            f"bridge damage image, measure classification {label}",
        ]
        for label in TAISAKU_LABELS
    },
    "damage_type": {
        label: [
            f"a photograph of {label} in a bridge",
            f"a road bridge showing {label}",
            f"a bridge inspection photograph where {label} is observed",
        ]
        for label in DAMAGE_TYPE_LABELS
    },
    "damage_loc": {
        label: [
            f"a damage photograph of the {label} of a bridge",
            f"the {label} section of a road bridge",
            f"a bridge inspection photograph where damage is observed on the {label}",
        ]
        for label in DAMAGE_LOC_LABELS
    },
}
