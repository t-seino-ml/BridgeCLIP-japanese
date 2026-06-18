# -*- coding: utf-8 -*-
"""
Unified label loader. Selects JA or EN label definitions based on the
`LABEL_LANG` environment variable.

Usage:
    from classification.data.labels import (
        ALL_LABEL_SETS, NUM_CLASSES,
        KENZENUDO_LABELS, TAISAKU_LABELS,
        DAMAGE_TYPE_LABELS, DAMAGE_LOC_LABELS,
    )

Set `LABEL_LANG=en` in the environment before launching Python to use English
canonical labels (I/II/III/IV, main girder, cracking, etc.). Default: `ja`.

This adapter intentionally imports the same module-level names from either
`label_definitions` or `label_definitions_en`, so downstream code can stay
language-agnostic.
"""

from __future__ import annotations

import os

LABEL_LANG = os.environ.get("LABEL_LANG", "ja").lower()

if LABEL_LANG == "en":
    from classification.data.label_definitions_en import (  # noqa: F401
        ALL_LABEL_SETS,
        KENZENUDO_LABELS,
        TAISAKU_LABELS,
        DAMAGE_TYPE_LABELS,
        DAMAGE_LOC_LABELS,
        NUM_CLASSES,
    )
elif LABEL_LANG in ("ja", "jp", ""):
    from classification.data.label_definitions import (  # noqa: F401
        ALL_LABEL_SETS,
        KENZENUDO_LABELS,
        TAISAKU_LABELS,
        DAMAGE_TYPE_LABELS,
        DAMAGE_LOC_LABELS,
        NUM_CLASSES,
    )
else:
    raise ValueError(
        f"Unknown LABEL_LANG={LABEL_LANG!r}. Must be 'ja' or 'en'."
    )
