# -*- coding: utf-8 -*-
"""
Unified CLIP prompt loader. Selects JA or EN templates based on `LABEL_LANG`.

Usage:
    from classification.prompts_unified import CLIP_TEMPLATES
"""

from __future__ import annotations

import os

_LANG = os.environ.get("LABEL_LANG", "ja").lower()

if _LANG == "en":
    from classification.prompts_en import CLIP_TEMPLATES  # noqa: F401
else:
    from classification.prompts import CLIP_TEMPLATES  # noqa: F401
