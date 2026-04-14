from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # open_clip_train checkpoints usually have: {'epoch','name','state_dict','optimizer','scaler',...}
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        # sometimes the ckpt itself is the state_dict
        sd = ckpt
    return sd

def _strip_prefix(k: str, prefixes: Tuple[str, ...]) -> str:
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k

def load_openclip_checkpoint_into_model(model: nn.Module, ckpt_path: str | Path, device: str = "cpu") -> Dict[str, Any]:
    """
    Loads open_clip_train checkpoint into an open_clip model safely.
    - removes only safe prefixes like 'module.' or 'model.'.
    - NEVER strips 'visual.' or changes key hierarchy.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = _extract_state_dict(ckpt)

    # Safe prefix stripping ONLY
    safe_prefixes = ("module.", "model.", "clip.", "open_clip.")
    new_sd = {}
    for k, v in sd.items():
        nk = _strip_prefix(k, safe_prefixes)
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    info = {
        "missing": missing,
        "unexpected": unexpected,
        "ckpt_keys": list(ckpt.keys()) if isinstance(ckpt, dict) else [],
    }
    return info