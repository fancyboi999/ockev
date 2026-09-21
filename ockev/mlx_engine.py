"""
Native Apple Silicon MLX Decision Engine for Ockev.
Runs on Apple M-series chips via Metal GPU with zero PyTorch/CUDA dependencies.
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
except ImportError:
    mx = None

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

def sanitize_user_tokens(tok, text: str) -> List[int]:
    clean = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tok.encode(clean, add_special_tokens=False)

class MLXOckevEngine:
    """
    Apple Silicon MLX implementation of Ockev.
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-1.5B"):
        if mx is None:
            raise RuntimeError("MLX is not installed. Install via: pip install mlx mlx-lm")
        
        self.model, self.tokenizer = mlx_load(model_name_or_path)
        self.head_dim = 256
        self.hidden_size = self.model.model.embed_tokens.weight.shape[1]
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.q_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)

    def evaluate(self, state: Any, question_spec: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        inst = question_spec.get("instructions", "")
        criteria = question_spec.get("criteria", {})
        opt_keys = list(criteria.keys())
        
        state_str = json.dumps(state, ensure_ascii=False) if isinstance(state, dict) else str(state)
        tok = self.tokenizer
        state_tokens = sanitize_user_tokens(tok, state_str)[:384]
        
        s_id = tok.convert_tokens_to_ids(SPECIAL[0])
        q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
        
        full_ids = [s_id] + state_tokens + [q_id] + sanitize_user_tokens(tok, inst)
        opt_end_indices = []
        
        for k in opt_keys:
            desc = criteria[k]
            tokens = [o_id] + sanitize_user_tokens(tok, f"Option {k}: {desc}") + [c_id]
            full_ids.extend(tokens)
            opt_end_indices.append(len(full_ids) - 1)
            
        decide_idx = len(full_ids)
        full_ids.append(d_id)
        
        input_ids = mx.array([full_ids])
        hidden_states = self.model.model(input_ids)
        
        h_decide = hidden_states[0, decide_idx, :]
        h_opts = mx.stack([hidden_states[0, idx, :] for idx in opt_end_indices])
        
        q_vec = self.q_proj(h_decide)
        k_mat = self.k_proj(h_opts)
        
        logits = mx.matmul(k_mat, q_vec) * self.scale
        probs_mx = mx.softmax(logits, axis=-1)
        probs = [float(p) for p in probs_mx]
        
        mx.eval(probs_mx)
        dt_ms = (time.time() - t0) * 1000
        
        prob_dict = {k: round(probs[i], 4) for i, k in enumerate(opt_keys)}
        best_idx = probs.index(max(probs))
        choice = opt_keys[best_idx]
        sorted_probs = sorted(probs, reverse=True)
        conf = round(sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0), 4)
        
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": prob_dict,
            "confidence": conf,
            "latency_ms": round(dt_ms, 1),
            "backend": "apple_silicon_mlx"
        }
