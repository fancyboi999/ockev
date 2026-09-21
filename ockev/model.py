import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional

SPECIAL_TOKENS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

class OckevPointerHead(nn.Module):
    """
    Pointer readout head for zero-generation discriminative decisions.
    Extracts the query vector at the decision token and key vectors at option tokens,
    then evaluates their scaled dot-product similarity.
    """
    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        # h_decide: (B, H) -> q: (B, D)
        # h_opts: (B, N, H) -> k: (B, N, D)
        q = self.q_proj(h_decide).unsqueeze(-1)  # (B, D, 1)
        k = self.k_proj(h_opts)                   # (B, N, D)
        logits = torch.bmm(k, q).squeeze(-1) * self.scale  # (B, N)
        return logits

class OckevPointerModel(nn.Module):
    """
    Complete Ockev Decision Model combining a Causal LM backbone with a Pointer Head.
    """
    def __init__(self, base_model_name_or_path: str = "Qwen/Qwen2.5-1.5B", head_dim: int = 256):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        hidden_size = self.base_model.config.hidden_size
        self.pointer_head = OckevPointerHead(hidden_size, head_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decide_indices: torch.Tensor,
        opt_indices: List[List[int]]
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]
        batch_size = input_ids.size(0)
        
        # Gather decide hidden states
        h_decide = torch.stack([hidden_states[b, decide_indices[b], :] for b in range(batch_size)])
        
        # Gather option hidden states
        h_opts_list = []
        for b in range(batch_size):
            opts_for_b = torch.stack([hidden_states[b, idx, :] for idx in opt_indices[b]])
            h_opts_list.append(opts_for_b)
        h_opts = torch.stack(h_opts_list)
        
        logits = self.pointer_head(h_decide, h_opts)
        return logits
