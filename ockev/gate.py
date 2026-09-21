"""
OckevGate: High-level gatekeeper for verifying AI agent deliverables.
Eliminates conversational noise, discarded alternatives, and negative echoes.
"""

from typing import Dict, Any, Optional
import os

class OckevGate:
    def __init__(self, backend: str = "auto", model_name_or_path: str = "Qwen/Qwen2.5-1.5B"):
        self.backend = backend
        self.model_name = model_name_or_path
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        if self.backend == "auto":
            try:
                import mlx.core as mx
                from .mlx_engine import MLXOckevEngine
                self._engine = MLXOckevEngine(self.model_name)
                self.backend = "mlx"
                return
            except Exception:
                pass
            
            try:
                import torch
                from .model import OckevPointerModel
                self._engine = OckevPointerModel(self.model_name)
                self.backend = "pytorch"
                return
            except Exception:
                pass
            
            self.backend = "mock"
        elif self.backend == "mlx":
            from .mlx_engine import MLXOckevEngine
            self._engine = MLXOckevEngine(self.model_name)
        elif self.backend == "pytorch":
            from .model import OckevPointerModel
            self._engine = OckevPointerModel(self.model_name)

    def review_deliverable(
        self,
        candidate_text: str,
        task_context: Optional[str] = None,
        uncertain_margin: float = 0.15
    ) -> Dict[str, Any]:
        """
        Check if candidate deliverable adheres to the Final State Principle:
        - Describes adopted behavior and current facts directly.
        - Free from conversational residue (e.g. 'I didn't include pork', 'we tried X and failed').
        - Retains legitimate requirements, invariants, and architectural constraints.
        """
        question_spec = {
            "instructions": (
                "Determine whether the candidate text represents a clean, audience-facing final deliverable, "
                "or contains conversational residue, discarded alternatives, or internal development history."
            ),
            "criteria": {
                "pass": "Text cleanly describes adopted behavior, current facts, and valid constraints directly without conversational clutter.",
                "revise": "Text contains conversational apologies, discarded alternative explanations, negative echoes, or private scaffolding."
            }
        }
        
        state = {
            "candidate_text": candidate_text,
            "task_context": task_context or "General engineering deliverable review."
        }
        
        if self._engine is None:
            # Fallback heuristic / mock when neither MLX nor PyTorch is ready
            return self._heuristic_review(candidate_text)
            
        result = self._engine.evaluate(state, question_spec)
        probs = result.get("probabilities", {})
        p_pass = probs.get("pass", 0.5)
        p_rev = probs.get("revise", 0.5)
        
        margin = abs(p_pass - p_rev)
        if margin < uncertain_margin:
            verdict = "uncertain"
        elif p_pass > p_rev:
            verdict = "pass"
        else:
            verdict = "revise"
            
        return {
            "verdict": verdict,
            "confidence": result.get("confidence", 0.0),
            "probabilities": probs,
            "latency_ms": result.get("latency_ms", 0.0),
            "backend": self.backend
        }

    def _heuristic_review(self, text: str) -> Dict[str, Any]:
        """Lightweight heuristic fallback for environments without local GPU."""
        indicators = [
            "排查过程", "弯路", "尝试过", "未采用", "排除了",
            "本菜品不含", "没有使用", "之前报错", "修复前",
            "we tried", "discarded", "did not include", "not using"
        ]
        text_lower = text.lower()
        has_issue = any(ind in text_lower for ind in indicators)
        return {
            "verdict": "revise" if has_issue else "pass",
            "confidence": 0.85 if has_issue else 0.90,
            "probabilities": {"pass": 0.1 if has_issue else 0.9, "revise": 0.9 if has_issue else 0.1},
            "latency_ms": 1.2,
            "backend": "heuristic_fallback"
        }
