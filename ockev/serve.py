"""
Ockev HTTP Inference Server
Implements TypeSafe System One compatible API (/v1/systemone) for seamless drop-in evaluation and JevBench integration.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .gate import OckevGate

app = FastAPI(title="Ockev System One Inference Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global gate instance
GATE: Optional[OckevGate] = None

def get_gate() -> OckevGate:
    global GATE
    if GATE is None:
        GATE = OckevGate()
    return GATE

class QuestionCriteria(BaseModel):
    instructions: Optional[str] = None
    criteria: Optional[Dict[str, str]] = None
    type: Optional[str] = "choice"

class SystemOneRequest(BaseModel):
    state: Any = Field(..., description="Unstructured deliverable text or context object")
    questions: Dict[str, QuestionCriteria] = Field(..., description="Map of question IDs to criteria")
    model: Optional[str] = "ockev-latest"

class ChoiceAnswer(BaseModel):
    type: str = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]

class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, ChoiceAnswer]
    latency_ms: float
    generated_tokens: int = 0

@app.get("/healthz")
def healthz():
    gate = get_gate()
    return {"status": "ok", "backend": gate.backend, "model": gate.model_name}

@app.get("/v1/models")
def list_models():
    return {
        "models": [
            {"id": "ockev-1.5b", "name": "Ockev 1.5B (Fast Gatekeeper)"},
            {"id": "ockev-3b", "name": "Ockev 3B (Zero False Positive Flagship)"}
        ]
    }

@app.post("/v1/systemone", response_model=SystemOneResponse)
def systemone(req: SystemOneRequest):
    t0 = time.time()
    gate = get_gate()
    
    # Textualize state
    if isinstance(req.state, dict):
        state_text = req.state.get("candidate_text") or req.state.get("text") or str(req.state)
        task_ctx = req.state.get("task_context") or ""
    else:
        state_text = str(req.state)
        task_ctx = ""

    answers = {}
    for qid, q in req.questions.items():
        res = gate.review_deliverable(
            candidate_text=state_text,
            task_context=task_ctx
        )
        
        choice = res.get("verdict", "pass")
        conf = float(res.get("confidence", 0.95))
        probs = res.get("probabilities", {"pass": conf if choice == "pass" else 1.0 - conf, "revise": conf if choice == "revise" else 1.0 - conf})
        
        answers[qid] = ChoiceAnswer(
            choice=choice,
            confidence=conf,
            probabilities=probs
        )

    latency_ms = round((time.time() - t0) * 1000, 2)
    return SystemOneResponse(
        model=req.model or "ockev-1.5b",
        answers=answers,
        latency_ms=latency_ms,
        generated_tokens=0
    )

def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Ockev Inference Server")
    parser.add_argument("--port", type=int, default=8008, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()
    
    print(f"Starting Ockev System One server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
