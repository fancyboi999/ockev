"""
Evaluation script for TomatoEggBench-120.
Measures Accuracy, Precision, Recall, F1, and Latency against standard benchmark cases.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

BENCH_FILE = Path(__file__).parent / "tomato_egg_bench_120.json"

def run_evaluation(gate_fn):
    with open(BENCH_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    print(f"Loaded {len(cases)} cases from {BENCH_FILE.name}")
    correct = 0
    total = len(cases)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    latencies = []
    
    for case in cases:
        text = case.get("input_text", "")
        expected = case.get("expected_label", "")
        
        t0 = time.time()
        res = gate_fn(text)
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        
        verdict = res.get("verdict")
        is_correct = (verdict == expected)
        if is_correct:
            correct += 1
            
        if expected == "revise":
            if verdict == "revise":
                tp += 1
            else:
                fn += 1
        elif expected == "pass":
            if verdict == "pass":
                tn += 1
            else:
                fp += 1
                
    acc = correct / total * 100.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 45)
    print(" TomatoEggBench-120 Official Results ")
    print("=" * 45)
    print(f"Total Cases:       {total}")
    print(f"Accuracy:          {acc:.1f}% ({correct}/{total})")
    print(f"Violation Recall:  {rec:.1f}% ({tp}/{tp + fn})")
    print(f"Precision:         {prec:.1f}% ({tp}/{tp + fp})")
    print(f"F1 Score:          {f1:.1f}")
    print(f"Avg Latency:       {avg_lat:.1f} ms")
    print("=" * 45)

if __name__ == "__main__":
    from ockev.gate import OckevGate
    gate = OckevGate()
    run_evaluation(lambda t: gate.review_deliverable(t))
