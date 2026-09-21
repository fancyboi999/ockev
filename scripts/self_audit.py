"""
Local Final State Review runner for Ockev repository.
Dogfoods OckevGate locally to audit the repository before pushing to GitHub.
"""

import os
from pathlib import Path
from ockev.gate import OckevGate

REPO_DIR = Path("/Users/nowcoder/Desktop/auto-code-work/ockev")

FILES_TO_AUDIT = [
    "README.md",
    "pyproject.toml",
    "ockev/__init__.py",
    "ockev/model.py",
    "ockev/mlx_engine.py",
    "ockev/gate.py",
    "ockev/cli.py",
    "benchmarks/eval_benchmark.py",
    "skills/final-state-review/SKILL.md",
]

def main():
    gate = OckevGate()
    print("=======================================================")
    print(" Ockev Local Dogfooding: Final State Review on Repo    ")
    print("=======================================================\n")
    
    all_passed = True
    
    for rel_path in FILES_TO_AUDIT:
        fpath = REPO_DIR / rel_path
        if not fpath.exists():
            print(f"[-] MISSING: {rel_path}")
            all_passed = False
            continue
            
        content = fpath.read_text(encoding="utf-8")
        res = gate.review_deliverable(content, task_context=f"Reviewing {rel_path} for public open source release.")
        
        # Check forbidden private words
        private_terms = ["companydev", "fancy-dev", "/data1", "nowcoder", "192.168.", "gho_", "TYPESAFE_API_KEY=ts_"]
        leaks = [t for t in private_terms if t in content]
        
        status = res["verdict"].upper()
        if leaks:
            status = "LEAK_FOUND"
            all_passed = False
            
        print(f"[{status:^10}] {rel_path:<35} | Conf: {res['confidence']:.2f} | Latency: {res['latency_ms']:.1f}ms")
        if leaks:
            print(f"             CRITICAL: Found private terms: {leaks}")
        if res["verdict"] == "revise":
            all_passed = False
            print(f"             Warning: Model flagged for conversational residue: {res['probabilities']}")
            
    print("\n=======================================================")
    if all_passed:
        print(" [RESULT] ALL CHECKS PASSED. Ready for GitHub push!    ")
    else:
        print(" [RESULT] ISSUES DETECTED. Please revise before push.  ")
    print("=======================================================")

if __name__ == "__main__":
    main()
