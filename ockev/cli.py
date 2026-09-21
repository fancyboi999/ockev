"""
Command Line Interface for Ockev.
"""

import argparse
import sys
from pathlib import Path
from .gate import OckevGate

def main():
    parser = argparse.ArgumentParser(
        prog="ockev",
        description="Ockev: Ending the Tomato-Egg Problem in AI Agent Deliverables."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # check command
    check_parser = subparsers.add_parser("check", help="Review deliverable text or file.")
    check_parser.add_argument("path", nargs="?", help="Path to text or markdown file.")
    check_parser.add_argument("--text", help="Direct text string to inspect.")
    check_parser.add_argument("--backend", choices=["auto", "mlx", "pytorch"], default="auto")
    
    args = parser.parse_args()
    
    if args.command == "check":
        content = ""
        if args.path:
            p = Path(args.path)
            if not p.exists():
                print(f"Error: File '{args.path}' not found.", file=sys.stderr)
                sys.exit(1)
            content = p.read_text(encoding="utf-8")
        elif args.text:
            content = args.text
        else:
            print("Error: Must provide file path or --text.", file=sys.stderr)
            sys.exit(1)
            
        gate = OckevGate(backend=args.backend)
        res = gate.review_deliverable(content)
        
        print(f"=== Ockev Deliverable Gate Review ===")
        print(f"Verdict:     {res['verdict'].upper()}")
        print(f"Confidence:  {res['confidence']:.2f}")
        print(f"Latency:     {res['latency_ms']} ms")
        print(f"Probabilities: {res['probabilities']}")
        
        if res["verdict"] == "revise":
            print("\n[FAIL] Deliverable contains conversational clutter, negative echoes, or discarded options.")
            sys.exit(1)
        elif res["verdict"] == "uncertain":
            print("\n[WARN] Margin below confidence threshold. Requires manual inspection.")
            sys.exit(2)
        else:
            print("\n[PASS] Clean deliverable. Ready to ship.")
            sys.exit(0)

if __name__ == "__main__":
    main()
