---
name: final-state-review
description: Review content for its intended audience and current verified state using Ockev or Jev. Use before handing off or publishing an artifact, plan, or task instructions.
---

# Final State Review (Ockev Edition)

Produce content that stands on its own for the next reader or agent. Ship the result, not the conversation.

## Core Rule: Ending the Tomato-Egg Problem

When an agent is asked for "Tomato Scrambled Eggs", it must not deliver:
> *"Here is your tomato scrambled eggs (note: this recipe contains no pork, no beef, and no fish, after we tried five alternative seasonings)."*

Deliverables must describe adopted behavior and verified facts directly:
- **No conversational residue**: Do not include apologies, explanations of discarded options, or narrative accounts of debugging trials.
- **No private scaffolding**: Do not expose internal machine hostnames, private local file paths, or intermediate scripts.
- **Preserve valid invariants**: Legitimate requirements, safety constraints, and migration histories must be retained when required by specifications.

## Review Architecture

Use `ockev` as the local gatekeeper:

```bash
# Verify a PR or document before publishing
ockev check path/to/deliverable.md
```

1. **Local First**: Evaluates with Apple Silicon MLX or PyTorch/CUDA in ~35ms with zero API cost and zero data leakage.
2. **Cloud Fallback**: When local GPU acceleration is unavailable and `TYPESAFE_API_KEY` is configured, automatically routes to TypeSafe Jev API.
