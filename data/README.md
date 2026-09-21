---
license: apache-2.0
task_categories:
- text-classification
tags:
- agent-evaluation
- decision-model
- system-one
- jev
- tomato-egg-problem
- benchmark
size_categories:
- n<1K
---

<div align="center">

# TomatoEggBench: Evaluating Final-State Discipline in AI Agent Deliverables

### "Ship the result, not the conversation."

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ockev_Models-yellow.svg)](https://huggingface.co/fancyboi999/ockev)
[![GitHub](https://img.shields.io/badge/GitHub-fancyboi999%2Fockev-blue.svg)](https://github.com/fancyboi999/ockev)

</div>

---

## Dataset Summary

**TomatoEggBench** is a 120-item public benchmark designed to test whether AI agents deliver clean, invariant final states or leak conversational scaffolding, discarded alternatives, and negative echoes into user deliverables.

The dataset is named after the "Tomato-Egg Problem" highlighted by UC Berkeley researcher Shangyin Tan:
> *"I asked an AI agent to write a recipe for tomato scrambled eggs. It proudly delivered: 'Here is your tomato scrambled eggs (note: this dish contains no pork, no beef, no chicken, and no fish).' Why are AI agents so obsessed with telling us what they didn't do? Ship the result, not the conversation."*

## Data Composition

The benchmark covers 5 major engineering and enterprise deliverable domains:
1. **Code Repositories**: Commit messages, PR descriptions, code comments, and docstrings.
2. **Spreadsheets & Data**: Financial models, CSV/Excel summaries, accounting exports.
3. **Formal Documents & PDFs**: Technical specifications, executive summaries, compliance filings.
4. **Form Automation & RPA**: Robotic process automation input data, submission manifests.
5. **Executive Slide Decks**: Client pitches, strategy decks, board presentations.

| Split | Description | Total Items | Pass Rate | Revise Rate |
| :--- | :--- | :---: | :---: | :---: |
| `test.jsonl` | Standard Held-Out Benchmark | 120 | 50.0% (60) | 50.0% (60) |
| `community_40.jsonl` | Real Community Bug Reports | 40 | 37.5% (15) | 62.5% (25) |

## Key Findings on TomatoEggBench-120

| Model | Architecture | Accuracy | Violations Caught (Recall) | Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Ockev-3B** | Qwen2.5-3B + Pointer Head | **95.8%** | **98.3%** | 86 ms |
| **TypeSafe Jev 1.13.0** | Causal MoE + RLCD | 93.3% | 96.7% | 250 ms |
| **Ockev-1.5B** | Qwen2.5-1.5B + Pointer Head | 92.5% | 95.0% | 35 ms |
| **SemIf Qwen3.5-4B** | Qwen3.5-4B (Logits Readout) | 74.7% | 68.3% | 850 ms |
| **ModernBERT-151M** | Sequence Classification Head | 25.0% | 35.0% | 24 ms |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("fancyboi999/tomato-egg-bench")
print(dataset["test"][0])
```

## Citation & Code

Official repo: [https://github.com/fancyboi999/ockev](https://github.com/fancyboi999/ockev)
