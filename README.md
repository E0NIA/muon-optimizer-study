# Muon vs AdamW — an optimizer comparison series

A multi-part, receipts-first study comparing the [**Muon**](https://github.com/KellerJordan/Muon)
optimizer against the de-facto standard **AdamW**, across the regimes practitioners actually train
in. Every part ships the notebook, the raw logs, and the plots.

## The parts

### [Part 1 — Pretraining from scratch](part1-pretraining/)
GPT-2-medium, Gemma-3-270M and LFM2.5-350M trained **from scratch** on WikiText-2, AdamW vs Muon.
**Muon won 2 of 3** (−13% / −17% perplexity) and reached AdamW's best ~1.5–1.7× sooner — but
**tied on the big-vocab Gemma**, which turns out to be the most interesting result: Muon's edge
scales with the fraction of parameters it actually governs (embeddings stay on AdamW).

### [Part 2 — QLoRA fine-tuning](part2-qlora/)
Gemma-4-12B and LFM2.5-8B-A1B (MoE) fine-tuned with **4-bit QLoRA** on OpenR1-Math, AdamW vs Muon
on the adapters. **Muon won both** (−5% / −4% best-val perplexity) with **negligible wall-clock
overhead** — its edge transfers to LoRA even though the update is rank-capped.

## How it's run

- **Optimizers:** reference single-device **Muon** (Keller Jordan's quintic Newton-Schulz) +
  PyTorch's `torch.optim.AdamW`. Muon drives the 2D weight/adapter matrices; AdamW handles
  embeddings, LM head and norms (standard Muon usage).
- Notebooks are **self-contained** and run on a free Colab/Kaggle GPU (Part 1) or a single
  24–48GB GPU (Part 2 QLoRA).

```bash
pip install -r requirements.txt
```

## Honesty

These are exploratory, **single-seed** runs with learning rates at sensible defaults and **best
validation** reported (small datasets overfit late). Each part's README lists its own caveats. The
goal is a fair, reproducible comparison — issues and adversarial review welcome.

## Credits

Muon — [Keller Jordan](https://github.com/KellerJordan/Muon). Models — GPT-2 (OpenAI), Gemma
(Google), LFM2.5 (Liquid AI), Qwen (Alibaba), via 🤗 Transformers. Data — WikiText-2, OpenR1-Math-220k.
