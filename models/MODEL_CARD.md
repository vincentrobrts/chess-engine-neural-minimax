# Model Card: `lichess_policy_v1_best.pt`

## Purpose

`lichess_policy_v1_best.pt` is a self-trained PyTorch move-policy checkpoint for this chess engine. It is used to rank legal moves before alpha-beta minimax search in `minimax_policy_ordered`.

It is not a neural board evaluator, value network, win-probability model, mate-distance model, or Elo-rated engine.

## Model Details

- Architecture: `ChessPolicyNet` in `policy/model.py`
- Input: `18 x 8 x 8` board-state tensor from `policy/encoding.py`
- Output: raw logits over a deterministic UCI move vocabulary
- Vocabulary size: `1,968`
- Vocabulary checksum: `e25555119a2ace278793fbdc69d716181ae382c0f4885f072028699abf63180e`
- Parameters: `2,250,688`
- Checkpoint size: `9,014,078` bytes, about `8.6 MiB`
- Best checkpoint epoch: `2`

Architecture:

- `Conv2d(18, 64, kernel_size=3, padding=1)`
- GroupNorm + ReLU
- 3 residual blocks at 64 channels
- `Conv2d(64, 16, kernel_size=1)` policy head
- ReLU + flatten
- `Linear(8 * 8 * 16, 1968)`

## Training Data

- Source: official Lichess Stockfish evaluation database
- URL: `https://database.lichess.org/lichess_db_eval.jsonl.zst`
- License recorded in manifest: `CC0`
- Retrieval date: `2026-09-10`
- Target label: first move of Stockfish's principal variation from the highest-depth usable evaluation
- Minimum Stockfish depth: `18`
- Raw records scanned: `500,000`
- Valid records: `467,046`
- Final sampled positions: `250,000`
- Train/validation/test split: `200,327 / 24,862 / 24,811`
- Split method: SHA-256 over placement, turn, castling rights, and en-passant square

The old `TORCH_100EPOCHS.pth` checkpoint was not used for initialization, labels, or distillation.

## Training Configuration

- Framework: PyTorch `2.6.0+cpu`
- Device: CPU
- Seed: `42`
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Batch size: `256`
- Loss: cross entropy over legally masked move logits
- Requested epochs: `10`
- Early stopping patience: `3`
- Epochs run: `5`

## Metrics

Validation metrics for the selected checkpoint:

| Metric | Value |
|---|---:|
| Loss | 2.6300 |
| Top-1 legal accuracy | 0.2374 |
| Top-3 legal accuracy | 0.4735 |
| Top-5 legal accuracy | 0.5935 |
| MRR | 0.4035 |

Held-out test metrics:

| Metric | Value |
|---|---:|
| Loss | 2.6394 |
| Top-1 legal accuracy | 0.2357 |
| Top-3 legal accuracy | 0.4666 |
| Top-5 legal accuracy | 0.5902 |
| MRR | 0.3999 |
| Random legal top-1 baseline | 0.0633 |
| Random legal MRR baseline | 0.1813 |

## Search Benchmark Summary

Compared with classical minimax at fixed depth 3, `minimax_policy_ordered` preserved identical minimax scores in the tested positions but produced mixed performance:

- five built-in positions: median node reduction `54.7%`, mean node reduction `53.5%`
- 25 held-out test positions: median node reduction `35.4%`, mean node reduction `35.0%`
- held-out min/max node reduction: `-89.0%` to `84.3%`
- held-out median runtime change: `+76.5%` on CPU

This supports a narrow claim that neural ordering can reduce alpha-beta node expansion in this benchmark set. It does not support a claim of faster runtime, stronger play, or higher Elo.

## Intended Use

- Rank legal moves for alpha-beta move ordering.
- Run as an optional policy prior; if the checkpoint cannot load, the engine falls back to normal legal move ordering.

## Limitations

- Not a value/evaluation model.
- Not trained to predict win probability.
- Does not replace the handcrafted minimax leaf evaluator.
- Runtime is CPU-bound and can be slower than classical minimax despite fewer searched nodes.
- Search-efficiency gains are position-dependent.

