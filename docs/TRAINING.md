# Training Pipeline

This document records the reproducible training path for the project-owned chess move-policy model used by `minimax_policy_ordered`.

The model is a move-policy network for alpha-beta move ordering. It is not a neural board evaluator, value network, win-probability model, or Elo estimate.

## Objective

Train a small PyTorch policy network from scratch to rank legal moves before classical alpha-beta minimax searches them. Minimax leaf nodes still use the handcrafted material plus piece-square-table evaluator in `ai.py`.

## Source Data

- Source: official Lichess Stockfish evaluation database
- URL: `https://database.lichess.org/lichess_db_eval.jsonl.zst`
- Source license recorded in manifests: `CC0`
- Retrieval date: `2026-09-10`
- Streaming path: HTTP response -> incremental Zstandard decompression -> JSONL parsing -> filtering -> deterministic bounded sampling

The old `models/TORCH_100EPOCHS.pth` checkpoint was not used for initialization, labels, distillation, or evaluation.

## Dataset Preparation

Script: `training/prepare_data.py`

Each candidate record is parsed as a FEN position with Stockfish evaluations. The target label is the first UCI move from the highest-depth usable principal variation. The target must be legal in the parsed position and representable by this repository's move vocabulary.

Filters include:

- valid standard-chess FEN
- non-terminal position
- at least two legal moves
- minimum Stockfish depth
- nonempty principal variation
- legal target move
- target move present in the move vocabulary

Duplicate normalized positions are removed. Position identity uses SHA-256 over piece placement, side to move, castling rights, and en-passant square. Splits are assigned deterministically from that hash: approximately 80% train, 10% validation, and 10% test.

## Smoke Run

Command:

```bash
python -m training.prepare_data --data https://database.lichess.org/lichess_db_eval.jsonl.zst --output data/policy_dataset_smoke.jsonl --max-positions 5000 --max-records 100000 --min-depth 18 --seed 42
python -m training.train_policy --data data/policy_dataset_smoke.jsonl --epochs 2 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --device cpu --seed 42 --num-workers 0 --output models/lichess_policy_v1 --patience 3
python -m training.evaluate_policy --data data/policy_dataset_smoke.jsonl --checkpoint models/lichess_policy_v1_best.pt --split test --batch-size 256 --device cpu --num-workers 0
```

Measured smoke throughput:

| Metric | Value |
|---|---:|
| Raw records scanned | 100,000 |
| Final positions | 5,000 |
| Preprocessing records/sec | 2,676.9 |
| Valid positions/sec | 133.8 |
| Training examples/sec | 1,626.9 |
| Seconds/epoch | 2.45 |

Smoke test metrics for the best checkpoint from epoch 1:

| Metric | Value |
|---|---:|
| Test examples | 505 |
| Loss | 3.0389 |
| Top-1 legal accuracy | 0.1287 |
| Top-3 legal accuracy | 0.2851 |
| Top-5 legal accuracy | 0.3881 |
| MRR | 0.2617 |
| Random legal top-1 baseline | 0.0641 |
| Random legal MRR baseline | 0.1836 |

The smoke run verified streaming, parsing, legal labels, finite loss, validation, checkpoint saving/loading, legal masking, and engine invocation.

## Real Training Run

Smoke throughput indicated that a 250k-position run was practical on this CPU-only machine. A 500k-position run was not required for this first public model.

Command:

```bash
python -m training.prepare_data --data https://database.lichess.org/lichess_db_eval.jsonl.zst --output data/policy_dataset_250k.jsonl --max-positions 250000 --max-records 500000 --min-depth 18 --seed 42
python -m training.train_policy --data data/policy_dataset_250k.jsonl --epochs 10 --batch-size 256 --learning-rate 3e-4 --weight-decay 1e-4 --device cpu --seed 42 --num-workers 0 --output models/lichess_policy_v1 --patience 3
python -m training.evaluate_policy --data data/policy_dataset_250k.jsonl --checkpoint models/lichess_policy_v1_best.pt --split test --batch-size 256 --device cpu --num-workers 0
```

Dataset manifest:

| Metric | Value |
|---|---:|
| Raw records scanned | 500,000 |
| Valid records | 467,046 |
| Low-depth rejects | 22,733 |
| Forced-move rejects | 10,099 |
| Illegal targets | 16 |
| Vocabulary misses | 0 |
| Duplicates | 106 |
| Malformed records | 0 |
| Train examples | 200,327 |
| Validation examples | 24,862 |
| Test examples | 24,811 |
| Preprocessing time | 173.87 seconds |

Training configuration:

| Setting | Value |
|---|---|
| Seed | `42` |
| Device | `cpu` |
| Optimizer | `AdamW` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | `256` |
| Loss | cross entropy over legally masked policy logits |
| Early stopping patience | `3` |
| Epochs run | `5` |
| Best checkpoint epoch | `2` |
| Training examples/sec | `1,754.2` |
| Seconds/epoch | `114.20` |

Validation metrics for the selected checkpoint:

| Metric | Value |
|---|---:|
| Validation loss | 2.6300 |
| Top-1 legal accuracy | 0.2374 |
| Top-3 legal accuracy | 0.4735 |
| Top-5 legal accuracy | 0.5935 |
| MRR | 0.4035 |

Held-out test metrics:

| Metric | Value |
|---|---:|
| Test loss | 2.6394 |
| Top-1 legal accuracy | 0.2357 |
| Top-3 legal accuracy | 0.4666 |
| Top-5 legal accuracy | 0.5902 |
| MRR | 0.3999 |
| Random legal top-1 baseline | 0.0633 |
| Random legal MRR baseline | 0.1813 |

## Board Encoding

File: `policy/encoding.py`

`encode_board()` returns an `(18, 8, 8)` float32 tensor from White's perspective:

- planes 0-5: white pawn, knight, bishop, rook, queen, king
- planes 6-11: black pawn, knight, bishop, rook, queen, king
- plane 12: side to move, filled with 1.0 for White and 0.0 for Black
- planes 13-16: white kingside, white queenside, black kingside, black queenside castling rights
- plane 17: en-passant target square

Rows use rank 8 at row 0 and rank 1 at row 7. Legal moves are not encoded as input features; they are handled with an output mask.

## Move Vocabulary

File: `policy/vocab.py`

The vocabulary is generated deterministically from chess move geometry and promotion suffixes.

| Property | Value |
|---|---|
| Vocabulary size | 1,968 |
| Checksum | `e25555119a2ace278793fbdc69d716181ae382c0f4885f072028699abf63180e` |

It supports ordinary moves, captures, castling, en passant, and queen/rook/bishop/knight promotions. Checkpoint loading validates the vocabulary size and checksum before inference.

## Model Architecture

File: `policy/model.py`

`ChessPolicyNet`:

- input: `18 x 8 x 8`
- stem: `Conv2d(18, 64, kernel_size=3, padding=1)`, GroupNorm, ReLU
- 3 residual blocks at 64 channels
- policy head: `Conv2d(64, 16, kernel_size=1)`, ReLU, flatten
- output layer: `Linear(8 * 8 * 16, 1968)`

Model parameters: `2,250,688`

Checkpoint: `models/lichess_policy_v1_best.pt`

Checkpoint size: `9,014,078` bytes, about `8.6 MiB`

## Engine Integration

Files: `neural_eval.py`, `policy/inference.py`, `ai.py`

`minimax_policy_ordered` loads `models/lichess_policy_v1_best.pt`, encodes each unique searched position once, masks logits to legal moves, and sorts legal moves by policy logit before alpha-beta search. Inference uses `model.eval()` and `torch.inference_mode()`. Cache keys include piece placement, side to move, castling rights, and en-passant state.

Leaf evaluation remains `evaluate_classic_board()` from `ai.py`.

## Search Benchmark

Environment:

- CPU: `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`
- Python: `3.12.14`
- PyTorch: `2.6.0+cpu`
- Inference device: CPU
- CUDA available: no

Fixed-depth benchmark command:

```bash
python benchmark.py --depth 3 --runs 5 --mode minimax --mode minimax_policy_ordered
python benchmark.py --depth 3 --runs 5 --mode minimax --mode minimax_policy_ordered --positions-file data/policy_dataset_250k.jsonl --max-positions 25
```

Fixed five-position benchmark at depth 3:

| Result | Value |
|---|---:|
| Positions | 5 |
| Median node reduction | 54.7% |
| Mean node reduction | 53.5% |
| Min node reduction | 6.8% |
| Max node reduction | 88.2% |
| Median runtime change | +22.0% |
| Mean runtime change | +94.9% |
| Fixed-depth score differences | 0 |

Held-out 25-position benchmark at depth 3:

| Result | Value |
|---|---:|
| Positions | 25 |
| Median node reduction | 35.4% |
| Mean node reduction | 35.0% |
| Min node reduction | -89.0% |
| Max node reduction | 84.3% |
| Median runtime change | +76.5% |
| Mean runtime change | +139.1% |
| Fixed-depth score differences | 0 |

Some held-out positions returned equal scores with different move tie-breaks. That is expected when multiple root moves have the same minimax value.

Time-budget smoke benchmark on the first 5 held-out test positions:

| Time budget | Median node reduction | Mean node reduction | Median runtime change | Median depth result |
|---|---:|---:|---:|---|
| 250 ms | 78.9% | 77.9% | +0.6% | neural-ordered tied or lagged classical depth |
| 500 ms | 80.0% | 79.1% | +0.3% | neural-ordered tied or lagged classical depth |
| 1000 ms | 69.0% | 65.8% | +0.2% | neural-ordered tied or lagged classical depth |

## Limitations

- The model is a policy model, not a scalar value evaluator.
- The node-reduction result is position-dependent; one held-out position expanded more nodes with neural ordering.
- CPU inference overhead made runtime worse in the fixed-depth held-out benchmark despite fewer searched nodes.
- Time-budgeted searches did not consistently complete deeper search with neural ordering.
- These results do not show stronger chess play or justify an Elo claim.
- Raw Lichess data and processed JSONL datasets are intentionally excluded from version control.

