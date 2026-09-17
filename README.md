# Chess Engine with Neural-Assisted Minimax

A Python chess engine that combines alpha-beta minimax search with a
self-trained PyTorch move-policy model for neural-assisted move ordering. The
project includes interactive Human-vs-AI gameplay, TSV opening-book support,
time-bounded iterative deepening, and a reproducible training and benchmark
pipeline.

Supported AI modes:
- `random`: picks a random legal move
- `minimax`: uses depth-limited minimax with alpha-beta pruning, material values, and piece-square tables
- `minimax_policy_ordered`: uses the PyTorch move-policy model only to order legal moves before classical minimax search
- `minimax_nn`: uses an experimental PyTorch move-policy model as a neural proxy inside minimax when the model is available
- `minimax_hybrid`: blends handcrafted evaluation with the neural policy proxy
- `opening book`: the AI will follow one randomly chosen TSV opening line when the played moves match it, then fall back to normal AI when the line ends or is broken

## Technologies

- Python 3.10+
- [python-chess](https://python-chess.readthedocs.io/)
- [pygame](https://www.pygame.org/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)

## File Structure

- `main.py` - game loop and app entry point
- `game.py` - game state coordination and move handling
- `ai.py` - random AI and minimax AI logic
- `benchmark.py` - fixed-position search diagnostics
- `neural_eval.py` - PyTorch model loading, board encoding reuse, and neural move scoring
- `policy/` - move vocabulary, board encoder, policy model, checkpoint loading, and inference
- `training/` - Lichess eval data preparation, policy training, and model evaluation scripts
- `opening_book.py` - TSV opening-book loading and opening selection
- `ui.py` - board drawing, piece rendering, and click mapping
- `constants.py` - shared settings (colors, dimensions, defaults)
- `docs/AI_MODES.md` - technical explanation of each AI mode
- `docs/TRAINING.md` - reproducible training pipeline and measured smoke/full-run results
- `models/` - self-trained policy checkpoint and model card
- `openings/` - local TSV opening database files
- `requirements.txt` - project dependencies

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run the Game

```bash
python main.py
```

At startup, the game lets you choose whether to play White or Black.
The board is automatically oriented from the human player's perspective.
Each side also has a 10-minute chess clock that runs only on its own turn.
Use the AI mode selector in the sidebar to switch modes during a game. Mode changes are disabled while the AI is thinking.

## Benchmarking

Run fixed-position search benchmarks without opening the Pygame GUI:

```bash
python benchmark.py
python benchmark.py --depth 3 --runs 5
python benchmark.py --mode minimax --mode minimax_policy_ordered
python benchmark.py --mode minimax --mode minimax_policy_ordered --time-ms 1000
```

The benchmark performs one warm-up run and then repeated measured runs for each
fixed FEN position. It reports every measured runtime, mean/median runtime,
standard deviation, selected move, score, nodes searched, leaf nodes,
alpha-beta cutoffs, nodes per second, completed depth, timeout status, neural
forward-pass count, and Minimax vs Neural-Ordered Minimax node-reduction
percentages.

On one local CPU-only run at fixed depth 3, Neural-Ordered Minimax reduced
median alpha-beta node expansion by 54.7% across five built-in benchmark
positions and 35.4% across 25 held-out test positions while preserving
identical minimax scores. The held-out result was position-dependent, including
one regression in node count, and CPU runtime was slower overall; this is not
evidence of stronger chess play or a general runtime improvement.

## Tests

```bash
python -m py_compile ai.py benchmark.py constants.py game.py main.py neural_eval.py opening_book.py ui.py tests/test_engine_behavior.py tests/test_policy_components.py
python -m unittest discover -s tests -v
python -c "import main; print('main import ok')"
```

## Training

The current policy checkpoint was trained from scratch from a bounded sample of
the official Lichess Stockfish evaluation database. The pipeline streams the
compressed source, samples positions deterministically, uses the first move of
Stockfish's principal variation as the policy target, and stores dataset
provenance in a manifest.

```bash
pip install -r requirements-train.txt
python -m training.prepare_data --output data/policy_dataset_250k.jsonl --max-positions 250000 --max-records 500000 --min-depth 18 --seed 42
python -m training.train_policy --data data/policy_dataset_250k.jsonl --epochs 10 --batch-size 256 --learning-rate 3e-4 --weight-decay 1e-4 --device auto --seed 42 --num-workers 0 --output models/lichess_policy_v1 --patience 3
python -m training.evaluate_policy --data data/policy_dataset_250k.jsonl --checkpoint models/lichess_policy_v1_best.pt --split test
```

## AI Modes

- **Random AI**: easier opponent, useful for testing and demoing flow.
- **Minimax AI**: classic handcrafted evaluation only.
- **Neural-Ordered Minimax**: uses the PyTorch move-policy network for move ordering only; leaf evaluation remains the same material + piece-square-table evaluator used by Minimax.
- **Policy-Proxy Minimax**: uses the move-prediction network to score legal moves and build a lightweight policy-derived proxy for leaf evaluation when PyTorch and the model files are available.
- **Hybrid Policy-Proxy**: combines handcrafted evaluation with the policy-derived proxy.
- **Opening book behavior**: when the position matches one of the TSV opening lines, the AI chooses one matching line at random and follows it. If the human deviates from that line, or if the line runs out of moves, the AI returns to its normal move selection.
- **Time management**: AI search uses iterative deepening plus a per-move time budget based on remaining clock time, so it always returns the best fully searched move before the deadline.

See [`docs/AI_MODES.md`](docs/AI_MODES.md) for the detailed algorithm and neural-network audit.

To adjust minimax strength/speed tradeoff, edit `MINIMAX_DEPTH` in `constants.py`.
Higher depth increases the search horizon and may improve move quality, but it
costs more computation.

## Design Notes

- `python-chess` handles legal move generation, turn order, and game-over rules.
- The opening book is intentionally simple: it reads SAN moves from the TSV files and uses prefix matching against the moves played in the current game.
- The minimax evaluation stays interview-friendly: it uses classic material values and piece-square tables instead of more advanced engine heuristics.
- The PyTorch model is a move-policy model, not a true value network, so the engine uses it as a policy-based proxy and for move ordering.
- Standard minimax currently uses the legal move order provided by `python-chess`; Neural-Ordered Minimax sorts those legal moves by policy logits before alpha-beta search.
- The clock and search are connected: the UI keeps updating while the AI thinks, and the AI stops search cleanly when its time budget is almost exhausted.
- UI and game logic are separated to keep code easy to explain.
- AI logic is isolated in `ai.py` so strategy changes do not affect UI code.
- Scope is intentionally minimal to stay readable and practical.

## Limitations

- The engine is not benchmarked against Stockfish and does not claim an Elo rating.
- Node-reduction benchmarks should not be described as stronger chess play unless separate strength testing supports that claim.
- Legal move generation is handled by `python-chess`; this project focuses on UI, game flow, search, evaluation, and model integration.
- The repository includes a reproducible policy-training pipeline, but raw Lichess data and large processed datasets are intentionally not committed.
- The neural model is a move-policy model, not a scalar value evaluator.

## License and Third-Party Assets

The original source code and documentation in this repository are licensed
under the MIT License. See [LICENSE](LICENSE).

Third-party assets retain their original licenses:

- Chess opening data in `openings/` comes from
  `lichess-org/chess-openings` and is released under CC0.
- The Lichess Stockfish evaluation data used to train the included policy
  checkpoint is released under CC0. Raw training data is not redistributed.
- Chess piece images in `pieces/` are attributed to Colin M.L. Burnett's
  Wikimedia Commons chess-piece set and licensed under CC BY-SA 3.0.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and
license details.
