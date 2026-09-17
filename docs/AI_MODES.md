# AI Modes

This document describes the chess AI modes implemented in the repository. It is based on the executable code, not on marketing claims.

## Project Flow

The Pygame loop in `main.py` collects mouse input, updates clocks, starts AI search on a background thread, and redraws the board. `ChessGame` in `game.py` owns the `chess.Board`, selected square, move history, clocks, opening-book state, and active AI mode. Legal moves and game-over rules come from `python-chess`.

An AI move follows this path:

1. `main.py` starts `start_ai_search()` when it is the AI side's turn.
2. `start_ai_search()` copies the current board and active AI mode into a worker thread.
3. `ChessGame.choose_ai_move()` checks the opening book first.
4. If the book does not return a move, `ai.choose_move()` selects a move using the configured mode.
5. `main.py` applies the returned legal move to the live `ChessGame`.
6. `ui.py` redraws the board, sidebar, move history, clocks, and mode selector.

Mode changes are ignored while an AI search thread is active. The worker thread uses the mode snapshot captured before search starts.

## Opening Book

Opening-book handling is implemented in `opening_book.py` and is called from `ChessGame.choose_ai_move()` before normal AI search. The book loads `openings/*.tsv`, reads each row's ECO code, name, and PGN text, normalizes PGN tokens into SAN moves using `python-chess`, and stores each usable line as an `OpeningLine`.

The book is available to every AI mode because it sits above the mode-specific search layer. If the human deviates from the selected line, the line ends, or a SAN move cannot be parsed, the book is disabled for the rest of that game and the engine falls back to the selected AI mode.

## Mode: Random

Purpose:
Provide a simple baseline opponent and a low-cost way to verify game flow.

Search:
No search is performed. The mode selects one move with `random.choice(list(board.legal_moves))`.

Evaluation:
No board evaluation is used.

Neural network:
No neural network is used.

Relevant files:

- `constants.py`: `AI_MODE_RANDOM`, `AI_MODES`
- `ai.py`: `choose_move_with_stats()`
- `game.py`: `choose_ai_move()`

## Mode: Minimax

Purpose:
Use a classical handcrafted search and evaluation path.

Search:
Depth-limited minimax with alpha-beta pruning. Search is run with iterative deepening from depth 1 through the requested depth. The current code does not implement transposition tables or quiescence search.

Evaluation:
Leaf nodes use `evaluate_classic_board()`, which combines material values with piece-square tables. The king uses a middle-game table or endgame table based on a simple material rule.

Time control:
`ChessGame.get_ai_time_budget()` chooses a per-move budget from the active side's remaining clock. `ai.check_deadline()` raises `SearchTimeout` when the deadline is reached, and iterative deepening returns the best fully completed depth.

Neural network:
No neural network is used.

Execution path:
`legal moves -> iterative deepening -> minimax -> alpha-beta pruning -> classic leaf evaluation -> best move`

Relevant files:

- `constants.py`: `AI_MODE_MINIMAX`, `MINIMAX_DEPTH`
- `ai.py`: `iterative_deepening_search_with_stats()`, `minimax()`, `evaluate_classic_board()`
- `game.py`: `get_ai_time_budget()`, `choose_ai_move()`

## Mode: Neural-Ordered Minimax

Purpose:
Use the self-trained PyTorch move-policy model as a search-ordering prior while keeping minimax leaf evaluation purely classical.

Search:
The search algorithm is the same iterative-deepening alpha-beta minimax used by classical minimax. Standard minimax uses the legal move order provided by `python-chess`; this mode sorts legal moves by policy logits before search. Legal moves absent from the vocabulary are preserved and sorted after scored moves.

Evaluation:
Leaf nodes use only `evaluate_classic_board()`, the same evaluator used by normal minimax. This mode does not call `evaluate_neural_proxy()`, does not add `confidence_score()`, and does not blend policy logits into the board score.

Neural network:
`NeuralMoveScorer` loads `models/lichess_policy_v1_best.pt`, validates the embedded vocabulary size/checksum, encodes the board once per unique position, masks to legal moves, and returns legal-move logits for ordering. If the checkpoint cannot load, move ordering falls back to normal legal move order and leaf evaluation remains classical.

Execution path:
`legal moves -> policy logits -> legal-move sorting -> iterative deepening -> minimax -> classic leaf evaluation -> alpha-beta propagation -> best move`

Relevant files:

- `constants.py`: `AI_MODE_MINIMAX_POLICY_ORDERED`, `NEURAL_MODEL_PATH`
- `ai.py`: `choose_move_with_stats()`, `order_moves()`, `minimax()`, `evaluate_classic_board()`
- `neural_eval.py`: `NeuralMoveScorer`
- `policy/`: vocabulary, encoding, model, checkpoint, and inference code
- `models/lichess_policy_v1_best.pt`

## Mode: Policy-Proxy Minimax

Purpose:
Keep the legacy experimental idea of using the move-policy model as a policy-derived scoring proxy inside minimax.

Search:
The search algorithm is the same iterative-deepening alpha-beta minimax. Move ordering uses the policy model when available.

Evaluation:
This is not a true scalar value network. `evaluate_neural_proxy()` asks the policy model for the highest-scoring legal move, pushes that move, evaluates the resulting board with the classical evaluator, then adds a small confidence term based on legal-move logits.

Neural network:
The same self-trained move-policy checkpoint is used. If it cannot load, this mode falls back to the classical score internally.

Execution path:
`legal moves -> policy move ordering -> minimax -> policy proxy at leaf nodes -> alpha-beta propagation -> best move`

Relevant files:

- `constants.py`: `AI_MODE_MINIMAX_NN`
- `ai.py`: `evaluate_neural_proxy()`, `order_moves()`
- `neural_eval.py`: `NeuralMoveScorer`
- `policy/`

## Mode: Hybrid Policy-Proxy

Purpose:
Blend the handcrafted evaluator with the policy-derived proxy.

Search:
The search algorithm is the same iterative-deepening alpha-beta minimax. Move ordering uses the policy model when available.

Evaluation:
Leaf evaluation computes the classic score and the policy proxy score, then blends them:

`(1.0 - HYBRID_NEURAL_WEIGHT) * classic_score + HYBRID_NEURAL_WEIGHT * neural_score`

At the time of writing, `HYBRID_NEURAL_WEIGHT` is 0.35.

Neural network:
The same self-trained move-policy checkpoint is used. If it cannot load, the proxy returns the classic score, making this mode effectively classical.

Execution path:
`legal moves -> policy move ordering -> minimax -> classic score + policy proxy blend -> alpha-beta propagation -> best move`

Relevant files:

- `constants.py`: `AI_MODE_MINIMAX_HYBRID`, `HYBRID_NEURAL_WEIGHT`
- `ai.py`: `evaluate_board()`, `evaluate_neural_proxy()`, `order_moves()`
- `neural_eval.py`: `NeuralMoveScorer`
- `policy/`

## Policy Network Implementation

Architecture:
`ChessPolicyNet` in `policy/model.py` is a small residual PyTorch network:

- `Conv2d(18, 64, kernel_size=3, padding=1)`
- GroupNorm + ReLU
- 3 residual blocks at 64 channels
- policy head: `Conv2d(64, 16, kernel_size=1)`, ReLU, flatten
- `Linear(8 * 8 * 16, move_vocabulary_size)`

Input representation:
`encode_board()` in `policy/encoding.py` produces an `(18, 8, 8)` float32 tensor from White's perspective. Planes 0-11 encode white and black pieces. Plane 12 marks side to move. Planes 13-16 encode castling rights. Plane 17 marks the en-passant target square when present. Legal destination squares are not encoded as input features.

Output representation:
The model outputs raw logits over a deterministic UCI move vocabulary generated in `policy/vocab.py`. Illegal moves are masked outside the model during training, evaluation, and inference.

Training:
The current checkpoint was trained from scratch with `training/prepare_data.py` and `training/train_policy.py` using a bounded sample of the official Lichess Stockfish evaluation database. The policy target is the first move of Stockfish's principal variation from the highest-depth usable evaluation.

Checkpoint:
`models/lichess_policy_v1_best.pt` stores the model weights plus architecture config, input plane count, vocabulary size/checksum, epoch, validation metrics, training config, dataset manifest, seed, PyTorch version, data source, and data license.

Important limitation:
This is a move-policy model. It predicts promising moves for ordering and proxy experiments. It is not a neural board evaluator, value network, win-probability model, or mate-distance model.

