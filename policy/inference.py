"""Runtime move-policy inference with legal-move masking."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from policy.encoding import encode_board, position_key
from policy.vocab import MoveVocabulary, get_move_vocabulary

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None

if torch is not None:
    from policy.checkpoint import load_policy_checkpoint


class PolicyMoveScorer:
    """Load a project-owned move-policy checkpoint and score legal moves."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.available = False
        self.forward_pass_count = 0
        self._logit_cache: dict[str, np.ndarray] = {}
        self._load_error: str | None = None
        self.vocabulary: MoveVocabulary = get_move_vocabulary()

        if torch is None:
            self._load_error = "PyTorch is not installed."
            return

        self.device = torch.device(device)
        base_dir = Path(__file__).resolve().parent.parent
        self.checkpoint_path = base_dir / checkpoint_path

        try:
            self.model, self.checkpoint = load_policy_checkpoint(
                self.checkpoint_path,
                device=self.device,
                vocabulary=self.vocabulary,
            )
            self.available = True
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            self._load_error = f"Could not load policy checkpoint: {exc}"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def clear_cache(self) -> None:
        self._logit_cache.clear()

    def reset_forward_pass_count(self) -> None:
        self.forward_pass_count = 0

    def best_legal_move(self, board: chess.Board) -> chess.Move | None:
        scored_moves = self.score_legal_moves(board)
        if not scored_moves:
            return None
        return max(scored_moves, key=scored_moves.get)

    def order_moves(self, board: chess.Board, legal_moves: list[chess.Move]) -> list[chess.Move]:
        if not self.available:
            return legal_moves

        scored_moves = self.score_legal_moves(board)
        return sorted(
            legal_moves,
            key=lambda move: scored_moves.get(move, float("-inf")),
            reverse=True,
        )

    def confidence_score(self, board: chess.Board) -> float:
        if not self.available:
            return 0.0

        scored_moves = self.score_legal_moves(board)
        if not scored_moves:
            return 0.0

        logits = np.array(list(scored_moves.values()), dtype=np.float32)
        score = (float(np.max(logits)) - float(np.mean(logits))) * 25.0
        return score if board.turn == chess.WHITE else -score

    def score_legal_moves(self, board: chess.Board) -> dict[chess.Move, float]:
        if not self.available:
            return {}

        logits = self._get_logits(board)
        legal_scores: dict[chess.Move, float] = {}
        for move in board.legal_moves:
            move_id = self.vocabulary.move_to_id.get(move.uci())
            if move_id is not None:
                legal_scores[move] = float(logits[move_id])
        return legal_scores

    def _get_logits(self, board: chess.Board) -> np.ndarray:
        cache_key = position_key(board)
        cached = self._logit_cache.get(cache_key)
        if cached is not None:
            return cached

        matrix = encode_board(board)
        input_tensor = torch.from_numpy(matrix).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(input_tensor).squeeze(0).detach().cpu().numpy()

        self.forward_pass_count += 1
        if len(self._logit_cache) > 20000:
            self._logit_cache.clear()
        self._logit_cache[cache_key] = logits
        return logits

