"""AI move selection logic (random and minimax variants)."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

import chess

from constants import AI_MODE_RANDOM, AI_MODES, HYBRID_NEURAL_WEIGHT, TIME_BUFFER_SECONDS
from neural_eval import get_neural_scorer

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_MIDDLE_GAME_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

KING_END_GAME_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}


class SearchTimeout(Exception):
    """Raised when iterative deepening runs out of time."""


@dataclass
class SearchStats:
    """Measurement data collected for one AI search."""

    nodes: int = 0
    leaf_nodes: int = 0
    cutoffs: int = 0
    depth_completed: int = 0
    max_depth_reached: int = 0
    elapsed_ms: float = 0.0
    nodes_per_second: float = 0.0
    neural_forward_passes: int = 0
    timed_out: bool = False
    mode: str = ""
    requested_depth: int = 0
    selected_move: str | None = None
    score: float | None = None


@dataclass
class SearchResult:
    """Move, score, and diagnostics returned by a measured search."""

    move: chess.Move | None
    score: float | None
    stats: SearchStats


def choose_move(
    board: chess.Board,
    mode: str,
    depth: int = 2,
    time_limit_seconds: float | None = None,
) -> chess.Move | None:
    """Return one legal move for the active side based on the selected AI mode."""
    return choose_move_with_stats(board, mode, depth, time_limit_seconds).move


def choose_move_with_stats(
    board: chess.Board,
    mode: str,
    depth: int = 2,
    time_limit_seconds: float | None = None,
) -> SearchResult:
    """Return an AI move plus search diagnostics for benchmarking."""
    start_time = time.perf_counter()
    stats = SearchStats(mode=mode, requested_depth=depth)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return finalize_search_result(None, None, stats, start_time)

    if mode == AI_MODE_RANDOM:
        return finalize_search_result(random.choice(legal_moves), None, stats, start_time)

    mode_config = AI_MODES.get(mode)
    neural_scorer = prepare_neural_scorer(mode)
    evaluation_mode = get_evaluation_mode(mode)
    deadline = None
    if time_limit_seconds is not None:
        deadline = time.perf_counter() + max(0.0, time_limit_seconds - TIME_BUFFER_SECONDS)

    score, move = iterative_deepening_search_with_stats(
        board=board,
        max_depth=depth,
        evaluation_mode=evaluation_mode,
        deadline=deadline,
        stats=stats,
        use_neural_ordering=bool(mode_config and mode_config.order_with_neural),
    )
    return finalize_search_result(move, score, stats, start_time, neural_scorer)


def prepare_neural_scorer(mode: str):
    config = AI_MODES.get(mode)
    if not config or not config.uses_neural:
        return None

    scorer = get_neural_scorer()
    scorer.reset_forward_pass_count()
    return scorer


def finalize_search_result(
    move: chess.Move | None,
    score: float | None,
    stats: SearchStats,
    start_time: float,
    neural_scorer=None,
) -> SearchResult:
    """Fill in derived timing fields and produce a stable result object."""
    elapsed_seconds = time.perf_counter() - start_time
    stats.elapsed_ms = elapsed_seconds * 1000.0
    stats.nodes_per_second = stats.nodes / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if neural_scorer is not None:
        stats.neural_forward_passes = neural_scorer.forward_pass_count
    stats.selected_move = move.uci() if move is not None else None
    stats.score = score
    return SearchResult(move=move, score=score, stats=stats)


def iterative_deepening_search(
    board: chess.Board,
    max_depth: int,
    evaluation_mode: str,
    deadline: float | None,
    use_neural_ordering: bool | None = None,
) -> chess.Move | None:
    """Search depth 1, 2, 3... and keep the deepest fully completed result."""
    stats = SearchStats(mode=evaluation_mode, requested_depth=max_depth)
    _, move = iterative_deepening_search_with_stats(
        board,
        max_depth,
        evaluation_mode,
        deadline,
        stats,
        use_neural_ordering,
    )
    return move


def iterative_deepening_search_with_stats(
    board: chess.Board,
    max_depth: int,
    evaluation_mode: str,
    deadline: float | None,
    stats: SearchStats,
    use_neural_ordering: bool | None = None,
) -> tuple[float | None, chess.Move | None]:
    """Search depth 1, 2, 3... and keep the deepest fully completed result."""
    if use_neural_ordering is None:
        use_neural_ordering = evaluation_mode != "classic"

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, None

    fallback_move = legal_moves[0]
    best_move = fallback_move
    best_score: float | None = None
    maximizing_player = board.turn == chess.WHITE

    for depth in range(1, max_depth + 1):
        try:
            score, move = minimax(
                board=board,
                depth=depth,
                alpha=-float("inf"),
                beta=float("inf"),
                maximizing_player=maximizing_player,
                evaluation_mode=evaluation_mode,
                deadline=deadline,
                stats=stats,
                use_neural_ordering=use_neural_ordering,
            )
        except SearchTimeout:
            stats.timed_out = True
            break

        stats.depth_completed = depth
        if move is not None:
            best_move = move
            best_score = score

        if deadline is not None and time.perf_counter() >= deadline:
            if depth < max_depth:
                stats.timed_out = True
            break

    return best_score, best_move


def get_evaluation_mode(ai_mode: str) -> str:
    config = AI_MODES.get(ai_mode)
    if config and config.evaluation_mode is not None:
        return config.evaluation_mode
    raise ValueError(f"Unsupported AI mode: {ai_mode}")


def evaluate_board(board: chess.Board, mode: str = "classic") -> float:
    """
    Evaluate the position from White's perspective.

    Available modes:
    - classic: handcrafted material + piece-square tables
    - neural: policy-network proxy score
    - hybrid: weighted blend of classic and neural
    """
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    classic_score = evaluate_classic_board(board)
    if mode == "classic":
        return classic_score

    neural_score = evaluate_neural_proxy(board, classic_score)
    if mode == "neural":
        return neural_score
    if mode == "hybrid":
        return (1.0 - HYBRID_NEURAL_WEIGHT) * classic_score + HYBRID_NEURAL_WEIGHT * neural_score

    raise ValueError(f"Unsupported evaluation mode: {mode}")


def evaluate_classic_board(board: chess.Board) -> int:
    """Handcrafted evaluation using material plus piece-square tables."""
    score = 0
    endgame = is_endgame(board)

    for piece_type, piece_value in PIECE_VALUES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            score += piece_value + piece_square_value(piece_type, square, chess.WHITE, endgame)
        for square in board.pieces(piece_type, chess.BLACK):
            score -= piece_value + piece_square_value(piece_type, square, chess.BLACK, endgame)

    return score


def evaluate_neural_proxy(board: chess.Board, fallback_score: int) -> float:
    """
    Derive a usable minimax score from the move-prediction network.

    The PyTorch model predicts moves rather than scalar values, so this is a
    policy-based proxy rather than a true value evaluation.
    """
    scorer = get_neural_scorer()
    if not scorer.available:
        return fallback_score

    best_move = scorer.best_legal_move(board)
    if best_move is None:
        return fallback_score

    board.push(best_move)
    rollout_score = evaluate_classic_board(board)
    board.pop()

    return rollout_score + scorer.confidence_score(board)


def piece_square_value(piece_type: int, square: int, color: bool, endgame: bool) -> int:
    """Return the table bonus/penalty for a piece on a given square."""
    if piece_type == chess.KING:
        table = KING_END_GAME_TABLE if endgame else KING_MIDDLE_GAME_TABLE
    else:
        table = PIECE_SQUARE_TABLES[piece_type]

    rank = chess.square_rank(square)
    file = chess.square_file(square)
    row = 7 - rank if color == chess.WHITE else rank
    return table[row * 8 + file]


def is_endgame(board: chess.Board) -> bool:
    """
    Follow the simple rule:
    endgame starts when there are no queens, or each side with a queen has
    at most one minor piece and no rooks.
    """
    return side_is_in_endgame(board, chess.WHITE) and side_is_in_endgame(board, chess.BLACK)


def side_is_in_endgame(board: chess.Board, color: bool) -> bool:
    if not board.pieces(chess.QUEEN, color):
        return True

    rooks = len(board.pieces(chess.ROOK, color))
    minor_pieces = len(board.pieces(chess.BISHOP, color)) + len(board.pieces(chess.KNIGHT, color))
    return rooks == 0 and minor_pieces <= 1


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    evaluation_mode: str,
    deadline: float | None,
    stats: SearchStats | None = None,
    use_neural_ordering: bool | None = None,
    ply: int = 0,
) -> tuple[float, chess.Move | None]:
    """Depth-limited minimax with alpha-beta pruning and deadline checks."""
    if use_neural_ordering is None:
        use_neural_ordering = evaluation_mode != "classic"

    check_deadline(deadline)
    if stats is not None:
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, ply)

    if depth == 0 or board.is_game_over():
        if stats is not None:
            stats.leaf_nodes += 1
        return evaluate_board(board, evaluation_mode), None

    legal_moves = list(board.legal_moves)
    legal_moves = order_moves(board, legal_moves, use_neural_ordering)
    best_move = None

    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            check_deadline(deadline)
            board.push(move)
            try:
                evaluation, _ = minimax(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    evaluation_mode,
                    deadline,
                    stats=stats,
                    use_neural_ordering=use_neural_ordering,
                    ply=ply + 1,
                )
            finally:
                board.pop()

            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                if stats is not None:
                    stats.cutoffs += 1
                break
        return max_eval, best_move

    min_eval = float("inf")
    for move in legal_moves:
        check_deadline(deadline)
        board.push(move)
        try:
            evaluation, _ = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                True,
                evaluation_mode,
                deadline,
                stats=stats,
                use_neural_ordering=use_neural_ordering,
                ply=ply + 1,
            )
        finally:
            board.pop()

        if evaluation < min_eval:
            min_eval = evaluation
            best_move = move
        beta = min(beta, evaluation)
        if beta <= alpha:
            if stats is not None:
                stats.cutoffs += 1
            break
    return min_eval, best_move


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout


def order_moves(
    board: chess.Board,
    legal_moves: list[chess.Move],
    use_neural_ordering: bool,
) -> list[chess.Move]:
    """Use the neural policy for move ordering when available."""
    if not use_neural_ordering:
        return legal_moves

    scorer = get_neural_scorer()
    if not scorer.available:
        return legal_moves
    return scorer.order_moves(board, legal_moves)
