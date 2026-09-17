"""Deterministic UCI move vocabulary for standard chess."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import chess

PROMOTION_PIECES = ("q", "r", "b", "n")


@dataclass(frozen=True)
class MoveVocabulary:
    moves: tuple[str, ...]
    move_to_id: dict[str, int]
    id_to_move: dict[int, str]
    checksum: str

    def encode(self, move: chess.Move | str) -> int:
        uci = move if isinstance(move, str) else move.uci()
        return self.move_to_id[uci]

    def decode(self, move_id: int) -> chess.Move:
        return chess.Move.from_uci(self.id_to_move[move_id])

    def contains(self, move: chess.Move | str) -> bool:
        uci = move if isinstance(move, str) else move.uci()
        return uci in self.move_to_id


def build_move_vocabulary() -> MoveVocabulary:
    moves = tuple(sorted(_generate_standard_move_uci()))
    move_to_id = {move: index for index, move in enumerate(moves)}
    id_to_move = {index: move for move, index in move_to_id.items()}
    checksum = hashlib.sha256("\n".join(moves).encode("utf-8")).hexdigest()
    return MoveVocabulary(
        moves=moves,
        move_to_id=move_to_id,
        id_to_move=id_to_move,
        checksum=checksum,
    )


def legal_move_ids(board: chess.Board, vocabulary: MoveVocabulary | None = None) -> list[int]:
    vocab = vocabulary or get_move_vocabulary()
    return [vocab.encode(move) for move in board.legal_moves if vocab.contains(move)]


def _generate_standard_move_uci() -> set[str]:
    moves: set[str] = set()
    for from_square in chess.SQUARES:
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue
            file_delta = chess.square_file(to_square) - from_file
            rank_delta = chess.square_rank(to_square) - from_rank
            if _is_standard_piece_vector(file_delta, rank_delta):
                moves.add(chess.square_name(from_square) + chess.square_name(to_square))

    for from_square in chess.SQUARES:
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        if from_rank not in (1, 6):
            continue
        promotion_rank = 7 if from_rank == 6 else 0
        for file_delta in (-1, 0, 1):
            to_file = from_file + file_delta
            if 0 <= to_file < 8:
                base = chess.square_name(from_square) + chess.square_name(
                    chess.square(to_file, promotion_rank)
                )
                for promotion_piece in PROMOTION_PIECES:
                    moves.add(base + promotion_piece)

    return moves


def _is_standard_piece_vector(file_delta: int, rank_delta: int) -> bool:
    abs_file = abs(file_delta)
    abs_rank = abs(rank_delta)
    if file_delta == 0 or rank_delta == 0:
        return True
    if abs_file == abs_rank:
        return True
    return (abs_file, abs_rank) in {(1, 2), (2, 1)}


_DEFAULT_VOCABULARY: MoveVocabulary | None = None


def get_move_vocabulary() -> MoveVocabulary:
    global _DEFAULT_VOCABULARY
    if _DEFAULT_VOCABULARY is None:
        _DEFAULT_VOCABULARY = build_move_vocabulary()
    return _DEFAULT_VOCABULARY

