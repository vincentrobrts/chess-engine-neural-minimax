"""Board encoding for the project-owned move-policy network."""

from __future__ import annotations

import hashlib

import chess
import numpy as np

INPUT_PLANES = 18
BOARD_SHAPE = (INPUT_PLANES, 8, 8)

PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

SIDE_TO_MOVE_PLANE = 12
WHITE_KINGSIDE_CASTLE_PLANE = 13
WHITE_QUEENSIDE_CASTLE_PLANE = 14
BLACK_KINGSIDE_CASTLE_PLANE = 15
BLACK_QUEENSIDE_CASTLE_PLANE = 16
EN_PASSANT_PLANE = 17


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board from White's perspective as 18 float32 planes."""
    matrix = np.zeros(BOARD_SHAPE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[(piece.color, piece.piece_type)]
        row, col = square_to_plane_coords(square)
        matrix[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        matrix[SIDE_TO_MOVE_PLANE, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        matrix[WHITE_KINGSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        matrix[WHITE_QUEENSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        matrix[BLACK_KINGSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        matrix[BLACK_QUEENSIDE_CASTLE_PLANE, :, :] = 1.0

    if board.ep_square is not None:
        row, col = square_to_plane_coords(board.ep_square)
        matrix[EN_PASSANT_PLANE, row, col] = 1.0

    return matrix


def square_to_plane_coords(square: int) -> tuple[int, int]:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    return 7 - rank_index, file_index


def normalized_position_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def position_key(board: chess.Board) -> str:
    return hashlib.sha256(normalized_position_fen(board).encode("utf-8")).hexdigest()

