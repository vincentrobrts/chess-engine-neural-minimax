"""Load and query a lightweight opening book from TSV files."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import chess

RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MOVE_NUMBER_PATTERN = re.compile(r"^\d+\.(\.\.)?$")


@dataclass(frozen=True)
class OpeningLine:
    """A single opening line stored as SAN moves."""

    eco: str
    name: str
    san_moves: list[str]

    def matches_history(self, history: list[str]) -> bool:
        return self.san_moves[: len(history)] == history

    def next_san(self, history: list[str], turn: bool) -> str | None:
        if not self.matches_history(history) or len(history) >= len(self.san_moves):
            return None

        expected_turn = chess.WHITE if len(history) % 2 == 0 else chess.BLACK
        if turn != expected_turn:
            return None
        return self.san_moves[len(history)]


class OpeningBook:
    """Loads opening lines and chooses a matching line when needed."""

    def __init__(self, openings_dir: str = "openings") -> None:
        base_dir = Path(__file__).resolve().parent
        self.openings_dir = base_dir / openings_dir
        self.lines = self._load_lines()

    def choose_opening(self, history: list[str], turn: bool) -> OpeningLine | None:
        candidates = [
            line
            for line in self.lines
            if line.next_san(history, turn) is not None
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    @staticmethod
    def san_to_move(board: chess.Board, san: str) -> chess.Move | None:
        try:
            return board.parse_san(san)
        except ValueError:
            return None

    def _load_lines(self) -> list[OpeningLine]:
        lines: list[OpeningLine] = []
        if not self.openings_dir.exists():
            return lines

        for path in sorted(self.openings_dir.glob("*.tsv")):
            with path.open("r", encoding="utf-8", newline="") as tsv_file:
                reader = csv.DictReader(tsv_file, delimiter="\t")
                for row in reader:
                    san_moves = self._normalize_pgn(row["pgn"])
                    if san_moves:
                        lines.append(
                            OpeningLine(
                                eco=row["eco"],
                                name=row["name"],
                                san_moves=san_moves,
                            )
                        )
        return lines

    def _normalize_pgn(self, pgn_text: str) -> list[str]:
        board = chess.Board()
        san_moves: list[str] = []

        for token in pgn_text.split():
            if token in RESULT_TOKENS or MOVE_NUMBER_PATTERN.fullmatch(token):
                continue

            move = self.san_to_move(board, token)
            if move is None:
                return []

            san_moves.append(board.san(move))
            board.push(move)

        return san_moves
