"""Core game flow, clocks, and move handling."""

from __future__ import annotations

import time

import chess

from ai import choose_move
from constants import (
    AI_MODES,
    MAX_THINK_TIME_SECONDS,
    MIN_THINK_TIME_SECONDS,
    MINIMAX_DEPTH,
    STARTING_TIME_SECONDS,
    THINK_TIME_FRACTION,
    WHITE,
    get_ai_mode_label,
)
from opening_book import OpeningBook, OpeningLine


class ChessGame:
    """Coordinates board state, clocks, user input, and AI turns."""

    def __init__(
        self,
        ai_mode: str,
        human_color: bool = chess.WHITE,
        minimax_depth: int = MINIMAX_DEPTH,
    ) -> None:
        if ai_mode not in AI_MODES:
            raise ValueError(f"AI mode must be one of {set(AI_MODES)}")

        self.board = chess.Board()
        self.ai_mode = ai_mode
        self.minimax_depth = minimax_depth
        self.human_color = human_color
        self.selected_square: int | None = None
        self.last_move: chess.Move | None = None
        self.move_history_san: list[str] = []

        self.white_time = STARTING_TIME_SECONDS
        self.black_time = STARTING_TIME_SECONDS
        self.turn_started_at = time.perf_counter()
        self.timeout_winner: str | None = None

        self.opening_book = OpeningBook()
        self.active_opening: OpeningLine | None = None
        self.opening_book_enabled = True
        self.opening_status = f"Opening book ready ({len(self.opening_book.lines)} lines loaded)."

    @property
    def is_game_over(self) -> bool:
        return self.timeout_winner is not None or self.board.is_game_over()

    def is_human_turn(self) -> bool:
        return self.board.turn == self.human_color

    def set_ai_mode(self, ai_mode: str) -> None:
        if ai_mode not in AI_MODES:
            raise ValueError(f"AI mode must be one of {set(AI_MODES)}")
        self.ai_mode = ai_mode

    def update_clock(self) -> None:
        if self.is_game_over:
            return

        now = time.perf_counter()
        elapsed = now - self.turn_started_at
        if elapsed <= 0:
            return

        if self.board.turn == WHITE:
            self.white_time = max(0.0, self.white_time - elapsed)
            if self.white_time == 0.0:
                self.timeout_winner = "Black"
        else:
            self.black_time = max(0.0, self.black_time - elapsed)
            if self.black_time == 0.0:
                self.timeout_winner = "White"

        self.turn_started_at = now

    def get_clock_text(self) -> tuple[str, str]:
        return format_clock(self.white_time), format_clock(self.black_time)

    def get_legal_targets(self, from_square: int | None) -> list[int]:
        if from_square is None:
            return []

        return [
            move.to_square
            for move in self.board.legal_moves
            if move.from_square == from_square
        ]

    def handle_human_click(self, square: int) -> bool:
        if self.is_game_over or not self.is_human_turn():
            return False

        piece = self.board.piece_at(square)

        if self.selected_square is None:
            if piece and piece.color == self.human_color:
                self.selected_square = square
            return False

        if piece and piece.color == self.human_color:
            self.selected_square = square
            return False

        chosen_move = self._find_legal_move(self.selected_square, square)
        if chosen_move is None:
            self.selected_square = None
            return False

        self._push_recorded_move(chosen_move)
        self.selected_square = None
        return True

    def get_ai_time_budget(self) -> float:
        remaining_time = self.white_time if self.board.turn == WHITE else self.black_time
        if remaining_time <= 0:
            return 0.0

        budget = remaining_time * THINK_TIME_FRACTION
        budget = max(MIN_THINK_TIME_SECONDS, budget)
        budget = min(MAX_THINK_TIME_SECONDS, budget)
        return min(budget, max(0.0, remaining_time - 0.05))

    def choose_ai_move(self, time_budget: float | None = None) -> chess.Move | None:
        if self.is_game_over or self.is_human_turn():
            return None

        move = self._get_opening_book_move()
        if move is not None:
            return move

        if time_budget is None:
            time_budget = self.get_ai_time_budget()

        return choose_move(
            board=self.board,
            mode=self.ai_mode,
            depth=self.minimax_depth,
            time_limit_seconds=time_budget,
        )

    def apply_ai_move(self, move: chess.Move | None) -> bool:
        if move is None or move not in self.board.legal_moves or self.is_game_over or self.is_human_turn():
            return False

        self._push_recorded_move(move)
        return True

    def status_text(self, ai_thinking: bool = False) -> str:
        if self.timeout_winner is not None:
            return f"Time out! {self.timeout_winner} wins on time."
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins."
        if self.board.is_stalemate():
            return "Stalemate."
        if self.board.is_insufficient_material():
            return "Draw by insufficient material."
        if self.board.is_seventyfive_moves():
            return "Draw by 75-move rule."
        if self.board.is_fivefold_repetition():
            return "Draw by fivefold repetition."

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        player = "Human" if self.is_human_turn() else "AI"
        if ai_thinking and not self.is_human_turn():
            player = "AI thinking"

        human_side = "White" if self.human_color == chess.WHITE else "Black"
        mode_label = get_ai_mode_label(self.ai_mode)
        return f"{turn}: {player}. You are {human_side}. Mode: {mode_label}."

    def turn_indicator_text(self, ai_thinking: bool = False) -> str:
        if self.timeout_winner is not None:
            return f"{self.timeout_winner} wins on time"
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"{winner} wins by checkmate"
        if self.board.is_stalemate():
            return "Stalemate"
        if self.board.is_insufficient_material():
            return "Draw"

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        if ai_thinking and not self.is_human_turn():
            return f"{turn} AI thinking"
        return f"{turn} to move"

    def get_opening_text(self) -> str:
        return self.opening_status

    def formatted_move_history(self) -> list[str]:
        lines: list[str] = []
        for index in range(0, len(self.move_history_san), 2):
            move_number = index // 2 + 1
            white_move = self.move_history_san[index]
            black_move = self.move_history_san[index + 1] if index + 1 < len(self.move_history_san) else ""
            line = f"{move_number}. {white_move}"
            if black_move:
                line += f" {black_move}"
            lines.append(line)
        return lines

    def _find_legal_move(self, from_square: int, to_square: int) -> chess.Move | None:
        queen_promotion = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        if queen_promotion in self.board.legal_moves:
            return queen_promotion

        basic_move = chess.Move(from_square, to_square)
        if basic_move in self.board.legal_moves:
            return basic_move
        return None

    def _push_recorded_move(self, move: chess.Move) -> None:
        self.update_clock()

        san_move = self.board.san(move)
        self.move_history_san.append(san_move)
        self.last_move = move
        self.board.push(move)
        self.turn_started_at = time.perf_counter()
        self._update_opening_status()

    def _get_opening_book_move(self) -> chess.Move | None:
        if not self.opening_book_enabled:
            return None

        if self.active_opening is None:
            self.active_opening = self.opening_book.choose_opening(self.move_history_san, self.board.turn)
            if self.active_opening is None:
                self._disable_opening_book("No matching opening line found. Using normal AI.")
                return None

            self.opening_status = (
                f"Following opening: {self.active_opening.name} ({self.active_opening.eco})"
            )

        next_san = self.active_opening.next_san(self.move_history_san, self.board.turn)
        if next_san is None:
            self._disable_opening_book("Opening line ended. Using normal AI.")
            return None

        move = self.opening_book.san_to_move(self.board, next_san)
        if move is None:
            self._disable_opening_book("Opening line could not be parsed. Using normal AI.")
            return None

        return move

    def _update_opening_status(self) -> None:
        if self.active_opening is None or not self.opening_book_enabled:
            return

        if not self.active_opening.matches_history(self.move_history_san):
            self._disable_opening_book("Line was broken. Using normal AI.")
            return

        if len(self.move_history_san) >= len(self.active_opening.san_moves):
            self._disable_opening_book("Opening line finished. Using normal AI.")

    def _disable_opening_book(self, reason: str) -> None:
        if self.active_opening is not None:
            self.opening_status = f"{self.active_opening.name} ({self.active_opening.eco}) - {reason}"
        else:
            self.opening_status = reason

        self.active_opening = None
        self.opening_book_enabled = False


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes = total_seconds // 60
    remainder = total_seconds % 60
    return f"{minutes:02d}:{remainder:02d}"
