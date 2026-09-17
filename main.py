"""Entry point for the pygame chess game."""

from __future__ import annotations

import threading

import chess
import pygame

from constants import DEFAULT_AI_MODE, FPS, MINIMAX_DEPTH
from game import ChessGame
from ui import (
    create_fonts,
    create_window,
    draw_color_selection_screen,
    draw_screen,
    load_piece_images,
    mode_at_position,
    mouse_to_square,
)


def choose_player_color(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> bool | None:
    white_button, black_button = draw_color_selection_screen(screen, title_font, body_font)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if white_button.collidepoint(event.pos):
                    return chess.WHITE
                if black_button.collidepoint(event.pos):
                    return chess.BLACK


def start_ai_search(game: ChessGame, ai_result: dict[str, chess.Move | None]) -> threading.Thread:
    """
    Run AI search on a background thread so the UI stays responsive while the
    chess clock continues updating.
    """
    board_snapshot = game.board.copy(stack=True)
    ai_mode = game.ai_mode
    ai_depth = game.minimax_depth
    think_time = game.get_ai_time_budget()

    def worker() -> None:
        search_game = ChessGame(ai_mode=ai_mode, human_color=game.human_color, minimax_depth=ai_depth)
        search_game.board = board_snapshot
        search_game.move_history_san = game.move_history_san[:]
        search_game.opening_book = game.opening_book
        search_game.active_opening = game.active_opening
        search_game.opening_book_enabled = game.opening_book_enabled
        search_game.opening_status = game.opening_status
        ai_result["move"] = search_game.choose_ai_move(time_budget=think_time)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def run_game(ai_mode: str = DEFAULT_AI_MODE, minimax_depth: int = MINIMAX_DEPTH) -> None:
    screen = create_window()
    piece_images = load_piece_images()
    title_font, body_font, small_font = create_fonts()
    human_color = choose_player_color(screen, title_font, body_font)
    if human_color is None:
        pygame.quit()
        return

    game = ChessGame(ai_mode=ai_mode, human_color=human_color, minimax_depth=minimax_depth)
    clock = pygame.time.Clock()
    ai_thread: threading.Thread | None = None
    ai_result: dict[str, chess.Move | None] = {"move": None}

    running = True
    while running:
        clock.tick(FPS)
        game.update_clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and ai_thread is None:
                selected_mode = mode_at_position(*event.pos)
                if selected_mode is not None:
                    game.set_ai_mode(selected_mode)
                elif game.is_human_turn():
                    square = mouse_to_square(*event.pos, game.human_color)
                    if square is not None:
                        game.handle_human_click(square)

        if not game.is_game_over and not game.is_human_turn():
            if ai_thread is None:
                ai_result = {"move": None}
                ai_thread = start_ai_search(game, ai_result)
            elif not ai_thread.is_alive():
                fallback_move = next(iter(game.board.legal_moves), None)
                chosen_move = ai_result.get("move") or fallback_move
                game.apply_ai_move(chosen_move)
                ai_thread = None

        draw_screen(
            screen=screen,
            board=game.board,
            piece_images=piece_images,
            title_font=title_font,
            body_font=body_font,
            small_font=small_font,
            status_text=game.status_text(ai_thinking=ai_thread is not None),
            turn_text=game.turn_indicator_text(ai_thinking=ai_thread is not None),
            opening_text=game.get_opening_text(),
            move_history_lines=game.formatted_move_history(),
            clock_text=game.get_clock_text(),
            last_move=game.last_move,
            human_color=game.human_color,
            ai_mode=game.ai_mode,
            ai_thinking=ai_thread is not None,
            selected_square=game.selected_square,
            legal_targets=game.get_legal_targets(game.selected_square),
        )

    pygame.quit()


if __name__ == "__main__":
    run_game()
