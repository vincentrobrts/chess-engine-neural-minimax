"""Pygame drawing helpers for the chess UI."""

from pathlib import Path

import chess
import pygame

from constants import (
    AI_MODES,
    BACKGROUND,
    BOARD_SIZE,
    DARK_SQUARE,
    LIGHT_SQUARE,
    MOVE_HINT,
    PANEL_BACKGROUND,
    PANEL_BORDER,
    PROJECT_NAME,
    SELECTED_SQUARE,
    SIDEBAR_WIDTH,
    SQUARE_SIZE,
    TEXT_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    get_ai_mode_label,
)

PIECE_IMAGE_FILES = {
    "P": "Chess_plt60.png",
    "N": "Chess_nlt60.png",
    "B": "Chess_blt60.png",
    "R": "Chess_rlt60.png",
    "Q": "Chess_qlt60.png",
    "K": "Chess_klt60.png",
    "p": "Chess_pdt60.png",
    "n": "Chess_ndt60.png",
    "b": "Chess_bdt60.png",
    "r": "Chess_rdt60.png",
    "q": "Chess_qdt60.png",
    "k": "Chess_kdt60.png",
}

ARROW_COLOR = (30, 144, 255, 120)
TURN_PANEL_COLOR = (220, 235, 252)
TURN_PANEL_BORDER = (110, 150, 195)
MODE_BUTTON_COLOR = (246, 246, 246)
MODE_BUTTON_ACTIVE = (210, 230, 255)
MODE_BUTTON_DISABLED = (218, 218, 218)
MODE_BUTTON_BORDER = (160, 170, 185)
PANEL_MARGIN = 16
CONTENT_WIDTH = SIDEBAR_WIDTH - (PANEL_MARGIN * 2)
MODE_BUTTON_HEIGHT = 23
MODE_BUTTON_GAP = 3
MODE_SELECTOR_TOP = 214


def create_window() -> pygame.Surface:
    pygame.init()
    pygame.display.set_caption(PROJECT_NAME)
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def create_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    title_font = pygame.font.SysFont("arial", 26, bold=True)
    body_font = pygame.font.SysFont("arial", 21)
    small_font = pygame.font.SysFont("consolas", 18)
    return title_font, body_font, small_font


def load_piece_images() -> dict[str, pygame.Surface]:
    base_dir = Path(__file__).resolve().parent
    images: dict[str, pygame.Surface] = {}

    for piece_symbol, filename in PIECE_IMAGE_FILES.items():
        image_path = base_dir / "pieces" / filename
        image = pygame.image.load(str(image_path)).convert_alpha()
        images[piece_symbol] = pygame.transform.smoothscale(image, (SQUARE_SIZE - 10, SQUARE_SIZE - 10))

    return images


def draw_screen(
    screen: pygame.Surface,
    board: chess.Board,
    piece_images: dict[str, pygame.Surface],
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    status_text: str,
    turn_text: str,
    opening_text: str,
    move_history_lines: list[str],
    clock_text: tuple[str, str],
    last_move: chess.Move | None,
    human_color: bool,
    ai_mode: str,
    ai_thinking: bool,
    selected_square: int | None,
    legal_targets: list[int],
) -> None:
    screen.fill(BACKGROUND)
    draw_board(screen, human_color, selected_square, legal_targets)
    draw_last_move_arrow(screen, last_move, human_color)
    draw_pieces(screen, board, piece_images, human_color)
    draw_sidebar(
        screen,
        title_font,
        body_font,
        small_font,
        status_text,
        turn_text,
        opening_text,
        move_history_lines,
        clock_text,
        ai_mode,
        ai_thinking,
    )
    pygame.display.flip()


def draw_board(
    screen: pygame.Surface,
    human_color: bool,
    selected_square: int | None,
    legal_targets: list[int],
) -> None:
    for row in range(8):
        for col in range(8):
            square_color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, square_color, rect)

    if selected_square is not None:
        row, col = square_to_row_col(selected_square, human_color)
        rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, SELECTED_SQUARE, rect, width=5)

    for target_square in legal_targets:
        row, col = square_to_row_col(target_square, human_color)
        center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
        pygame.draw.circle(screen, MOVE_HINT, center, 10)


def draw_last_move_arrow(
    screen: pygame.Surface,
    last_move: chess.Move | None,
    human_color: bool,
) -> None:
    """Draw a subtle translucent arrow for the most recent move."""
    if last_move is None:
        return

    start_row, start_col = square_to_row_col(last_move.from_square, human_color)
    end_row, end_col = square_to_row_col(last_move.to_square, human_color)

    start_pos = square_center(start_row, start_col)
    end_pos = square_center(end_row, end_col)

    overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    pygame.draw.line(overlay, ARROW_COLOR, start_pos, end_pos, 10)

    direction = pygame.Vector2(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    if direction.length_squared() > 0:
        direction = direction.normalize()
        arrow_length = 18
        arrow_width = 10
        tip = pygame.Vector2(end_pos)
        back = tip - direction * arrow_length
        left = back + pygame.Vector2(-direction.y, direction.x) * arrow_width
        right = back + pygame.Vector2(direction.y, -direction.x) * arrow_width
        pygame.draw.polygon(overlay, ARROW_COLOR, [tip, left, right])

    screen.blit(overlay, (0, 0))


def draw_pieces(
    screen: pygame.Surface,
    board: chess.Board,
    piece_images: dict[str, pygame.Surface],
    human_color: bool,
) -> None:
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        row, col = square_to_row_col(square, human_color)
        image = piece_images[piece.symbol()]
        x = col * SQUARE_SIZE + (SQUARE_SIZE - image.get_width()) // 2
        y = row * SQUARE_SIZE + (SQUARE_SIZE - image.get_height()) // 2
        screen.blit(image, (x, y))


def draw_sidebar(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    status_text: str,
    turn_text: str,
    opening_text: str,
    move_history_lines: list[str],
    clock_text: tuple[str, str],
    ai_mode: str,
    ai_thinking: bool,
) -> None:
    panel_x = BOARD_SIZE
    panel_rect = pygame.Rect(panel_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, PANEL_BACKGROUND, panel_rect)
    pygame.draw.line(screen, PANEL_BORDER, (panel_x, 0), (panel_x, WINDOW_HEIGHT), width=2)

    content_x = panel_x + PANEL_MARGIN
    title_rect = pygame.Rect(content_x, 12, CONTENT_WIDTH, 58)
    draw_text_in_rect(screen, PROJECT_NAME.replace("_", " "), title_font, title_rect)

    turn_rect = pygame.Rect(content_x, 76, CONTENT_WIDTH, 46)
    pygame.draw.rect(screen, TURN_PANEL_COLOR, turn_rect, border_radius=8)
    pygame.draw.rect(screen, TURN_PANEL_BORDER, turn_rect, width=2, border_radius=8)
    turn_font = title_font if title_font.size(turn_text)[0] <= turn_rect.width - 24 else body_font
    draw_single_line_text(screen, turn_text, turn_font, turn_rect.inflate(-20, -8))

    white_clock = body_font.render(f"White: {clock_text[0]}", True, TEXT_COLOR)
    black_clock = body_font.render(f"Black: {clock_text[1]}", True, TEXT_COLOR)
    screen.blit(white_clock, (content_x, 136))
    screen.blit(black_clock, (content_x, 166))

    draw_ai_mode_selector(screen, small_font, ai_mode, ai_thinking)

    draw_section_label(screen, "Status", small_font, content_x, 350)
    status_rect = pygame.Rect(content_x, 370, CONTENT_WIDTH, 48)
    draw_text_box(screen, status_text, body_font, status_rect)

    draw_section_label(screen, "Opening", small_font, content_x, 426)
    opening_rect = pygame.Rect(content_x, 446, CONTENT_WIDTH, 48)
    draw_text_box(screen, opening_text, small_font, opening_rect)

    moves_label = title_font.render("Moves", True, TEXT_COLOR)
    screen.blit(moves_label, (content_x, 510))

    y_position = 544
    for move_line in move_history_lines[-4:]:
        move_surface = small_font.render(move_line, True, TEXT_COLOR)
        screen.blit(move_surface, (content_x, y_position))
        y_position += 22


def draw_ai_mode_selector(
    screen: pygame.Surface,
    font: pygame.font.Font,
    current_mode: str,
    disabled: bool,
) -> None:
    content_x = BOARD_SIZE + PANEL_MARGIN
    draw_section_label(screen, "AI Mode", font, content_x, 194)

    for mode, rect in get_ai_mode_button_rects().items():
        active = mode == current_mode
        if active:
            fill_color = MODE_BUTTON_ACTIVE
        elif disabled:
            fill_color = MODE_BUTTON_DISABLED
        else:
            fill_color = MODE_BUTTON_COLOR

        pygame.draw.rect(screen, fill_color, rect, border_radius=6)
        pygame.draw.rect(screen, MODE_BUTTON_BORDER, rect, width=1, border_radius=6)

        text_color = TEXT_COLOR if active or not disabled else (120, 120, 120)
        label = fit_with_ellipsis(get_ai_mode_label(mode), font, rect.width - 18)
        text_surface = font.render(label, True, text_color)
        screen.blit(text_surface, (rect.x + 9, rect.centery - text_surface.get_height() // 2))


def get_ai_mode_button_rects() -> dict[str, pygame.Rect]:
    rects: dict[str, pygame.Rect] = {}
    x = BOARD_SIZE + PANEL_MARGIN
    y = MODE_SELECTOR_TOP
    for mode in AI_MODES:
        rects[mode] = pygame.Rect(x, y, CONTENT_WIDTH, MODE_BUTTON_HEIGHT)
        y += MODE_BUTTON_HEIGHT + MODE_BUTTON_GAP
    return rects


def mode_at_position(mouse_x: int, mouse_y: int) -> str | None:
    for mode, rect in get_ai_mode_button_rects().items():
        if rect.collidepoint(mouse_x, mouse_y):
            return mode
    return None


def draw_section_label(
    screen: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    x: int,
    y: int,
) -> None:
    label_surface = font.render(text, True, TEXT_COLOR)
    screen.blit(label_surface, (x, y))


def draw_text_box(
    screen: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(screen, MODE_BUTTON_COLOR, rect, border_radius=6)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=1, border_radius=6)
    draw_text_in_rect(screen, text, font, rect.inflate(-14, -10))


def draw_text_in_rect(
    screen: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    rect: pygame.Rect,
) -> None:
    line_height = font.get_linesize()
    max_lines = max(1, rect.height // line_height)
    lines = wrap_text(text, font, rect.width)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_with_ellipsis(lines[-1], font, rect.width)

    for index, line in enumerate(lines):
        line_surface = font.render(line, True, TEXT_COLOR)
        screen.blit(line_surface, (rect.x, rect.y + index * line_height))


def draw_single_line_text(
    screen: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    rect: pygame.Rect,
) -> None:
    fitted_text = fit_with_ellipsis(text, font, rect.width)
    text_surface = font.render(fitted_text, True, TEXT_COLOR)
    screen.blit(text_surface, (rect.x, rect.centery - text_surface.get_height() // 2))


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    lines: list[str] = []

    for raw_line in text.splitlines() or [""]:
        words = raw_line.split()
        if not words:
            lines.append("")
            continue

        current_line = ""
        for word in words:
            trial_line = word if not current_line else f"{current_line} {word}"
            if font.size(trial_line)[0] <= max_width:
                current_line = trial_line
                continue

            if current_line:
                lines.append(current_line)
                if font.size(word)[0] <= max_width:
                    current_line = word
                else:
                    lines.append(fit_with_ellipsis(word, font, max_width))
                    current_line = ""
            else:
                lines.append(fit_with_ellipsis(word, font, max_width))
                current_line = ""

        if current_line:
            lines.append(current_line)

    return lines


def fit_with_ellipsis(text: str, font: pygame.font.Font, max_width: int) -> str:
    if font.size(text)[0] <= max_width:
        return text

    ellipsis = "..."
    if font.size(ellipsis)[0] > max_width:
        return ""

    shortened = text
    while shortened and font.size(shortened.rstrip() + ellipsis)[0] > max_width:
        shortened = shortened[:-1]
    return shortened.rstrip() + ellipsis if shortened else ellipsis


def draw_color_selection_screen(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> tuple[pygame.Rect, pygame.Rect]:
    screen.fill(BACKGROUND)

    title = title_font.render("Choose Your Color", True, TEXT_COLOR)
    prompt = body_font.render("Pick the White pieces or Black pieces.", True, TEXT_COLOR)

    white_button = pygame.Rect(140, 250, 180, 70)
    black_button = pygame.Rect(360, 250, 180, 70)

    screen.blit(title, ((BOARD_SIZE - title.get_width()) // 2, 120))
    screen.blit(prompt, ((BOARD_SIZE - prompt.get_width()) // 2, 170))

    pygame.draw.rect(screen, (250, 250, 250), white_button, border_radius=8)
    pygame.draw.rect(screen, (50, 50, 50), black_button, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER, white_button, width=2, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER, black_button, width=2, border_radius=8)

    white_text = body_font.render("Play White", True, TEXT_COLOR)
    black_text = body_font.render("Play Black", True, (245, 245, 245))
    screen.blit(
        white_text,
        (white_button.centerx - white_text.get_width() // 2, white_button.centery - white_text.get_height() // 2),
    )
    screen.blit(
        black_text,
        (black_button.centerx - black_text.get_width() // 2, black_button.centery - black_text.get_height() // 2),
    )

    pygame.display.flip()
    return white_button, black_button


def mouse_to_square(mouse_x: int, mouse_y: int, human_color: bool) -> int | None:
    if mouse_x < 0 or mouse_x >= BOARD_SIZE or mouse_y < 0 or mouse_y >= BOARD_SIZE:
        return None

    col = mouse_x // SQUARE_SIZE
    row = mouse_y // SQUARE_SIZE
    if human_color == chess.WHITE:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def square_to_row_col(square: int, human_color: bool) -> tuple[int, int]:
    col = chess.square_file(square)
    rank = chess.square_rank(square)
    if human_color == chess.WHITE:
        return 7 - rank, col
    return rank, 7 - col


def square_center(row: int, col: int) -> tuple[int, int]:
    return (
        col * SQUARE_SIZE + SQUARE_SIZE // 2,
        row * SQUARE_SIZE + SQUARE_SIZE // 2,
    )
