"""Shared configuration for the chess app."""

from __future__ import annotations

from dataclasses import dataclass
import chess

PROJECT_NAME = "CHESS_AI_MINIMAX_PROJECT"

BOARD_SIZE = 640
SIDEBAR_WIDTH = 280
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
SQUARE_SIZE = BOARD_SIZE // 8

FPS = 60

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_SQUARE = (246, 246, 105)
MOVE_HINT = (60, 180, 75)
TEXT_COLOR = (20, 20, 20)
BACKGROUND = (245, 245, 245)
PANEL_BACKGROUND = (232, 232, 232)
PANEL_BORDER = (190, 190, 190)


@dataclass(frozen=True)
class AIModeConfig:
    key: str
    label: str
    evaluation_mode: str | None
    uses_neural: bool
    order_with_neural: bool = False


AI_MODE_RANDOM = "random"
AI_MODE_MINIMAX = "minimax"
AI_MODE_MINIMAX_POLICY_ORDERED = "minimax_policy_ordered"
AI_MODE_MINIMAX_NN = "minimax_nn"
AI_MODE_MINIMAX_HYBRID = "minimax_hybrid"

AI_MODES = {
    AI_MODE_RANDOM: AIModeConfig(
        key=AI_MODE_RANDOM,
        label="Random",
        evaluation_mode=None,
        uses_neural=False,
    ),
    AI_MODE_MINIMAX: AIModeConfig(
        key=AI_MODE_MINIMAX,
        label="Minimax",
        evaluation_mode="classic",
        uses_neural=False,
    ),
    AI_MODE_MINIMAX_POLICY_ORDERED: AIModeConfig(
        key=AI_MODE_MINIMAX_POLICY_ORDERED,
        label="Neural-Ordered Minimax",
        evaluation_mode="classic",
        uses_neural=True,
        order_with_neural=True,
    ),
    AI_MODE_MINIMAX_NN: AIModeConfig(
        key=AI_MODE_MINIMAX_NN,
        label="Policy-Proxy Minimax",
        evaluation_mode="neural",
        uses_neural=True,
        order_with_neural=True,
    ),
    AI_MODE_MINIMAX_HYBRID: AIModeConfig(
        key=AI_MODE_MINIMAX_HYBRID,
        label="Hybrid Policy-Proxy",
        evaluation_mode="hybrid",
        uses_neural=True,
        order_with_neural=True,
    ),
}

DEFAULT_AI_MODE = AI_MODE_MINIMAX
MINIMAX_DEPTH = 4

STARTING_TIME_SECONDS = 600.0
MIN_THINK_TIME_SECONDS = 0.75
MAX_THINK_TIME_SECONDS = 12.0
THINK_TIME_FRACTION = 0.04
TIME_BUFFER_SECONDS = 0.05

NEURAL_MODEL_PATH = "models/lichess_policy_v1_best.pt"
HYBRID_NEURAL_WEIGHT = 0.35

VALID_AI_MODES = set(AI_MODES)
WHITE = chess.WHITE
BLACK = chess.BLACK


def get_ai_mode_label(mode: str) -> str:
    config = AI_MODES.get(mode)
    return config.label if config else mode
