"""Runtime access to the project-owned move-policy checkpoint."""

from __future__ import annotations

from constants import NEURAL_MODEL_PATH
from policy.inference import PolicyMoveScorer

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


class NeuralMoveScorer(PolicyMoveScorer):
    """
    Backward-compatible name for the move-policy scorer used by the engine.

    The loaded model is a policy network. It ranks legal moves and can support
    the legacy policy-proxy modes, but it is not a scalar value network.
    """

    def __init__(self, model_path: str = NEURAL_MODEL_PATH, device: str = "cpu") -> None:
        super().__init__(checkpoint_path=model_path, device=device)


_DEFAULT_SCORER: NeuralMoveScorer | None = None


def get_neural_scorer() -> NeuralMoveScorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = NeuralMoveScorer()
    return _DEFAULT_SCORER
