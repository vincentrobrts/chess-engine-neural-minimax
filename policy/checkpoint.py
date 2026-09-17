"""Checkpoint helpers for the project-owned move-policy network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from policy.encoding import INPUT_PLANES
from policy.model import ChessPolicyNet, PolicyModelConfig
from policy.vocab import MoveVocabulary, get_move_vocabulary


def save_policy_checkpoint(
    path: str | Path,
    *,
    model: ChessPolicyNet,
    epoch: int,
    validation_metrics: dict[str, float],
    training_config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    seed: int,
) -> None:
    vocabulary = get_move_vocabulary()
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "input_planes": INPUT_PLANES,
        "move_vocabulary_size": len(vocabulary.moves),
        "move_vocabulary_checksum": vocabulary.checksum,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "training_config": training_config,
        "dataset_manifest": dataset_manifest,
        "seed": seed,
        "pytorch_version": str(torch.__version__),
        "data_source": dataset_manifest.get("source_url", "unknown"),
        "data_license": dataset_manifest.get("source_license", "unknown"),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    vocabulary: MoveVocabulary | None = None,
) -> tuple[ChessPolicyNet, dict[str, Any]]:
    vocab = vocabulary or get_move_vocabulary()
    checkpoint = torch.load(Path(path), map_location=device)

    expected_size = len(vocab.moves)
    actual_size = checkpoint.get("move_vocabulary_size")
    if actual_size != expected_size:
        raise ValueError(
            f"Checkpoint vocabulary size {actual_size} does not match expected {expected_size}."
        )

    expected_checksum = vocab.checksum
    actual_checksum = checkpoint.get("move_vocabulary_checksum")
    if actual_checksum != expected_checksum:
        raise ValueError("Checkpoint move vocabulary checksum does not match this codebase.")

    if checkpoint.get("input_planes") != INPUT_PLANES:
        raise ValueError(
            f"Checkpoint input planes {checkpoint.get('input_planes')} does not match {INPUT_PLANES}."
        )

    model_config = PolicyModelConfig(**checkpoint["model_config"])
    model = ChessPolicyNet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
