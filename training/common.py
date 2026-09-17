"""Shared dataset and metric utilities for policy training."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from policy.encoding import encode_board
from policy.vocab import get_move_vocabulary


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str | Path, max_positions: int | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
            if max_positions is not None and len(records) >= max_positions:
                break
    return records


def load_manifest_for_data(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


class PolicyDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], split: str) -> None:
        self.records = [record for record in records if record["split"] == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_policy_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocabulary = get_move_vocabulary()
    inputs = np.stack([encode_board(chess.Board(record["fen"])) for record in batch])
    targets = torch.tensor([record["target_id"] for record in batch], dtype=torch.long)
    legal_mask = torch.zeros((len(batch), len(vocabulary.moves)), dtype=torch.bool)
    for row_index, record in enumerate(batch):
        legal_mask[row_index, record["legal_ids"]] = True
    return torch.from_numpy(inputs), targets, legal_mask


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask.to(logits.device), -1.0e9)


def random_legal_baseline(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"top1": 0.0, "mrr": 0.0}
    top1 = 0.0
    mrr = 0.0
    for record in records:
        legal_count = max(1, len(record["legal_ids"]))
        top1 += 1.0 / legal_count
        mrr += sum(1.0 / rank for rank in range(1, legal_count + 1)) / legal_count
    total = float(len(records))
    return {"top1": top1 / total, "mrr": mrr / total}


def empty_metric_totals() -> Counter[str]:
    return Counter(
        {
            "examples": 0,
            "loss_sum": 0.0,
            "top1": 0,
            "top3": 0,
            "top5": 0,
            "mrr_sum": 0.0,
        }
    )


def update_metrics(
    totals: Counter[str],
    *,
    loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> None:
    batch_size = targets.shape[0]
    totals["examples"] += batch_size
    totals["loss_sum"] += float(loss.detach().cpu()) * batch_size

    ranked = torch.argsort(masked_logits(logits.detach(), legal_mask), dim=1, descending=True)
    targets_cpu = targets.detach().cpu()
    ranked_cpu = ranked.cpu()
    for index in range(batch_size):
        target = int(targets_cpu[index])
        row = ranked_cpu[index].tolist()
        rank = row.index(target) + 1
        totals["top1"] += 1 if rank <= 1 else 0
        totals["top3"] += 1 if rank <= 3 else 0
        totals["top5"] += 1 if rank <= 5 else 0
        totals["mrr_sum"] += 1.0 / rank


def finalize_metrics(totals: Counter[str]) -> dict[str, float]:
    examples = max(1, totals["examples"])
    return {
        "loss": totals["loss_sum"] / examples,
        "top1": totals["top1"] / examples,
        "top3": totals["top3"] / examples,
        "top5": totals["top5"] / examples,
        "mrr": totals["mrr_sum"] / examples,
        "examples": float(totals["examples"]),
    }
