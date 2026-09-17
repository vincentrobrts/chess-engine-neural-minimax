"""Evaluate a trained move-policy checkpoint on a held-out split."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from policy.checkpoint import load_policy_checkpoint
from training.common import (
    PolicyDataset,
    collate_policy_batch,
    empty_metric_totals,
    finalize_metrics,
    load_records,
    masked_logits,
    random_legal_baseline,
    update_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="models/lichess_policy_v1_best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_policy_checkpoint(args.checkpoint, device=device)
    records = load_records(args.data, args.max_positions)
    dataset = PolicyDataset(records, args.split)
    if not dataset:
        raise SystemExit(f"No records found for split {args.split}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_policy_batch,
    )
    metrics = evaluate(model, loader, device)
    baseline = random_legal_baseline(dataset.records)
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Split: {args.split}")
    print(f"Examples: {int(metrics['examples'])}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1 legal accuracy: {metrics['top1']:.4f}")
    print(f"Top-3 legal accuracy: {metrics['top3']:.4f}")
    print(f"Top-5 legal accuracy: {metrics['top5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Random legal top-1 baseline: {baseline['top1']:.4f}")
    print(f"Random legal MRR baseline: {baseline['mrr']:.4f}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = empty_metric_totals()
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for inputs, targets, legal_mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            legal_mask = legal_mask.to(device)
            logits = model(inputs)
            loss = loss_fn(masked_logits(logits, legal_mask), targets)
            update_metrics(totals, loss=loss, logits=logits, targets=targets, legal_mask=legal_mask)
    return finalize_metrics(totals)


if __name__ == "__main__":
    main()

