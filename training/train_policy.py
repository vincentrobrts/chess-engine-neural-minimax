"""Train the project-owned chess move-policy network."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from policy.checkpoint import save_policy_checkpoint
from policy.model import build_policy_model, count_parameters
from policy.vocab import get_move_vocabulary
from training.common import (
    PolicyDataset,
    collate_policy_batch,
    empty_metric_totals,
    finalize_metrics,
    load_manifest_for_data,
    load_records,
    masked_logits,
    random_legal_baseline,
    set_global_seed,
    update_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a move-policy network.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="models/lichess_policy_v1")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    records = load_records(args.data, args.max_positions)
    manifest = load_manifest_for_data(args.data)
    train_dataset = PolicyDataset(records, "train")
    val_dataset = PolicyDataset(records, "val")
    test_dataset = PolicyDataset(records, "test")
    if not train_dataset or not val_dataset:
        raise SystemExit("Dataset must contain at least one train and one validation record.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_policy_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_policy_batch,
    )

    vocabulary = get_move_vocabulary()
    model = build_policy_model(vocab_size=len(vocabulary.moves)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    output_prefix = Path(args.output)
    best_path = output_prefix.with_name(output_prefix.name + "_best.pt")
    latest_path = output_prefix.with_name(output_prefix.name + "_latest.pt")
    training_config = vars(args).copy()
    training_config["device_resolved"] = str(device)
    training_config["parameter_count"] = count_parameters(model)
    training_config["vocabulary_checksum"] = vocabulary.checksum

    print(f"Device: {device}")
    print(f"Train/val/test examples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Random legal baseline: {random_legal_baseline(val_dataset.records)}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    total_train_examples = 0
    total_train_seconds = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        train_seconds = time.perf_counter() - epoch_started_at
        val_metrics = evaluate(model, val_loader, device)
        total_train_examples += int(train_metrics["examples"])
        total_train_seconds += train_seconds

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_top3={val_metrics['top3']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} "
            f"seconds={train_seconds:.2f}"
        )

        save_policy_checkpoint(
            latest_path,
            model=model,
            epoch=epoch,
            validation_metrics=val_metrics,
            training_config=training_config,
            dataset_manifest=manifest,
            seed=args.seed,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_policy_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                validation_metrics=val_metrics,
                training_config=training_config,
                dataset_manifest=manifest,
                seed=args.seed,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    examples_per_second = (
        total_train_examples / total_train_seconds if total_train_seconds > 0 else 0.0
    )
    seconds_per_epoch = total_train_seconds / max(1, epoch - start_epoch + 1)
    print(f"Training examples/sec: {examples_per_second:.1f}")
    print(f"Seconds/epoch: {seconds_per_epoch:.2f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    totals = empty_metric_totals()
    loss_fn = nn.CrossEntropyLoss()
    for inputs, targets, legal_mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        legal_mask = legal_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(masked_logits(logits, legal_mask), targets)
        loss.backward()
        optimizer.step()
        update_metrics(totals, loss=loss, logits=logits, targets=targets, legal_mask=legal_mask)
    return finalize_metrics(totals)


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

