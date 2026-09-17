"""Prepare a bounded move-policy dataset from Lichess Stockfish eval JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator
from urllib.request import urlopen

import chess

from policy.encoding import normalized_position_fen, position_key
from policy.vocab import get_move_vocabulary, legal_move_ids

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SOURCE_LICENSE = "CC0"
PREPROCESSING_VERSION = "policy-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare policy training data.")
    parser.add_argument("--data", default=LICHESS_EVAL_URL, help="Input .jsonl, .jsonl.zst, URL, or 'fixture'.")
    parser.add_argument("--output", default="data/policy_dataset_smoke.jsonl")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-positions", type=int, default=10000)
    parser.add_argument("--max-records", type=int, default=500000)
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    stats, selected = prepare_dataset(
        data=args.data,
        max_positions=args.max_positions,
        max_records=args.max_records,
        min_depth=args.min_depth,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started_at
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = sorted(selected.values(), key=lambda record: record["position_key"])
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = build_manifest(args, stats, records, elapsed)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    print_summary(stats, records, elapsed, output_path, manifest_path)


def prepare_dataset(
    *,
    data: str,
    max_positions: int,
    max_records: int,
    min_depth: int,
    seed: int,
) -> tuple[Counter[str], dict[str, dict[str, Any]]]:
    rng = random.Random(seed)
    vocabulary = get_move_vocabulary()
    stats: Counter[str] = Counter()
    selected: dict[str, dict[str, Any]] = {}
    selected_keys: list[str] = []
    seen_depth_by_key: dict[str, int] = {}
    valid_unique_seen = 0

    for raw_record in iter_records(data):
        if stats["raw_records_scanned"] >= max_records:
            break
        stats["raw_records_scanned"] += 1

        parsed = parse_candidate(raw_record, min_depth, vocabulary)
        if parsed is None:
            reason = raw_record.get("_reject_reason", "malformed_record")
            stats[reason] += 1
            continue

        key = parsed["position_key"]
        previous_depth = seen_depth_by_key.get(key)
        if previous_depth is not None:
            stats["duplicates"] += 1
            if parsed["stockfish_depth"] > previous_depth and key in selected:
                selected[key] = parsed
                seen_depth_by_key[key] = parsed["stockfish_depth"]
            continue

        seen_depth_by_key[key] = parsed["stockfish_depth"]
        valid_unique_seen += 1
        stats["valid_records"] += 1

        if len(selected_keys) < max_positions:
            selected_keys.append(key)
            selected[key] = parsed
            continue

        replacement_index = rng.randrange(valid_unique_seen)
        if replacement_index < max_positions:
            replaced_key = selected_keys[replacement_index]
            selected.pop(replaced_key, None)
            selected_keys[replacement_index] = key
            selected[key] = parsed

    return stats, selected


def parse_candidate(
    record: dict[str, Any],
    min_depth: int,
    vocabulary,
) -> dict[str, Any] | None:
    fen = record.get("fen")
    if not isinstance(fen, str):
        record["_reject_reason"] = "malformed_records"
        return None

    try:
        board = chess.Board(fen)
    except ValueError:
        record["_reject_reason"] = "malformed_records"
        return None

    if board.is_game_over() or board.is_checkmate() or board.is_stalemate():
        record["_reject_reason"] = "terminal_positions"
        return None

    legal_moves = list(board.legal_moves)
    if len(legal_moves) < 2:
        record["_reject_reason"] = "forced_move_rejects"
        return None

    selected_eval = select_best_eval(record.get("evals", []), min_depth)
    if selected_eval is None:
        record["_reject_reason"] = "low_depth_rejects"
        return None

    depth, pv = selected_eval
    target_uci = first_pv_move(pv)
    if target_uci is None:
        record["_reject_reason"] = "empty_pv_rejects"
        return None

    try:
        target_move = board.parse_uci(target_uci)
    except ValueError:
        record["_reject_reason"] = "illegal_targets"
        return None

    if target_move not in board.legal_moves:
        record["_reject_reason"] = "illegal_targets"
        return None
    if not vocabulary.contains(target_move):
        record["_reject_reason"] = "vocabulary_misses"
        return None

    legal_ids = legal_move_ids(board, vocabulary)
    if not legal_ids:
        record["_reject_reason"] = "vocabulary_misses"
        return None

    key = position_key(board)
    return {
        "fen": normalized_position_fen(board),
        "target_move": target_move.uci(),
        "target_id": vocabulary.encode(target_move),
        "legal_ids": legal_ids,
        "stockfish_depth": depth,
        "split": split_for_key(key),
        "position_key": key,
    }


def select_best_eval(evals: Any, min_depth: int) -> tuple[int, Any] | None:
    if not isinstance(evals, list):
        return None

    candidates = []
    for evaluation in evals:
        if not isinstance(evaluation, dict):
            continue
        depth = evaluation.get("depth")
        if not isinstance(depth, int) or depth < min_depth:
            continue
        pvs = evaluation.get("pvs")
        if isinstance(pvs, list) and pvs:
            candidates.append((depth, pvs[0]))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def first_pv_move(pv: Any) -> str | None:
    if isinstance(pv, dict):
        line = pv.get("line") or pv.get("pv")
    else:
        line = pv
    if not isinstance(line, str) or not line.strip():
        return None
    return line.split()[0]


def split_for_key(key: str) -> str:
    value = int(key[:16], 16) / float(16**16)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def iter_records(data: str) -> Iterator[dict[str, Any]]:
    if data == "fixture":
        yield from fixture_records()
        return

    if data.startswith("http://") or data.startswith("https://"):
        response = urlopen(data, timeout=30)
        source = response
    else:
        source = Path(data).open("rb")

    try:
        if data.endswith(".zst"):
            try:
                import zstandard as zstd
            except ImportError as exc:
                raise SystemExit("Install zstandard or use an uncompressed JSONL file.") from exc
            reader = zstd.ZstdDecompressor().stream_reader(source)
            text_stream = _iter_text_lines(reader)
        else:
            text_stream = _iter_text_lines(source)

        for line in text_stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_reject_reason": "malformed_records"}
    finally:
        source.close()


def _iter_text_lines(binary_stream) -> Iterator[str]:
    buffer = b""
    while True:
        chunk = binary_stream.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line:
                yield line.decode("utf-8")
    if buffer:
        yield buffer.decode("utf-8")


def fixture_records() -> Iterator[dict[str, Any]]:
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
        "r2q1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
        "8/5pk1/6p1/8/4P3/6P1/5PK1/8 w - - 0 1",
    ]
    for index in range(2000):
        board = chess.Board(fens[index % len(fens)])
        for _ in range(index % 8):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[(index + board.ply()) % len(moves)])
            if board.is_game_over():
                break
        legal_moves = list(board.legal_moves)
        if len(legal_moves) < 2:
            continue
        target = choose_fixture_target(board, legal_moves)
        yield {
            "fen": board.fen(),
            "evals": [
                {
                    "depth": 18 + (index % 8),
                    "pvs": [{"line": target.uci()}],
                }
            ],
        }


def choose_fixture_target(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
    captures = [move for move in legal_moves if board.is_capture(move)]
    if captures:
        return captures[0]
    checks = []
    for move in legal_moves:
        board.push(move)
        if board.is_check():
            checks.append(move)
        board.pop()
    return checks[0] if checks else legal_moves[0]


def build_manifest(
    args: argparse.Namespace,
    stats: Counter[str],
    records: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    vocabulary = get_move_vocabulary()
    return {
        "source_url": LICHESS_EVAL_URL if args.data != "fixture" else "fixture",
        "source_license": SOURCE_LICENSE if args.data != "fixture" else "synthetic fixture",
        "retrieval_date": date.today().isoformat(),
        "raw_records_scanned": stats["raw_records_scanned"],
        "valid_records": stats["valid_records"],
        "malformed_records": stats["malformed_records"],
        "low_depth_rejects": stats["low_depth_rejects"],
        "forced_move_rejects": stats["forced_move_rejects"],
        "illegal_targets": stats["illegal_targets"],
        "duplicates": stats["duplicates"],
        "vocabulary_misses": stats["vocabulary_misses"],
        "terminal_positions": stats["terminal_positions"],
        "empty_pv_rejects": stats["empty_pv_rejects"],
        "final_train_count": split_counts["train"],
        "final_validation_count": split_counts["val"],
        "final_test_count": split_counts["test"],
        "max_positions": args.max_positions,
        "max_records": args.max_records,
        "min_depth": args.min_depth,
        "split_method": "SHA-256 over placement, turn, castling rights, and en-passant square",
        "seed": args.seed,
        "vocabulary_size": len(vocabulary.moves),
        "vocabulary_checksum": vocabulary.checksum,
        "preprocessing_version": PREPROCESSING_VERSION,
        "elapsed_seconds": elapsed,
    }


def print_summary(
    stats: Counter[str],
    records: list[dict[str, Any]],
    elapsed: float,
    output_path: Path,
    manifest_path: Path,
) -> None:
    valid_per_second = len(records) / elapsed if elapsed > 0 else 0.0
    scanned_per_second = stats["raw_records_scanned"] / elapsed if elapsed > 0 else 0.0
    split_counts = Counter(record["split"] for record in records)
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Preprocessing records/sec: {scanned_per_second:.1f}")
    print(f"Valid positions/sec: {valid_per_second:.1f}")
    print(f"Raw records scanned: {stats['raw_records_scanned']}")
    print(f"Valid unique records: {stats['valid_records']}")
    print(f"Final train/val/test: {split_counts['train']}/{split_counts['val']}/{split_counts['test']}")
    for key in sorted(stats):
        if key not in {"raw_records_scanned", "valid_records"}:
            print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
