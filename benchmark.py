"""Run repeated fixed-position benchmarks for the chess AI search modes."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
import platform
import random
import statistics
import sys

try:
    import chess

    import neural_eval
    from ai import SearchResult, choose_move_with_stats
    from constants import (
        AI_MODE_MINIMAX,
        AI_MODE_MINIMAX_POLICY_ORDERED,
        AI_MODES,
    )
    from neural_eval import get_neural_scorer
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        f"Missing dependency: {exc}. Install project dependencies with "
        "`pip install -r requirements.txt` before running benchmarks."
    ) from exc


BENCHMARK_POSITIONS = [
    (
        "Starting position",
        chess.STARTING_FEN,
    ),
    (
        "Early Italian",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
    ),
    (
        "Tactical middlegame",
        "r2q1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    ),
    (
        "Quiet middlegame",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    ),
    (
        "Simple endgame",
        "8/5pk1/6p1/8/4P3/6P1/5PK1/8 w - - 0 1",
    ),
]

NEURAL_MODES = {mode for mode, config in AI_MODES.items() if config.uses_neural}
SEARCH_MODES = [
    mode
    for mode, config in AI_MODES.items()
    if config.evaluation_mode is not None
]


@dataclass(frozen=True)
class BenchmarkRecord:
    position: str
    mode: str
    run: int
    requested_depth: int
    selected_move: str
    score: float | None
    elapsed_ms: float
    nodes: int
    leaf_nodes: int
    cutoffs: int
    nodes_per_second: float
    depth_completed: int
    neural_forward_passes: int
    timed_out: bool


@dataclass(frozen=True)
class SummaryRecord:
    position: str
    mode: str
    runs: int
    runtimes_ms: list[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    median_nodes: float
    median_leaf_nodes: float
    median_cutoffs: float
    median_neural_forward_passes: float
    median_depth_completed: float
    moves: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chess AI search modes.")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Requested minimax depth for each position. Default: 2.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=SEARCH_MODES,
        help="Mode to benchmark. Repeat to run multiple modes. Default: all search modes.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Measured runs per position after warm-up. Default: 5.",
    )
    parser.add_argument(
        "--time-ms",
        type=float,
        default=None,
        help="Optional per-move search time limit in milliseconds.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the unmeasured warm-up run.",
    )
    parser.add_argument(
        "--positions-file",
        default=None,
        help="Optional JSONL dataset file; uses held-out test positions from it.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=50,
        help="Maximum positions to read from --positions-file. Default: 50.",
    )
    return parser.parse_args()


def neural_status() -> tuple[bool, str | None]:
    """Return whether neural search modes should be benchmarked."""
    try:
        scorer = get_neural_scorer()
    except Exception as exc:  # pragma: no cover - defensive benchmark UX
        return False, f"Could not initialize neural scorer: {exc}"

    if scorer.available:
        return True, None
    return False, scorer.load_error or "Neural scorer is unavailable."


def selected_modes(requested_modes: list[str] | None) -> list[str]:
    if requested_modes:
        return requested_modes
    return SEARCH_MODES[:]


def runnable_modes(modes: Iterable[str]) -> list[str]:
    modes = list(dict.fromkeys(modes))
    needs_neural = any(mode in NEURAL_MODES for mode in modes)
    if not needs_neural:
        return modes

    neural_available, reason = neural_status()
    if neural_available:
        return modes

    print(f"Warning: skipping neural modes because {reason}")
    return [mode for mode in modes if mode not in NEURAL_MODES]


def reset_neural_measurement(mode: str) -> None:
    if mode not in NEURAL_MODES:
        return

    scorer = get_neural_scorer()
    scorer.clear_cache()
    scorer.reset_forward_pass_count()


def run_search(
    *,
    position_name: str,
    fen: str,
    mode: str,
    depth: int,
    time_limit_seconds: float | None,
    run_number: int,
) -> BenchmarkRecord:
    reset_neural_measurement(mode)
    board = chess.Board(fen)
    random.seed(0)
    result = choose_move_with_stats(
        board=board,
        mode=mode,
        depth=depth,
        time_limit_seconds=time_limit_seconds,
    )
    return build_record(position_name, run_number, result)


def build_record(position_name: str, run_number: int, result: SearchResult) -> BenchmarkRecord:
    stats = result.stats
    return BenchmarkRecord(
        position=position_name,
        mode=stats.mode,
        run=run_number,
        requested_depth=stats.requested_depth,
        selected_move=stats.selected_move or "-",
        score=stats.score,
        elapsed_ms=stats.elapsed_ms,
        nodes=stats.nodes,
        leaf_nodes=stats.leaf_nodes,
        cutoffs=stats.cutoffs,
        nodes_per_second=stats.nodes_per_second,
        depth_completed=stats.depth_completed,
        neural_forward_passes=stats.neural_forward_passes,
        timed_out=stats.timed_out,
    )


def run_benchmark(
    *,
    depth: int,
    modes: list[str],
    runs: int,
    time_ms: float | None,
    warmup: bool,
    positions: list[tuple[str, str]],
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    time_limit_seconds = None if time_ms is None else max(0.0, time_ms / 1000.0)

    for position_name, fen in positions:
        for mode in modes:
            try:
                if warmup:
                    run_search(
                        position_name=position_name,
                        fen=fen,
                        mode=mode,
                        depth=depth,
                        time_limit_seconds=time_limit_seconds,
                        run_number=0,
                    )

                for run_number in range(1, runs + 1):
                    records.append(
                        run_search(
                            position_name=position_name,
                            fen=fen,
                            mode=mode,
                            depth=depth,
                            time_limit_seconds=time_limit_seconds,
                            run_number=run_number,
                        )
                    )
            except Exception as exc:  # pragma: no cover - benchmark should continue
                print(f"Warning: {position_name} / {mode} failed: {exc}")

    return records


def summarize(records: list[BenchmarkRecord]) -> dict[tuple[str, str], SummaryRecord]:
    summaries: dict[tuple[str, str], SummaryRecord] = {}
    grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.position, record.mode), []).append(record)

    for key, group in grouped.items():
        runtimes = [record.elapsed_ms for record in group]
        summaries[key] = SummaryRecord(
            position=key[0],
            mode=key[1],
            runs=len(group),
            runtimes_ms=runtimes,
            mean_ms=statistics.mean(runtimes),
            median_ms=statistics.median(runtimes),
            std_ms=statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
            median_nodes=statistics.median(record.nodes for record in group),
            median_leaf_nodes=statistics.median(record.leaf_nodes for record in group),
            median_cutoffs=statistics.median(record.cutoffs for record in group),
            median_neural_forward_passes=statistics.median(
                record.neural_forward_passes for record in group
            ),
            median_depth_completed=statistics.median(record.depth_completed for record in group),
            moves=", ".join(dict.fromkeys(record.selected_move for record in group)),
        )
    return summaries


def print_environment() -> None:
    torch_module = neural_eval.torch
    if torch_module is None:
        torch_version = "not installed"
        inference_device = "unavailable"
        cuda_available = "no"
    else:
        torch_version = torch_module.__version__
        cuda_enabled = torch_module.cuda.is_available()
        inference_device = "cuda" if cuda_enabled else "cpu"
        cuda_available = "yes" if cuda_enabled else "no"

    print("Benchmark environment")
    print(f"  CPU: {cpu_name()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  PyTorch: {torch_version}")
    print(f"  Inference device: {inference_device}")
    print(f"  CUDA available: {cuda_available}")
    print()


def cpu_name() -> str:
    return (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
        or "Unknown CPU"
    )


def print_run_table(records: list[BenchmarkRecord]) -> None:
    print("Measured runs")
    if not records:
        print("No benchmark rows ran.")
        return

    headers = [
        "position",
        "mode",
        "run",
        "depth",
        "move",
        "score",
        "elapsed_ms",
        "nodes",
        "leaf_nodes",
        "cutoffs",
        "nodes/sec",
        "neural_calls",
        "depth_done",
        "timed_out",
    ]
    rows = [
        {
            "position": record.position,
            "mode": record.mode,
            "run": str(record.run),
            "depth": str(record.requested_depth),
            "move": record.selected_move,
            "score": format_number(record.score),
            "elapsed_ms": f"{record.elapsed_ms:.1f}",
            "nodes": str(record.nodes),
            "leaf_nodes": str(record.leaf_nodes),
            "cutoffs": str(record.cutoffs),
            "nodes/sec": f"{record.nodes_per_second:.0f}",
            "neural_calls": str(record.neural_forward_passes),
            "depth_done": str(record.depth_completed),
            "timed_out": "yes" if record.timed_out else "no",
        }
        for record in records
    ]
    print_table(headers, rows)
    print()


def print_summary_table(summaries: dict[tuple[str, str], SummaryRecord]) -> None:
    print("Summary by position and mode")
    if not summaries:
        print("No summary rows.")
        return

    headers = [
        "position",
        "mode",
        "runs",
        "runtimes_ms",
        "mean_ms",
        "median_ms",
        "std_ms",
        "median_nodes",
        "median_leafs",
        "median_cutoffs",
        "median_neural_calls",
        "median_depth",
        "moves",
    ]
    rows = []
    for summary in summaries.values():
        rows.append(
            {
                "position": summary.position,
                "mode": summary.mode,
                "runs": str(summary.runs),
                "runtimes_ms": ", ".join(f"{runtime:.1f}" for runtime in summary.runtimes_ms),
                "mean_ms": f"{summary.mean_ms:.1f}",
                "median_ms": f"{summary.median_ms:.1f}",
                "std_ms": f"{summary.std_ms:.1f}",
                "median_nodes": format_number(summary.median_nodes),
                "median_leafs": format_number(summary.median_leaf_nodes),
                "median_cutoffs": format_number(summary.median_cutoffs),
                "median_neural_calls": format_number(summary.median_neural_forward_passes),
                "median_depth": format_number(summary.median_depth_completed),
                "moves": summary.moves,
            }
        )
    print_table(headers, rows)
    print()


def print_policy_comparison(summaries: dict[tuple[str, str], SummaryRecord]) -> None:
    comparison_rows = []
    node_reductions = []
    runtime_changes = []

    for position_name in sorted({position for position, _mode in summaries}):
        baseline = summaries.get((position_name, AI_MODE_MINIMAX))
        ordered = summaries.get((position_name, AI_MODE_MINIMAX_POLICY_ORDERED))
        if baseline is None or ordered is None or baseline.median_nodes == 0:
            continue

        node_reduction = (
            (baseline.median_nodes - ordered.median_nodes) / baseline.median_nodes * 100.0
        )
        runtime_change = (
            (ordered.median_ms - baseline.median_ms) / baseline.median_ms * 100.0
            if baseline.median_ms > 0
            else 0.0
        )
        node_reductions.append(node_reduction)
        runtime_changes.append(runtime_change)
        comparison_rows.append(
            {
                "position": position_name,
                "minimax_nodes": format_number(baseline.median_nodes),
                "ordered_nodes": format_number(ordered.median_nodes),
                "node_reduction_%": f"{node_reduction:.1f}",
                "minimax_ms": f"{baseline.median_ms:.1f}",
                "ordered_ms": f"{ordered.median_ms:.1f}",
                "runtime_change_%": f"{runtime_change:.1f}",
                "ordered_neural_calls": format_number(ordered.median_neural_forward_passes),
            }
        )

    if not comparison_rows:
        return

    print("Minimax vs Neural-Ordered Minimax")
    print_table(
        [
            "position",
            "minimax_nodes",
            "ordered_nodes",
            "node_reduction_%",
            "minimax_ms",
            "ordered_ms",
            "runtime_change_%",
            "ordered_neural_calls",
        ],
        comparison_rows,
    )
    print()
    print(
        "Aggregate node reduction: "
        f"mean {statistics.mean(node_reductions):.1f}%, "
        f"median {statistics.median(node_reductions):.1f}%, "
        f"min {min(node_reductions):.1f}%, "
        f"max {max(node_reductions):.1f}%"
    )
    print(
        "Aggregate runtime change: "
        f"mean {statistics.mean(runtime_changes):.1f}%, "
        f"median {statistics.median(runtime_changes):.1f}%, "
        f"min {min(runtime_changes):.1f}%, "
        f"max {max(runtime_changes):.1f}%"
    )
    print()


def print_determinism_checks(records: list[BenchmarkRecord], time_ms: float | None) -> None:
    if time_ms is not None:
        return

    grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.position, record.mode), []).append(record)

    warnings = []
    for (position, mode), group in grouped.items():
        if any(record.timed_out for record in group):
            continue
        node_counts = {record.nodes for record in group}
        cutoff_counts = {record.cutoffs for record in group}
        if len(node_counts) > 1 or len(cutoff_counts) > 1:
            warnings.append(
                f"{position} / {mode}: nodes={sorted(node_counts)}, cutoffs={sorted(cutoff_counts)}"
            )

    if warnings:
        print("Determinism warnings")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("Determinism checks: fixed-depth node and cutoff counts were repeatable.")
    print()


def print_result_consistency(records: list[BenchmarkRecord], time_ms: float | None) -> None:
    if time_ms is not None:
        return

    grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.position, record.mode), []).append(record)

    notes = []
    for position_name in sorted({record.position for record in records}):
        baseline = grouped.get((position_name, AI_MODE_MINIMAX))
        ordered = grouped.get((position_name, AI_MODE_MINIMAX_POLICY_ORDERED))
        if not baseline or not ordered:
            continue
        if any(record.timed_out for record in baseline + ordered):
            continue

        baseline_depths = {record.depth_completed for record in baseline}
        ordered_depths = {record.depth_completed for record in ordered}
        baseline_scores = {record.score for record in baseline}
        ordered_scores = {record.score for record in ordered}
        baseline_moves = {record.selected_move for record in baseline}
        ordered_moves = {record.selected_move for record in ordered}

        if baseline_depths == ordered_depths and baseline_scores != ordered_scores:
            notes.append(
                f"Issue: {position_name} reached the same depth but scores differed "
                f"({sorted(baseline_scores)} vs {sorted(ordered_scores)})."
            )
        elif baseline_scores == ordered_scores and baseline_moves != ordered_moves:
            notes.append(
                f"Note: {position_name} returned equal score(s) with different move tie-breaks "
                f"({sorted(baseline_moves)} vs {sorted(ordered_moves)})."
            )

    if notes:
        print("Result consistency notes")
        for note in notes:
            print(f"  {note}")
    else:
        print("Result consistency: no fixed-depth score differences detected.")
    print()


def print_table(headers: list[str], rows: list[dict[str, str]]) -> None:
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def load_positions(path: str | None, max_positions: int) -> list[tuple[str, str]]:
    if path is None:
        return BENCHMARK_POSITIONS

    positions = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                continue
            fen = record.get("fen")
            if not isinstance(fen, str):
                continue
            positions.append((f"heldout-{len(positions) + 1}", fen))
            if len(positions) >= max_positions:
                break
    if not positions:
        raise SystemExit(f"No test-split positions found in {path}.")
    return positions


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    if args.depth < 1:
        raise SystemExit("--depth must be at least 1.")

    modes = runnable_modes(selected_modes(args.mode))
    positions = load_positions(args.positions_file, args.max_positions)
    print_environment()
    print(f"Requested depth: {args.depth}")
    print(f"Measured runs per position: {args.runs}")
    print(f"Warm-up run: {'no' if args.no_warmup else 'yes'}")
    print(f"Time limit: {'none' if args.time_ms is None else f'{args.time_ms:.0f} ms'}")
    print(f"Modes: {', '.join(modes) if modes else 'none'}")
    print(f"Positions: {len(positions)}")
    print()

    records = run_benchmark(
        depth=args.depth,
        modes=modes,
        runs=args.runs,
        time_ms=args.time_ms,
        warmup=not args.no_warmup,
        positions=positions,
    )
    summaries = summarize(records)
    print_run_table(records)
    print_summary_table(summaries)
    print_policy_comparison(summaries)
    print_determinism_checks(records, args.time_ms)
    print_result_consistency(records, args.time_ms)


if __name__ == "__main__":
    main()
