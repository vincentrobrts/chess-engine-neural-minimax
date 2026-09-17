import unittest
from pathlib import Path
from unittest.mock import patch

import chess

import neural_eval
from ai import choose_move_with_stats
from constants import (
    AI_MODE_MINIMAX,
    AI_MODE_MINIMAX_NN,
    AI_MODE_MINIMAX_POLICY_ORDERED,
    AI_MODES,
)
from game import ChessGame
from neural_eval import NeuralMoveScorer, get_neural_scorer
from opening_book import OpeningBook
from ui import get_ai_mode_button_rects, mode_at_position

DEFAULT_POLICY_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "lichess_policy_v1_best.pt"


class FakeNeuralScorer:
    available = True
    load_error = None

    def __init__(self):
        self.best_move_calls = 0
        self.confidence_calls = 0
        self.order_calls = 0
        self.forward_pass_count = 0

    def clear_cache(self):
        pass

    def reset_forward_pass_count(self):
        self.forward_pass_count = 0

    def order_moves(self, board, legal_moves):
        self.order_calls += 1
        self.forward_pass_count += 1
        return legal_moves

    def best_legal_move(self, board):
        self.best_move_calls += 1
        return next(iter(board.legal_moves), None)

    def confidence_score(self, board):
        self.confidence_calls += 1
        return 0.0


class ReverseOrderScorer(FakeNeuralScorer):
    def order_moves(self, board, legal_moves):
        self.order_calls += 1
        self.forward_pass_count += 1
        return list(reversed(legal_moves))


class EngineBehaviorTests(unittest.TestCase):
    def test_all_configured_ai_modes_can_be_selected(self):
        game = ChessGame(ai_mode=next(iter(AI_MODES)))

        for mode in AI_MODES:
            game.set_ai_mode(mode)
            self.assertEqual(game.ai_mode, mode)

    def test_all_ai_modes_return_legal_moves(self):
        for mode in AI_MODES:
            board = chess.Board()
            result = choose_move_with_stats(board, mode=mode, depth=1, time_limit_seconds=1.0)

            self.assertIsNotNone(result.move, mode)
            self.assertIn(result.move, board.legal_moves)

    def test_neural_mode_reaches_inference_path_when_scorer_is_available(self):
        board = chess.Board()
        scorer = FakeNeuralScorer()

        with patch("ai.get_neural_scorer", return_value=scorer):
            result = choose_move_with_stats(
                board,
                mode=AI_MODE_MINIMAX_NN,
                depth=1,
                time_limit_seconds=1.0,
            )

        self.assertIsNotNone(result.move)
        self.assertGreater(scorer.order_calls, 0)
        self.assertGreater(scorer.best_move_calls, 0)
        self.assertGreater(scorer.confidence_calls, 0)

    def test_policy_ordered_mode_uses_neural_ordering_but_classic_leaf_eval(self):
        board = chess.Board()
        scorer = FakeNeuralScorer()

        with (
            patch("ai.get_neural_scorer", return_value=scorer),
            patch("ai.evaluate_neural_proxy", side_effect=AssertionError("neural proxy should not be used")),
        ):
            result = choose_move_with_stats(
                board,
                mode=AI_MODE_MINIMAX_POLICY_ORDERED,
                depth=1,
                time_limit_seconds=1.0,
            )

        self.assertIsNotNone(result.move)
        self.assertGreater(scorer.order_calls, 0)
        self.assertEqual(scorer.best_move_calls, 0)
        self.assertEqual(scorer.confidence_calls, 0)
        self.assertGreater(result.stats.neural_forward_passes, 0)

    def test_policy_ordering_preserves_fixed_depth_minimax_score(self):
        positions = [
            chess.STARTING_FEN,
            "r2q1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
            "8/5pk1/6p1/8/4P3/6P1/5PK1/8 w - - 0 1",
        ]

        for fen in positions:
            board = chess.Board(fen)
            baseline = choose_move_with_stats(
                board.copy(),
                mode=AI_MODE_MINIMAX,
                depth=2,
                time_limit_seconds=None,
            )
            scorer = ReverseOrderScorer()

            with patch("ai.get_neural_scorer", return_value=scorer):
                ordered = choose_move_with_stats(
                    board.copy(),
                    mode=AI_MODE_MINIMAX_POLICY_ORDERED,
                    depth=2,
                    time_limit_seconds=None,
                )

            self.assertEqual(baseline.score, ordered.score, fen)
            self.assertEqual(baseline.stats.depth_completed, 2)
            self.assertEqual(ordered.stats.depth_completed, 2)
            self.assertGreater(scorer.order_calls, 0)

    def test_neural_ordering_ignores_unknown_legal_moves_without_dropping_them(self):
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        known_move = legal_moves[-1]
        scorer = object.__new__(NeuralMoveScorer)
        scorer.available = True
        scorer.score_legal_moves = lambda _board: {known_move: 1.0}

        ordered_moves = NeuralMoveScorer.order_moves(scorer, board, legal_moves)

        self.assertEqual(ordered_moves[0], known_move)
        self.assertCountEqual(ordered_moves, legal_moves)

    @unittest.skipIf(neural_eval.torch is None, "PyTorch is not installed")
    @unittest.skipUnless(DEFAULT_POLICY_CHECKPOINT.exists(), "Default policy checkpoint is not present")
    def test_real_neural_scorer_loads_default_checkpoint(self):
        neural_eval._DEFAULT_SCORER = None
        scorer = get_neural_scorer()

        self.assertTrue(scorer.available, scorer.load_error)
        self.assertFalse(scorer.model.training)
        scorer.clear_cache()
        scorer.reset_forward_pass_count()
        self.assertTrue(scorer.score_legal_moves(chess.Board()))
        self.assertGreater(scorer.forward_pass_count, 0)

    def test_opening_book_returns_legal_starting_move(self):
        book = OpeningBook()
        board = chess.Board()
        line = book.choose_opening([], chess.WHITE)

        self.assertIsNotNone(line)
        move = book.san_to_move(board, line.next_san([], chess.WHITE))
        self.assertIn(move, board.legal_moves)

    def test_mode_selector_hit_testing_uses_configured_modes(self):
        first_mode = next(iter(AI_MODES))
        rect = get_ai_mode_button_rects()[first_mode]

        self.assertEqual(mode_at_position(rect.centerx, rect.centery), first_mode)


if __name__ == "__main__":
    unittest.main()
