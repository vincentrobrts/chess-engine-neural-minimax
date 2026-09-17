import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from policy.encoding import (
    BLACK_KINGSIDE_CASTLE_PLANE,
    EN_PASSANT_PLANE,
    INPUT_PLANES,
    SIDE_TO_MOVE_PLANE,
    WHITE_KINGSIDE_CASTLE_PLANE,
    encode_board,
    position_key,
)
from policy.vocab import get_move_vocabulary, legal_move_ids

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None

if torch is not None:
    from policy.checkpoint import load_policy_checkpoint, save_policy_checkpoint
    from policy.model import build_policy_model
    from training.common import masked_logits


class PolicyComponentTests(unittest.TestCase):
    def test_encoder_shape_side_to_move_castling_and_en_passant(self):
        board = chess.Board()
        encoded = encode_board(board)

        self.assertEqual(encoded.shape, (INPUT_PLANES, 8, 8))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(np.all(encoded[SIDE_TO_MOVE_PLANE] == 1.0))
        self.assertTrue(np.all(encoded[WHITE_KINGSIDE_CASTLE_PLANE] == 1.0))
        self.assertTrue(np.all(encoded[BLACK_KINGSIDE_CASTLE_PLANE] == 1.0))

        board.push_san("e4")
        encoded_after_e4 = encode_board(board)
        ep_row, ep_col = 5, 4
        self.assertEqual(encoded_after_e4[EN_PASSANT_PLANE, ep_row, ep_col], 1.0)
        self.assertTrue(np.all(encoded_after_e4[SIDE_TO_MOVE_PLANE] == 0.0))

    def test_position_key_ignores_clocks_but_keeps_chess_state(self):
        board_a = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
        board_b = chess.Board("8/8/8/8/8/8/8/K6k w - - 12 37")
        board_c = chess.Board("8/8/8/8/8/8/8/K6k b - - 0 1")

        self.assertEqual(position_key(board_a), position_key(board_b))
        self.assertNotEqual(position_key(board_a), position_key(board_c))

    def test_vocabulary_round_trips_special_moves(self):
        vocabulary = get_move_vocabulary()
        for uci in ("e2e4", "e1g1", "e1c1", "e5d6", "a7a8q", "a7a8r", "a7a8b", "a7a8n"):
            self.assertTrue(vocabulary.contains(uci), uci)
            self.assertEqual(vocabulary.decode(vocabulary.encode(uci)).uci(), uci)

    def test_legal_move_ids_cover_current_legal_moves(self):
        board = chess.Board()
        vocabulary = get_move_vocabulary()
        encoded_legal_moves = {vocabulary.decode(move_id) for move_id in legal_move_ids(board, vocabulary)}

        self.assertEqual(encoded_legal_moves, set(board.legal_moves))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_legal_mask_blocks_illegal_logits(self):
        board = chess.Board()
        vocabulary = get_move_vocabulary()
        legal_ids = legal_move_ids(board, vocabulary)
        logits = torch.zeros((1, len(vocabulary.moves)), dtype=torch.float32)
        logits[0, vocabulary.encode("a1a8")] = 100.0
        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
        legal_mask[0, legal_ids] = True

        predicted_id = int(torch.argmax(masked_logits(logits, legal_mask), dim=1)[0])

        self.assertIn(vocabulary.decode(predicted_id), board.legal_moves)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_model_output_dimensions_and_checkpoint_compatibility(self):
        vocabulary = get_move_vocabulary()
        model = build_policy_model(vocab_size=len(vocabulary.moves), channels=16, residual_blocks=1)
        logits = model(torch.zeros((2, INPUT_PLANES, 8, 8), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, len(vocabulary.moves)))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            save_policy_checkpoint(
                path,
                model=model,
                epoch=1,
                validation_metrics={"loss": 1.0},
                training_config={"test": True},
                dataset_manifest={"source_url": "fixture", "source_license": "synthetic fixture"},
                seed=42,
            )
            loaded_model, checkpoint = load_policy_checkpoint(path)

        self.assertFalse(loaded_model.training)
        self.assertEqual(checkpoint["move_vocabulary_size"], len(vocabulary.moves))
        self.assertEqual(checkpoint["move_vocabulary_checksum"], vocabulary.checksum)


if __name__ == "__main__":
    unittest.main()

