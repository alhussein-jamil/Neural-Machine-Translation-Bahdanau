import unittest

import torch

from models.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 7
        self.seqlen = 5
        self.max_out_units = 13
        self.sample_entry = torch.rand(3, self.seqlen, self.hidden_size * 2)  # batch_size, Tx, hidden_size * 2
        self.embedding_size = 3
        self.vocab_size = 12

        config_alignment = dict(
            input_size=self.hidden_size * 3,
            output_size=self.seqlen,
            device="cpu",
        )

        config_rnn = dict(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            device="cpu",
            dropout=0,
            type="GRU",
            bidirectional=False,
        )

        decoder_embedding_cfg = dict(
            embedding_size=self.embedding_size,
            device="cpu",
        )

        output_nn_cfg = dict(
            embedding_size=self.embedding_size,
            max_out_units=self.max_out_units,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device="cpu",
        )
        self.config = dict(
            alignment=config_alignment,
            rnn=config_rnn,
            embedding=decoder_embedding_cfg,
            output_nn=output_nn_cfg,
        )

        return super().setUp()

    def test_initalization(self):
        decoder = Decoder(**self.config)
        self.assertIsInstance(decoder, torch.nn.Module)

    def test_forward(self):
        decoder = Decoder(**self.config)
        output = decoder(self.sample_entry)
        self.assertEqual(output.shape, torch.Size([3, 5, 12]))


if __name__ == "__main__":
    unittest.main()
