import unittest

import torch

from models.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 7
        self.seqlen = 5
        self.max_out_units = 13
        self.embedding_size = 3
        self.vocab_size = 12

        self.sample_entry_h = torch.rand(
            3, self.seqlen, self.hidden_size * 2
        ) # batch_size, Tx, hidden_size * 2
        self.sample_y = torch.rand(
            3, self.vocab_size
        ) # batch_size, Tx, hidden_size

        config_alignment = dict(
            input_size=self.hidden_size * 3,
            device="cpu",
        )

        config_rnn = dict(
            input_size=self.hidden_size * 2 + self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            device="cpu",
            dropout=0,
            type="GRU",
            bidirectional=False,
        )

        decoder_embedding_cfg = dict(
            embedding_size=self.embedding_size,
            vocab_size=self.vocab_size,
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
        decoder = Decoder(**self.config, traditional=False)
        self.assertIsInstance(decoder, torch.nn.Module)

        decoder = Decoder(**self.config, traditional=True)
        self.assertIsInstance(decoder, torch.nn.Module)
    @torch.autocast("cpu")
    def test_forward(self):
        decoder = Decoder(**self.config, traditional=False)
        h_emb = decoder.alignment.nn_h(self.sample_entry_h)
        output, _, _= decoder(0,self.sample_entry_h, h_emb=h_emb, s_i = None, y_i = self.sample_y)
        self.assertEqual(output.squeeze().shape, torch.Size([3,  12]))

        decoder = Decoder(**self.config, traditional=True)
        output, _,_ = decoder(0,self.sample_entry_h, h_emb=None, s_i = None, y_i = self.sample_y)
        self.assertEqual(output.squeeze().shape, torch.Size([3,  12]))


if __name__ == "__main__":
    unittest.main()
