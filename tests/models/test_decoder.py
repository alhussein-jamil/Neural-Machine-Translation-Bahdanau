import unittest

import torch

from models.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_entry = torch.rand(3, 5, 10)  # batch_size, Tx, hidden_size
        self.encoder_out_size = 10
        self.decoder_out_size = 12
        self.seqlen = 5
        config_alignment = dict(
            input_size=self.encoder_out_size + self.decoder_out_size,
            hidden_sizes=[10, 10],
            output_size=self.seqlen,
            device="cpu",
            activation=torch.nn.ReLU(),
            last_layer_activation=torch.nn.Sigmoid(),
            dropout=0.2,
        )
        config_birnn = dict(
            input_size=self.encoder_out_size,
            hidden_size=12,
            num_layers=1,
            device="cpu",
            dropout=0,
            type="GRU",
            bidirectional=False,
        )
        self.config = dict(alignment=config_alignment, birnn=config_birnn)
        return super().setUp()

    def test_initalization(self):
        decoder = Decoder(self.config)
        self.assertIsInstance(decoder, torch.nn.Module)

    def test_forward(self):
        decoder = Decoder(self.config)
        output = decoder(self.sample_entry)
        self.assertEqual(output.shape, torch.Size([3, 5, 12]))


if __name__ == "__main__":
    unittest.main()
