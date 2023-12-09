import unittest
import torch
from models.encoder import Encoder  

class TestEncoder(unittest.TestCase):

    def test_forward(self):
        # Define batch_size and sequence_length
        batch_size = 32
        sequence_length = 5

        # Create an instance of the Encoder class
        encoder = Encoder(
            rnn_hidden_size=10,
            rnn_num_layers=2,
            rnn_device="cpu",
            vocab_size=20,
            rnn_type="GRU"
        )

        # Generate dummy data for the test
        input_data = torch.randint(0, 20, (batch_size, sequence_length))

        # Call the forward function of the encoder
        rnn_output, rnn_hidden = encoder(input_data)

        # Check that the output and hidden state have the expected dimensions
        self.assertEqual(rnn_output.shape, (batch_size, sequence_length, 20))
        self.assertEqual(rnn_hidden.shape, (2 * encoder.rnn.num_layers, batch_size, 10))  # (2 * num_layers, batch_size, hidden_size)

if __name__ == '__main__':
    unittest.main()

