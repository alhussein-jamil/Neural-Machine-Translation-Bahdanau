import argparse

import torch

from models.translation_models import AlignAndTranslate
from src.data_preprocessing import load_data

# Define command-line arguments
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # Parse command-line arguments
    parser.add_argument(
        "--train_len",
        type=int,
        default=100000,
        help="Number of training examples",
    )
    parser.add_argument(
        "--val_len",
        type=int,
        default=None,
        help="Number of validation examples",
    )
    parser.add_argument("--Tx", type=int, default=7, help="Length of the input sequence")
    parser.add_argument("--Ty", type=int, default=7, help="Length of the output sequence")
    parser.add_argument(
        "--hidden_size",
        "-n",
        type=int,
        default=1000,
        help="Size of the hidden layers",
    )
    parser.add_argument(
        "--embedding_size",
        "-m",
        type=int,
        default=620,
        help="Size of the embedding",
    )
    parser.add_argument(
        "--max_out_units",
        "-l",
        type=int,
        default=500,
        help="Size of the hidden layers",
    )
    parser.add_argument(
        "--vocab_size_en",
        type=int,
        default=30000,
        help="Size of the English vocabulary",
    )
    parser.add_argument(
        "--vocab_size_fr",
        type=int,
        default=30000,
        help="Size of the French vocabulary",
    )

    args = parser.parse_args()

    # Load data
    (
        (train_data, train_dataloader),
        (val_data, val_dataloader),
        (english_vocab, french_vocab),
    ) = load_data(
        train_len=args.train_len,
        val_len=args.val_len,
        kx=args.vocab_size_en,
        ky=args.vocab_size_fr,
        Tx=args.Tx,
        Ty=args.Ty,
        batch_size=80,
    )

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Define configuration for the decoder
    config_rnn_decoder = dict(
        input_size=args.hidden_size * 2,
        hidden_size=args.hidden_size,
        num_layers=1,
        device=device,
        dropout=0,
        type="GRU",
        bidirectional=False,
    )

    alignment_cfg = dict(
        input_size=args.hidden_size * 3,
        hidden_sizes=[],
        output_size=args.Ty,
        device=device,
        activation=torch.nn.ReLU(),
        last_layer_activation=torch.nn.Tanh(),
        dropout=0,
    )

    maxout_cfg = dict(
        input_size=args.hidden_size,
        output_size=args.max_out_units,
        num_units=len(french_vocab) + 1,
        device=device,
    )

    output_nn_cfg = dict(
        embedding_size=args.embedding_size,
        max_out_units=args.max_out_units,
        hidden_size=args.hidden_size,
        vocab_size=len(french_vocab) + 1,
        device=device,
    )

    decoder_embedding_cfg = dict(
        embedding_size=args.embedding_size,
        device=device,
    )

    decoder_fcnn_cfg = dict(
        input_size=args.hidden_size,
        hidden_sizes=[args.max_out_units],
        output_size=len(french_vocab) + 1,
        device=device,
        activation=torch.nn.Tanh(),
        last_layer_activation=torch.nn.Identity(),
        dropout=0,
    )

    config_decoder = dict(
        alignment=alignment_cfg,
        rnn=config_rnn_decoder,
        maxout=maxout_cfg,
        fcnn=decoder_fcnn_cfg,
        output_nn=output_nn_cfg,
        embedding=decoder_embedding_cfg,
    )

    # Define configuration for the encoder
    config_encoder = dict(
        rnn_hidden_size=args.hidden_size,
        rnn_num_layers=1,
        rnn_device=device,
        vocab_size=len(english_vocab) + 1,
        rnn_type="GRU",
        embedding_size=args.embedding_size,
    )

    # Define training configuration
    training_cfg = dict(
        device=device,
        output_vocab_size=len(french_vocab) + 1,
        english_vocab=english_vocab,
        french_vocab=french_vocab,
        epochs=10,
    )

    # Define translator configuration
    translator_cfg = dict(encoder=config_encoder, decoder=config_decoder, training=training_cfg)

    # Create the model
    model = AlignAndTranslate(**translator_cfg)

    # Train the model
    model.train(train_loader=train_dataloader, val_loader=val_dataloader)
