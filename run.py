import argparse

from src.data_preprocessing import load_data
from models.translation_models import AlignAndTranslate
import torch

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # Parse arguments
    parser.add_argument(
        "--train_len", type=int, default=10000, help="Number of training examples"
    )
    parser.add_argument(
        "--val_len", type=int, default=1000, help="Number of validation examples"
    )
    parser.add_argument(
        "--Tx", type=int, default=30, help="Length of the input sequence"
    )
    parser.add_argument(
        "--Ty", type=int, default=30, help="Length of the output sequence"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=12, help="Size of the hidden layers"
    )
    parser.add_argument(    
        "--max_out_units", type=int, default=23, help="Size of the hidden layers"
    )

    args = parser.parse_args()

    # Load data
    (
        (train_data, train_dataloader),
        (val_data, val_dataloader),
        (bow_en, bow_fr),
    ) = load_data(
        train_len=args.train_len, val_len=args.val_len, kx=1000, ky=1000, Tx=30, Ty=30
    )
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    
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
            input_size=args.hidden_size * 2 + config_rnn_decoder["hidden_size"],
            hidden_sizes=[],
            output_size=args.Ty,
            device= device,
            activation=torch.nn.ReLU(),
            last_layer_activation=torch.nn.Tanh(),
            dropout=0.2,
    )

    maxout_cfg = dict(
        input_size=args.hidden_size,
        output_size=args.max_out_units,
        num_units= len(bow_fr) + 1,
        device=device
    )

    config_decoder = dict(alignment=alignment_cfg, rnn=config_rnn_decoder, maxout=maxout_cfg)


    config_encoder = dict(
        rnn_hidden_size = args.hidden_size,
        rnn_num_layers = 1,
        rnn_device = device,
        vocab_size=len(bow_en) + 1,
        rnn_type="GRU"
        )
    
    training_cfg = dict(
        device = device,
        output_vocab_size = len(bow_fr) + 1,
    )

    translator_cfg = dict(encoder=config_encoder, decoder=config_decoder, training=training_cfg)

    model = AlignAndTranslate(**translator_cfg)

    model.train(train_loader=train_dataloader, val_loader=val_dataloader)