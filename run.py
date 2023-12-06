<<<<<<< HEAD
import argparse

from src.data_preprocessing import load_data
<<<<<<< HEAD
# from models.translation_models import AlignAndTranslate
import torch
from src.translation_model import AlignAndTranslate
=======
from models.translation_models import AlignAndTranslate
import torch
>>>>>>> ea636e9 (Refactor code and update dependencies)

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
        "--enc_out_size", type=int, default=32, help="Size of the encoder output"
    )


    args = parser.parse_args()

    # Load data
    (
        (train_data, train_dataloader),
        (val_data, val_dataloader),
        (bow_en, bow_fr),
    ) = load_data(
        train_len=args.train_len, val_len=args.val_len, n=30000, m=30000, Tx=30, Ty=30
    )
    device = "cpu" if not torch.cuda.is_available() else "cuda"


    config_rnn_decoder = dict(
            input_size=args.enc_out_size * 2,
            hidden_size=10,
            num_layers=1,
            device=device,
            dropout=0,
            type="GRU",
            bidirectional=False,
        )
    alignment_cfg = dict(
            input_size=args.enc_out_size * 2 + config_rnn_decoder["hidden_size"],
            hidden_sizes=[10, 10],
            output_size=args.Ty,
            device= device,
            activation=torch.nn.ReLU(),
            last_layer_activation=torch.nn.Sigmoid(),
            dropout=0.2,
    )
    config_decoder = dict(alignment=alignment_cfg, rnn=config_rnn_decoder)


    config_encoder = dict(
        rnn_hidden_size = args.enc_out_size,
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