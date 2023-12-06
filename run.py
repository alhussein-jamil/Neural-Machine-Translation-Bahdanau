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

    alignment_cfg = dict(
            input_size=args.enc_out_size + args.Ty,
            hidden_sizes=[10, 10],
            output_size=args.Ty,
            device= "cpu" if not torch.cuda.is_available() else "cuda",
            activation=torch.nn.ReLU(),
            last_layer_activation=torch.nn.Sigmoid(),
            dropout=0.2,
<<<<<<< HEAD
    )
=======
    )
>>>>>>> ea636e9 (Refactor code and update dependencies)
