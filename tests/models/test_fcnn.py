from models.fcnn import FCNN
import torch
from torch import nn

if __name__ == "__main__":
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model = FCNN(
        input_size=3,
        hidden_sizes=[10,10],
        output_size=10,
        device=device,
        activation=nn.ReLU(),
        dropout=0.5,
    )
    print(model)
    #batch size de 5, 3 features en entrée
    x = torch.randn(5, 3).to(device)

    out = model(x)

    assert out.shape == (5, 10)
    