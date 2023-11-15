from models.rnn import RNN
import torch

if __name__ == "__main__":
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model = RNN(
        input_size=2,
        hidden_size=5,
        num_layers =1,
        device=device,
    )
    print(model)
    #batch size de 5 , une sequence de 3 mots, chaque mot à 2 features
    x = torch.randn(5, 3, 2).to(device)
    
    out = model(x)
    
    assert out.shape == (5, 3, 5)