import torch
import torch.nn.functional as F

class NLLLoss:
    """
    Classe pour l'utilisation de la fonction de perte Negative Log Likelihood (NLLLoss)
    avec PyTorch.

    Attributes:
        batch_size (int): Taille du batch.
        Tx (int): Dimension de la séquence en entrée.
        Ty (int): Dimension de la séquence en sortie.
        probas_tensor (torch.Tensor): Tenseur de probabilités.
        target (torch.Tensor): Tenseur des étiquettes cibles pour chaque exemple dans le batch.
        output (torch.Tensor): Tenseur des log-probabilités après l'application du log.
    """
    def __init__(self,batch_size,Tx,Ty):
        # Définir les dimensions du batch et des séquences
        self.batch_size = batch_size
        self.Tx = Tx
        self.Ty = Ty
        

    def calculate_nll(self, probas_tensor, target):
        # Créer un tenseur de probabilités (log-probabilités pour simplifier ici)
        self.probas_tensor = probas_tensor.requires_grad_(True)

        # Les étiquettes cibles pour chaque exemple dans le batch
        self.target = target

        # Appliquer Log à l'output (log-probabilités)
        self.output = torch.log(self.probas_tensor)

        # Ajouter une dimension pour correspondre aux attentes de la fonction nll_loss
        self.output = self.output.unsqueeze(1)

        # Sélectionner uniquement les étiquettes pour le premier pas de temps
        target_step1 = self.target[:, 0]

        # Utiliser NLLLoss
        loss = F.nll_loss(self.output[:, 0, :], target_step1)

        # Calculer le gradient
        loss.backward()

        print(loss.item())
        return loss.item()

# Exemple d'utilisation à l'extérieur de la classe

# probas_tensor = torch.tensor([[0.9, 0.99, 0.9], [0.6, 0.2, 0.1]], requires_grad=True)

# target = torch.tensor([[1, 0], [2, 1]])

# nll_loss_example = NLLLoss(2,3,2)
# nll_loss_example.calculate_nll(probas_tensor, target)

#Le test avec un tenseur des probas de 1 partout donne bien le résultat attendu qui est NLL=0.
# probas_tensor = torch.ones(2, 3)
# probas_tensor.requires_grad=True 