import torch
import torch.nn.functional as F
import nltk
import numpy as np

class BLEUScoreExample:
    """
    Classe pour calculer le score BLEU.

    Attributes:
        reference_tensor (torch.Tensor): Tenseur de références avec dimensions (batch_size, Tx).
        candidate_tensor (torch.Tensor): Tenseur de candidatures avec dimensions (batch_size, Ty).
    """

    def __init__(self, reference_tensor, candidate_tensor):
        
        self.reference_tensor = reference_tensor
        self.candidate_tensor = candidate_tensor

    def calculate_bleu_score(self):
        """
        Calcule le score BLEU entre les références et les candidatures pour chaque paire dans le même ordre.
        """
        bleu_scores = []

        for reference_sequence, candidate_sequence in zip(self.reference_tensor, self.candidate_tensor):

            reference_list = list(reference_sequence.numpy())
            candidate_list = list(candidate_sequence.numpy())

            # Calculer le score BLEU pour chaque paire
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_list], candidate_list)
            bleu_scores.append(BLEUscore)


        print("BLEU SCORE =", np.mean(bleu_scores))

# Exemple d'utilisation à l'extérieur de la classe
reference_tensor = torch.tensor([[1, 2, 3, 4, 8], [4, 5, 7, 10, 5]])  # Exemple de tenseur de références (batch_size, Tx)
candidate_tensor = torch.tensor([[1, 2, 3, 4, 8], [5, 5, 7, 10, 5]])  # Exemple de tenseur de candidatures (batch_size, Ty)

bleu_score_example = BLEUScoreExample(reference_tensor, candidate_tensor)
bleu_score_example.calculate_bleu_score()