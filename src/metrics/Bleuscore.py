import torch
import torch.nn.functional as F
import numpy as np
import collections
import nltk

class BLEUScore:
    
    """
    Classe pour calculer le score BLEU.

    Attributes:
        reference_tensor (torch.Tensor): Tenseur de références avec dimensions (batch_size, Tx).
        candidate_tensor (torch.Tensor): Tenseur de candidatures avec dimensions (batch_size, Ty).
    """

    def __init__(self, reference_tensor, candidate_tensor, n=2):
        
        self.reference_tensor = reference_tensor
        self.candidate_tensor = candidate_tensor
        self.n=n # nombre de n-grammes
        self.bleu_scores = []

    def calculate_bleu_score(self):
        """
        Calcule le score BLEU entre les références et les candidatures pour chaque paire dans le même ordre.
        """
        for reference_sequence, candidate_sequence in zip(self.reference_tensor, self.candidate_tensor):

            reference_list = list(reference_sequence.numpy())
            candidate_list = list(candidate_sequence.numpy())


           # Calcul des n-grammes pour la référence et le candidat
            reference_ngrams = [tuple(reference_list[i:i + self.n]) for i in range(len(reference_list) - self.n + 1)]
            candidate_ngrams = [tuple(candidate_list[i:i + self.n]) for i in range(len(candidate_list) - self.n + 1)]

            # Comptage des n-grammes dans la référence
            reference_ngram_counts = collections.Counter(reference_ngrams)

            # Comptage des n-grammes dans le candidat qui correspondent à la référence
            matching_ngram_counts = sum(min(reference_ngram_counts[ngram], candidate_ngrams.count(ngram)) for ngram in set(candidate_ngrams))

            # Calcul du score BLEU
            BLEUscore = matching_ngram_counts / len(candidate_ngrams)

            self.bleu_scores.append(BLEUscore)

        return torch.tensor(self.bleu_scores)

# Exemple d'utilisation à l'extérieur de la classe
# reference_tensor = torch.tensor([[1, 2, 3, 4, 8], [4, 5, 7, 10, 5]])  # Exemple de tenseur de références (batch_size, Tx)
# candidate_tensor = torch.tensor([[1, 2, 3, 4, 8], [5, 5, 7, 10, 5]])  # Exemple de tenseur de candidatures (batch_size, Ty)

# bleu_score_example = BLEUScore(reference_tensor, candidate_tensor)
# bleu_score_example.calculate_bleu_score()


class BLEUScoreNLTK:
    """
    Classe pour calculer le score BLEU.

    Attributes:
        reference_tensor (torch.Tensor): Tenseur de références avec dimensions (batch_size, Tx).
        candidate_tensor (torch.Tensor): Tenseur de candidatures avec dimensions (batch_size, Ty).
    """

    def __init__(self, reference_tensor, candidate_tensor):
        
        self.reference_tensor = reference_tensor
        self.candidate_tensor = candidate_tensor
        self.bleu_scores = []

    def calculate_bleu_score(self):
        """
        Calcule le score BLEU entre les références et les candidatures pour chaque paire dans le même ordre.
        """

        for reference_sequence, candidate_sequence in zip(self.reference_tensor, self.candidate_tensor):

            reference_list = list(reference_sequence.numpy())
            candidate_list = list(candidate_sequence.numpy())

            # Calculer le score BLEU pour chaque paire
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_list], candidate_list)
            self.bleu_scores.append(BLEUscore)

        return torch.tensor(self.bleu_scores)


        

