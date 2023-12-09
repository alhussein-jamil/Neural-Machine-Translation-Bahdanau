import unittest
import torch
from torch import nn
import torch.nn.functional as F
from models.NLLLoss import NLLLoss

class TestNLLLoss(unittest.TestCase):
    def setUp(self):
        # Définir des paramètres pour les tests
        self.batch_size = 2
        self.Tx = 3
        self.Ty = 2

        # Créer une instance de la classe NLLLoss
        self.nll_loss_example = NLLLoss(self.batch_size, self.Tx, self.Ty)

    def test_initialization(self):
        # Vérifier si les attributs sont correctement initialisés
        self.assertEqual(self.nll_loss_example.batch_size, self.batch_size)
        self.assertEqual(self.nll_loss_example.Tx, self.Tx)
        self.assertEqual(self.nll_loss_example.Ty, self.Ty)

    def test_nll_loss_calculation(self):
        # Définir les valeurs de test
        probas_tensor = torch.tensor([[0.9, 0.99, 0.9], [0.6, 0.2, 0.1]], requires_grad=True)
        target = torch.tensor([[1, 0], [2, 1]])

        # Calculer la perte NLL
        self.nll_loss_example.calculate_nll(probas_tensor, target)

        # Ajouter des assertions en fonction des résultats attendus
        self.assertIsNotNone(self.nll_loss_example.probas_tensor)
        self.assertIsNotNone(self.nll_loss_example.target)
        self.assertIsNotNone(self.nll_loss_example.output)

# Créer un test runner et exécuter les tests
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestNLLLoss))
