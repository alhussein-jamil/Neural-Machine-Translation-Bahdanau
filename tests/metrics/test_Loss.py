import unittest
import torch
from torch import nn
import torch.nn.functional as F
from metrics.Loss import Loss

class TestLoss(unittest.TestCase):
    def setUp(self):
        # Créer des exemples de tenseurs pour les tests
        self.x = torch.randn(2, 5, 10)  # Exemple de tenseur x
        self.y = torch.randint(0, 10, (2, 5))  # Exemple de tenseur y

        # Créer une instance de la classe Loss pour les tests
        self.loss_instance = Loss(loss_fn=nn.CrossEntropyLoss())

    def test_initialization(self):
        # Vérifier si l'attribut est correctement initialisé
        self.assertEqual(self.loss_instance.loss_fn, nn.CrossEntropyLoss())

    def test_forward_pass(self):
        # Exécuter le passage avant avec la méthode de la classe
        result = self.loss_instance(self.x, self.y)

        # Ajouter des assertions en fonction des résultats attendus
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(result.requires_grad)
        self.assertFalse(torch.isnan(result).any())

# Créer un test runner et exécuter les tests
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestLoss))
