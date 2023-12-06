import unittest
import torch
from torch import nn
from models.translation_models import AlignAndTranslate

class TestAlignAndTranslate(unittest.TestCase):
    def setUp(self):
        # Initialisation pour chaque test
        self.encoder_params = {"input_size": 100, "hidden_size": 50}
        self.decoder_params = {"output_size": 10}
        self.training_config = {
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(),
            "device": "cpu",
            "epochs": 2,
            "batch_size": 32,
            "print_every": 100,
            "save_every": 1000,
            "checkpoint": "checkpoint.pth",
            "best_model": "best_model.pth",
            "output_vocab_size": 100,
        }

        self.model = AlignAndTranslate(encoder=self.encoder_params, decoder=self.decoder_params, training=self.training_config)

        # Exemple de données factices pour les tests
        self.example_input_data = torch.randn((32, 100))  
        self.example_target_data = torch.randn((32, 100))  

    def test_forward_pass(self):
        # Vérifier si la passe avant (forward pass) fonctionne sans erreurs
        output_tensor = self.model(self.example_input_data)
        assert output_tensor.shape == torch.Size([32, 10])  # Adapter la taille en fonction de votre modèle

    def test_train_step(self):
        # Vérifier si une étape d'entraînement fonctionne sans erreurs
        loss = self.model.train_step(self.example_input_data, self.example_target_data)
        assert loss >= 0

    def test_train(self):
        # Vérifier si la fonction d'entraînement sur plusieurs époques fonctionne sans erreurs
        train_dataset = torch.utils.data.TensorDataset(self.example_input_data, self.example_target_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(self.example_input_data, self.example_target_data)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        self.model.train(train_loader, val_loader)


    def test_save_model(self):
        # Vérifier si la fonction de sauvegarde du modèle fonctionne sans erreurs
        self.model.save_model("test_checkpoint.pth")
        assert torch.isfile("test_checkpoint.pth")  

    def test_model_initialization(self):
        # Vérifier si l'initialisation du modèle est correcte
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, self.encoder_params["input_size"])
        self.assertEqual(self.model.hidden_size, self.encoder_params["hidden_size"])
        self.assertEqual(self.model.output_size, self.decoder_params["output_size"])
        self.assertIsInstance(self.model.optimizer, torch.optim.Adam)
        self.assertEqual(self.model.device, self.training_config["device"])
        self.assertEqual(self.model.epochs, self.training_config["epochs"])
        self.assertEqual(self.model.batch_size, self.training_config["batch_size"])
        self.assertEqual(self.model.print_every, self.training_config["print_every"])
        self.assertEqual(self.model.save_every, self.training_config["save_every"])
        self.assertEqual(self.model.checkpoint, self.training_config["checkpoint"])
        self.assertEqual(self.model.best_model, self.training_config["best_model"])
        self.assertEqual(self.model.output_vocab_size, self.training_config["output_vocab_size"])

if __name__ == '__main__':
    unittest.main()
