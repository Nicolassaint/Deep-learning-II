from models.RBM import RBM
import numpy as np
import matplotlib.pyplot as plt

class DBN:
    def __init__(self):
        self.rbm_layers = []

    def init_DBN(self, layer_sizes):
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i + 1])
            self.rbm_layers.append(rbm)

    def train_DBN(self, num_iterations, learning_rate, batch_size, data):
        for rbm in self.rbm_layers:
            rbm.train_RBM(data, learning_rate, batch_size, num_iterations)
            data = rbm.entree_sortie_RBM(data)

    def generer_image_DBN(self, nb_data, nb_gibbs):
        visible_size = len(self.rbm_layers[-1].bias_visible)  # Taille de la couche visible du dernier RBM
        generated_images = self.rbm_layers[-1].generer_image_RBM(nb_data, nb_gibbs, (visible_size,))
        for i in range(len(self.rbm_layers) - 2, -1, -1):
            generated_images = (
                np.random.rand(nb_data, len(self.rbm_layers[i].bias_visible))
                < self.rbm_layers[i].sortie_entree_RBM(generated_images)
            ) * 1
        return generated_images