import numpy as np
import matplotlib.pyplot as plt

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.bias_visible = np.zeros(num_visible)
        self.bias_hidden = np.zeros(num_hidden)
        self.weights = np.random.normal(size=(num_visible, num_hidden)) * np.sqrt(10**(-2))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def entree_sortie_RBM(self, visible):
        return self.sigmoid(visible @ self.weights + self.bias_hidden)

    def sortie_entree_RBM(self, hidden):
        return self.sigmoid(hidden @ self.weights.T + self.bias_visible)

    def train_RBM(self, data, learning_rate, batch_size, epochs, verbose=False):
        num_visible, num_hidden = self.weights.shape

        weights_history = []
        losses = []

        for epoch in range(epochs):

            np.random.shuffle(data)
            num_data = data.shape[0]
            for i_batch in range(0, num_data, batch_size):
                batch = data[i_batch:min(i_batch + batch_size, num_data), :]
                batch_size_i = batch.shape[0]

                visible0 = np.copy(batch)
                hidden_prob0 = self.entree_sortie_RBM(visible0)
                hidden0 = (np.random.rand(batch_size_i, num_hidden) < hidden_prob0) * 1

                visible_prob1 = self.sortie_entree_RBM(hidden0)
                visible1 = (np.random.rand(batch_size_i, num_visible) < visible_prob1) * 1

                hidden_prob1 = self.entree_sortie_RBM(visible1)

                delta_bias_visible = np.mean(visible0 - visible1, axis=0)
                delta_bias_hidden = np.mean(hidden_prob0 - hidden_prob1, axis=0)
                delta_weights = np.mean(np.expand_dims(visible0, axis=-1) * np.expand_dims(hidden_prob0, axis=1) -
                                        np.expand_dims(visible1, axis=-1) * np.expand_dims(hidden_prob1, axis=1), axis=0)

                self.bias_visible += learning_rate * delta_bias_visible
                self.bias_hidden += learning_rate * delta_bias_hidden
                self.weights += learning_rate * delta_weights

                weights_history.append(np.mean(self.weights))

            # Reconstruction error
            hidden = self.entree_sortie_RBM(data)
            reconstructed_data = self.sortie_entree_RBM(hidden)
            loss = np.mean((data - reconstructed_data) ** 2)
            losses.append(loss)
            if epoch % 10 == 0 and verbose:
                print("Epoch " + str(epoch) + "/" + str(epochs) + " - Loss: " + str(loss))

        if verbose:
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Evolution of the loss over ' + str(epochs) + ' epochs')
            plt.show()
            print("Final loss:", losses[-1])

            plt.xlabel('Epochs')
            plt.ylabel('Mean elements of weight matrix')
            plt.plot(weights_history)
            plt.show()

    def generer_image_RBM(self, num_images, num_iterations, image_size):
        num_visible, num_hidden = self.weights.shape
        images = []
        for _ in range(num_images):
            visible = (np.random.rand(num_visible) < 0.5) * 1
            for _ in range(num_iterations):
                hidden = (np.random.rand(num_hidden) < self.entree_sortie_RBM(visible)) * 1
                visible = (np.random.rand(num_visible) < self.sortie_entree_RBM(hidden)) * 1
            images.append(visible.reshape(image_size))
        return images
