import numpy as np
from models.RBM import RBM
from models.DBN import DBN

class DNN:
    def __init__(self):
        self.dbn = DBN()
        self.classification_layer = []

    def init_DNN(self, layer_sizes, classification_size):
        self.dbn.init_DBN(layer_sizes)
        self.classification_layer = RBM(layer_sizes[-1], classification_size)

    def pretrain_DNN(self, num_iterations, learning_rate, batch_size, data):
        self.dbn.train_DBN(num_iterations, learning_rate, batch_size, data)

    def calcul_softmax(self, rbm, data):
        output_data = np.dot(data,rbm.weights) + rbm.bias_hidden
        return np.exp(output_data) / np.sum(np.exp(output_data), axis=1, keepdims=True)

    def entree_sortie_reseau(self, data):
        sorties_par_couche = [data]
        hidden = data
        for rbm in self.dbn.rbm_layers:
            hidden = rbm.entree_sortie_RBM(hidden)
            sorties_par_couche.append(hidden)
        sorties_par_couche.append(self.calcul_softmax(self.classification_layer, hidden))
        return sorties_par_couche

    def retropropagation(self, data, labels, learning_rate, batch_size, epochs):
        num_data = data.shape[0]
        for epoch in range(epochs):
            for i_batch in range(0, num_data, batch_size):
                batch_data = data[i_batch:min(i_batch + batch_size, num_data)]
                batch_labels = labels[i_batch:min(i_batch + batch_size, num_data)]

                this_batch_size = batch_data.shape[0]
                # Forward pass
                layer_outputs = self.entree_sortie_reseau(batch_data)

                # error with one-hot form and softmax
                error = layer_outputs[-1] - batch_labels

                #calculate error for next layer, calculated before updating weights
                sigmo_prime = layer_outputs[-2] * (1 - layer_outputs[-2])
                backpropagated_error = np.dot(error,self.classification_layer.weights.T) * sigmo_prime

                # Update weights and biases of the classification layer
                delta_weights = np.dot(layer_outputs[-2].T, error) / this_batch_size
                delta_hidden_bias = np.mean(error, axis=0)

                self.classification_layer.weights -= learning_rate * delta_weights
                self.classification_layer.bias_hidden -= learning_rate * delta_hidden_bias
                # Backpropagate error through hidden layers of DBN (sigmoid)
                for i in range(len(self.dbn.rbm_layers)-1 , -1, -1):
                    rbm = self.dbn.rbm_layers[i]

                    #calculate error for next layer, calculated before updating weights
                    if i != 0:
                        sigmo_prime = layer_outputs[i] * (1 - layer_outputs[i])
                        next_backpropagated_error = np.dot(backpropagated_error, rbm.weights.T) * sigmo_prime

                    # Compute error for RBM layer
                    delta_weights = np.dot(layer_outputs[i].T, backpropagated_error) / this_batch_size
                    delta_hidden_bias = np.mean(backpropagated_error, axis=0)
                    rbm.weights -= learning_rate * delta_weights
                    rbm.bias_hidden -= learning_rate * delta_hidden_bias

                    if i != 0 : backpropagated_error = next_backpropagated_error
            predicted = self.entree_sortie_reseau(data)[-1]
            entropie_croisee = - np.mean(labels * np.log(predicted + 1e-8))
            print(
                f'DNN - Epoch: {epoch+1}/{epochs}, Error : {entropie_croisee}')

    def test_DNN(self, data, labels):
        sorties = self.entree_sortie_reseau(data)[-1]
        predictions = np.argmax(sorties, axis=1)
        true_labels = np.argmax(labels, axis=1)
        accuracy = np.mean(predictions == true_labels)
        print("Accuracy:", accuracy)
        return(accuracy)