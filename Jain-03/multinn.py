import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.mnn = []
        self.weights = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        self.num_nodes = num_nodes
        if not self.mnn:
            weights = np.random.randn(self.input_dimension, self.num_nodes)
        else:
            weights = np.random.randn(self.mnn[-1]["weights"].shape[1], self.num_nodes)
        bias = np.random.randn(self.num_nodes)
        self.weights.append(None)
        layer = {"transfer_function":transfer_function, "weights":weights, "bias":bias}
        self.mnn.append(layer)
        

    def get_weights_without_biases(self, layer_number):
        return self.mnn[layer_number]["weights"]

    def get_biases(self, layer_number):
        return self.mnn[layer_number]["bias"]

    def set_weights_without_biases(self, weights, layer_number):
        self.mnn[layer_number]["weights"]  = weights
        

    def set_biases(self, biases, layer_number):
        self.mnn[layer_number]["bias"] = biases

    def calculate_loss(self, y, y_hat):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat, name=None))
        
        


    def predict(self, X):
        x = X
        for i in range(len(self.mnn)):
            a = tf.matmul(x,self.mnn[i]["weights"]) + self.mnn[i]["bias"]
            if self.mnn[i]["transfer_function"] == "Linear" or self.mnn[i]["transfer_function"] == "linear":
                a = a
            elif self.mnn[i]["transfer_function"] == "Relu" or self.mnn[i]["transfer_function"] == "relu":
                a = tf.nn.relu(a, name='ReLU')
            else:
                a = tf.nn.sigmoid(a, name='sigmoid')
            x = a
        return x

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        for i in range(num_epochs):
            for k in range (0, len(X_train),batch_size):
                x = X_train[k:k+batch_size]
                r = []
                y = y_train[k:k+batch_size]
                s = []
               
                for i in range(len(self.mnn)):
                    r.append(self.mnn[i]["weights"])
#                     self.weights.append(self.mnn[i]["weights"])
                    s.append(self.mnn[i]["bias"])
                with tf.GradientTape() as tape:
                    pred = self.predict(x)
                    loss = self.calculate_loss(y, pred)
                    dl_dw, dl_db = tape.gradient(loss, [r, s])
                for m in range(len(r)):
                    wo = alpha*dl_dw[m]
                    bo = alpha*dl_db[m]
                    r[m].assign_sub(wo)
                    s[m].assign_sub(bo)

    def calculate_percent_error(self, X, y):
        pred = self.predict(X)
        pred = np.argmax(pred, axis = 1)
        c = 0
        for i in range(len(pred)):
            if y[i] != pred[i]:
                c+=1
        return(c/len(pred))

    def calculate_confusion_matrix(self, X, y):
        pred = self.predict(X)
        pred = np.argmax(pred, axis = 1)
        return tf.math.confusion_matrix(y,pred)
