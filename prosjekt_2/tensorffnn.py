import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
from sklearn.metrics import accuracy_score

class FFNN:
    def __init__(self, 
                 n_hidden_neurons=2,
                 n_hidden_layers=5,
                 n_features=None,
                 n_outputs=2,       
                 epochs=10,
                 batch_size=100,
                 eta=0.1,
                 lmbd=0.0,
                 softmax=False,
                 activation_function='sigmoid'):
        
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs
        self.n_features = n_features
        self.softmax = softmax
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.mses = []
        self.accuracies = []
        
        self.model = self._build_model(activation_function)

    def _build_model(self, activation_function):
        model = tf.keras.models.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(self.n_features, activation=activation_function, kernel_regularizer=tf.keras.regularizers.l2(self.lmbd)))
        
        # Hidden layers
        for _ in range(self.n_hidden_layers):
            model.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation=activation_function, kernel_regularizer=tf.keras.regularizers.l2(self.lmbd)))
        
        # Output layer
        if self.softmax:
            model.add(tf.keras.layers.Dense(self.n_outputs, activation='softmax'))
        else:
            model.add(tf.keras.layers.Dense(self.n_outputs))
        
        # Compile the model
        if self.softmax:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.eta), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.eta), loss='mean_squared_error')
        
        return model

    def train(self, X, y):
        self.X = X
        self.y = y
        mse_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[mse_callback])

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            if self.softmax:
                accuracy = self.get_accuracy(self.y, self.cat_predict(self.predict(self.X)))
                print(accuracy)
                self.accuracies.append(accuracy)
                
            else:
                mse_value = logs['loss']
                print(mse_value)
                self.mses.append(mse_value)
        
    def predict(self, X):
        return self.model.predict(X)

    def cat_predict(self, probabilities):
        return np.round(probabilities)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)
    
    def get_mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def get_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

