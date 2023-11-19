import numpy as np
import matplotlib.pyplot as plt 

class GradientDescent():
    '''
    Class for regular gradient descent.
    '''
    def __init__(self, X, y, learning_rate, max_epochs, lamb = 0):
        self.X = X  # Design matrix
        self.y = y  # Values to predict
        self.learning_rate = learning_rate  # Learning rate
        self.max_epochs = max_epochs  # Maximum number of epochs
        self.lamb = lamb # Ridge term
        self.n = X.shape[0]  # Number of samples
        self.parameters = X.shape[1]  # Number of features
        self.beta = np.random.uniform(-1, 1, (self.parameters, 1))
        self.epochs = 0
        self.tol = 1e-8

    def compute_gradients(self):
        gradients = -2.0 / self.n * np.dot(self.X.T, self.y - np.dot(self.X, self.beta)) + 2*self.lamb*self.beta
        return gradients
    
    def update_parameters(self, gradients):
        self.beta -= self.learning_rate * gradients

    def fit(self):
        y_pred = self.predict(self.X)
        while self.epochs < self.max_epochs and np.linalg.norm(self.y - y_pred) > self.tol:
            gradients = self.compute_gradients()
            self.update_parameters(gradients)
            y_pred = self.predict(self.X)
            self.epochs +=1
    
    def predict(self, X):
        return X@self.beta
    
    def betas(self):
        return self.beta
    
    def change_learning_rate(self, new_lerning_rate):
        self.learning_rate = new_lerning_rate

    def change_lamb(self, new_lamb):
        self.lamb = new_lamb

    def reset_model(self):
        self.beta = np.random.uniform(-1, 1, (self.parameters, 1))
        self.epochs = 0


class GDWithMomentum(GradientDescent):
    '''
    Class for regular gradient descent with momentum
    '''
    def __init__(self, X, y, learning_rate, max_epochs, lamb = 0, momentum = 0.2):
        super().__init__(X, y, learning_rate, max_epochs, lamb)
        self.momentum = momentum
        self.velocity = np.zeros((self.parameters, 1))

    def update_paramters(self, gradients):
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        self.beta -= self.velocity


class SGD(GradientDescent):
        '''
        Class for stochastic regular gradient descent
        '''
        def __init__(self, X, y, learning_rate, max_epochs, lamb = 0, n_batches = 5):
            super().__init__(X, y, learning_rate, max_epochs, lamb)
            self.n_batches = n_batches
            self.mini_batch_size = None

        def get_mini_batches(self):
            self.mini_batch_size = len(self.y) // self.n_batches
            mini_batches = []
            
            #Random shuffle 
            data = np.hstack((self.X, self.y))
            np.random.shuffle(data)

            for i in range(self.n_batches):
                mini_batch = data[i * self.mini_batch_size:(i + 1)*self.mini_batch_size, :]
                X_mini = mini_batch[:, :-1]
                y_mini = mini_batch[:, -1].reshape((-1, 1))
                mini_batches.append((X_mini, y_mini))
            
            if len(self.y) % self.mini_batch_size != 0:
                mini_batch = data[self.n_batches * self.mini_batch_size:]
                X_mini = mini_batch[:, :-1]
                y_mini = mini_batch[:, -1].reshape((-1, 1))
                mini_batches.append((X_mini, y_mini))
            
            return mini_batches
        
        def fit(self):
            for i in range(self.max_epochs):
                mini_batches = self.get_mini_batches()
                for X_m, y_m in mini_batches:
                    gradients = self.compute_gradients()
                    self.update_parameters(gradients)


class SGDWithMomentum(SGD):
        '''
        Class for stochastic gradient descent w/momentum
        '''
        def __init__(self, X, y,learning_rate, max_epochs, lamb = 0, n_batches = 5, momentum = 0.2):
            super().__init__(X, y, learning_rate, max_epochs, lamb, n_batches)
            self.momentum = momentum
            self.velocity = np.zeros((self.parameters, 1))
        
        def update_paramters(self, gradients):
            self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
            self.beta -= self.velocity


class Adagrad(GradientDescent):
    '''
    Regular Adagrad
    '''
    def __init__(self, X,y, learning_rate, max_epochs, lamb = 0, epsilon = 1e-8):
        super().__init__(X, y,learning_rate, max_epochs, lamb)
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1))

    def update_paramters(self, gradients):
        self.s += gradients**2
        self.beta -= self.learning_rate * gradients /  ( np.sqrt(self.s) + self.epsilon)

class AdagradWithMomentum(GDWithMomentum):
    """
    Adagrad with momentum
    """
    def __init__(self, X,y,learning_rate, max_epochs, lamb = 0, momentum = 0.2, epsilon = 1e-8):
        super().__init__(X, y, learning_rate, max_epochs, lamb, momentum)
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1))

    def update_paramters(self, gradients):
        self.s += gradients**2
        self.beta -= self.learning_rate * gradients /  ( np.sqrt(self.s) + self.epsilon)

class SGDAdagrad(SGD):
    """
    Adagrad with Stochastic gradient descent 
    """
    def __init__(self, X,y,learning_rate, max_epochs, lamb = 0,  n_batches = 5, epsilon = 1e-8):
        super().__init__(X, y, learning_rate, max_epochs, lamb, n_batches)
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1))
    
    def update_paramters(self, gradients):
        self.s += gradients**2
        self.beta -= self.learning_rate * gradients /  ( np.sqrt(self.s) + self.epsilon)


    
class SGDAdagradWithMomentum(SGDWithMomentum):
    """
    Adagrad with Stochastic gradient descent with momentum
    """
    def __init__(self, X, y, learning_rate, max_epochs, lamb = 0, n_batches = 5, momentum = 0.2, epsilon=1e-8):
        super().__init__(X, y, learning_rate, max_epochs, lamb, n_batches, momentum)
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1))

    def update_paramters(self, gradients):
        self.s += gradients**2
        self.beta -= self.learning_rate * gradients /  ( np.sqrt(self.s) + self.epsilon)
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients /  ( np.sqrt(self.s) + self.epsilon)
        self.beta -= self.velocity

class RMSprop(GradientDescent):
    def __init__(self,X,y, learning_rate, max_epochs,lamb = 0, decay_rate = 0.99, epsilon = 1e-8):
        super().__init__(X, y, learning_rate, lamb, max_epochs)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1)) #Average squared gradient 

    def update_paramters(self, gradients):
        self.s = self.decay_rate * self.s + (1 - self.decay_rate)*gradients**2
        self.beta -= self.learning_rate*gradients / (np.sqrt(self.s) + self.epsilon)

class SGD_RMSprop(SGD):
    def __init__(self, X, y, learning_rate, max_epochs, lamb = 0, n_batches = 5, decay_rate = 0.99, epsilon = 1e-8):
        super().__init__(X, y, learning_rate, max_epochs, lamb, n_batches)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.s = np.zeros((self.parameters,1)) #Average squared gradient 

    def update_paramters(self, gradients):
        self.s = self.decay_rate * self.s + (1 - self.decay_rate)*gradients**2
        self.beta -= self.learning_rate*gradients / (np.sqrt(self.s) + self.epsilon)
                         

class Adam(GradientDescent):
    def __init__(self, X, y, learning_rate, max_epochs, lamb = 0, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-8):
        super().__init__(X, y,learning_rate, max_epochs, lamb)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = np.zeros((self.parameters,1))
        self.s = np.zeros((self.parameters,1))
        self.t = 0 
    
    def update_paramters(self, gradients):
        self.t += 1 
        self.m = self.beta_1*self.m + (1- self.beta_1)*gradients
        self.s = self.beta_2*self.s + (1-self.beta_2)*gradients**2

        m_cor = self.m / (1 - self.beta_1**self.t)
        s_cor = self.s / (1 - self.beta_2**self.t)

        self.beta -= self.learning_rate*m_cor / (np.sqrt(s_cor) + self.epsilon)

class SGD_Adam(SGD):
        def __init__(self, X,y, learning_rate, max_epochs, lamb = 0,  n_batches = 5, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-8):
            super().__init__(X,y, learning_rate, max_epochs, lamb, n_batches)
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon
            self.m = np.zeros((self.parameters,1))
            self.s = np.zeros((self.parameters,1))
            self.t = 0 

        def update_paramters(self, gradients):
            self.t += 1 
            self.m = self.beta_1*self.m + (1- self.beta_1)*gradients
            self.s = self.beta_2*self.s + (1-self.beta_2)*gradients**2

            m_cor = self.m / (1 - self.beta_1**self.t)
            s_cor = self.s / (1 - self.beta_2**self.t)

            self.beta -= self.learning_rate*m_cor / (np.sqrt(s_cor) + self.epsilon)
            
