import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class artificial_neuron:
    def __init__(self, n_iter=1000, learning_rate=0.01):
        self.coef_ = None # W
        self.bias_ = None # b
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate        
        self.train_loss_ = []        
        self.train_acc_ = []
        self.test_acc_ = []
        self.tes_acc_ = []
        
        #light
        self.loss_ = []
        self.acc_ = []        

    def predict_proba(self,X):
        Z = X.dot(self.coef_) + self.bias_        
        return 1 / (1 + np.exp(-Z))
    
    def predict(self, X):
        A = self.predict_proba(X)    
        return A >= 0.5
        
    def display_loss_light(self):
        plt.plot(self.loss_)
        plt.show()
        
    def display_loss(self):      
        plt.plot(self.train_loss_, label='train_loss')
        plt.plot(self.test_loss_, label='test_loss')
        plt.legend()
        plt.show()

    def display_loss_light(self):      
        plt.plot(self.train_loss_, label='train_loss')        
        plt.legend()
        plt.show()
        
    def display_acc_light(self):
        plt.plot(self.acc_)
        plt.show()        
        
    def display_acc(self):      
        plt.plot(self.train_acc_, label='test_loss')
        plt.plot(self.test_acc_, label='test_acc')
        plt.legend()
        plt.show()        

    def log_loss(self, y, A):
        return (1 / len(y)) *  (np.sum(-y * np.log(A + 10E-15) - (1 - y) *  np.log(1 - A + 10E-15)))
        
    def fit_light (self, X, y):       
       #initialisation W, b
        self.coef_ = np.random.randn(X.shape[1],1)    
        self.bias_ = np.random.randn(1)

        self.loss_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser le loss.
        self.acc_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser acc.

        #Apprentissage
        for i in tqdm(range(self.n_iter_)):
            #Activation               
            A = self.predict_proba(X)            

            if i%10 == 0:
                #loss
                self.loss_.append(self.log_loss(y, A))
            
                #accuracy
                self.acc_.append(accuracy_score(y, self.predict(X)))

            #Gradients
            dW = 1 / len(y) * np.dot(X.T, A - y)
            db = 1 / len(y) * np.sum(A - y)

            #update
            self.coef_ = self.coef_ - self.learning_rate_ * dW
            self.bias_ = self.bias_ - self.learning_rate_ * db     
    
    def fit(self, X_train, y_train, X_test, y_test):       
        #initialisation W, b
        self.coef_ = np.random.randn(X_train.shape[1],1)    
        self.bias_ = np.random.randn(1)

        self.train_loss_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser le loss.
        self.train_acc_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser acc.
        
        self.test_loss_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser le loss.
        self.test_acc_ = [] #ici, on réinitialise le loss car si on refit notre modèle, il faut réinitialiser acc.

        #Apprentissage
        for i in tqdm(range(self.n_iter_)):
            #Activation               
            A_train = self.predict_proba(X_train) #Dans la boucle d'apprentissage, le train_set entre en jeux.            
            A_test = self.predict_proba(X_test) #Dans la boucle d'apprentissage, le test_set entre en jeux.            

            if i%10 == 0:
                #Train loss
                self.train_loss_.append(self.log_loss(y_train, A_train))            
                #accuracy
                self.train_acc_.append(accuracy_score(y_train, self.predict(X_train)))

                #Test loss                
                self.test_loss_.append(self.log_loss(y_test, A_test))            
                #accuracy
                self.test_acc_.append(accuracy_score(y_test, self.predict(X_test)))                

            #Gradients
            dW = 1 / len(y_train) * np.dot(X_train.T, A_train - y_train)
            db = 1 / len(y_train) * np.sum(A_train - y_train)

            #update
            self.coef_ = self.coef_ - self.learning_rate_ * dW
            self.bias_ = self.bias_ - self.learning_rate_ * db  
