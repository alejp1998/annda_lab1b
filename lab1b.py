# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Miguel Garcia Naude
# 19980512-T697
# magn2@kth.se

# Jonne van Haastregt
# 20010713-3316 
# jmvh@kth.se 

# Load packages
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary variables
colors = ['#1E90FF','#FF69B4']

# NEURAL NETWORK CLASS
class NeuralNetwork:
    def __init__(self, arch, activation_fn, activation_fn_der, momentum, alpha, lr, n_epochs):
        """ Constructor of the Neural Network."""
        self.arch = arch
        self.n_layers = len(self.arch)-1
        self.momentum = momentum
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_matrices = [np.zeros((self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]
        self.momentum_matrices = self.weight_matrices
        self.activation_fn = activation_fn
        self.activation_fn_der = activation_fn_der
        self.forward_mem = [i for i in range(self.n_layers)]
        self.backward_mem = [i for i in range(self.n_layers)]
        self.X = 0
        self.T = 0
        self.O = 0

    def initialize_weights(self) :
        self.weight_matrices = [np.random.normal(0, 1, size=(self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]
        self.momentum_matrices = [np.zeros((self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]

    def training_data(self,X,T):
        # Account for bias term in input
        X = np.append(X,[np.ones(np.shape(X)[1])],axis=0)
        self.X = X
        self.T = T
        
    def forward_pass(self,X):
        # Vectorize activation function
        vfunc = np.vectorize(self.activation_fn)
        # Initial results vector
        prev_result = X

        # Iterate frontwards over layers
        for layer in range(self.n_layers):
            # Account for bias term
            prev_result = np.append(prev_result,[np.ones(np.shape(X)[1])],axis=0)

            # Calculate new result
            result = self.weight_matrices[layer].dot(prev_result) # Multiply by weights matrix
            self.forward_mem[layer] = result # Store result before applying non-linear activation fn
            result = vfunc(result) # Apply activation function
            
            # Set as prev result for next iteration
            prev_result = result
        
        self.O = result
        return self.O
    
    def backward_pass(self,T):
        # Vectorize activation function derivative
        vfunc = np.vectorize(self.activation_fn_der)
        # Final results vector
        next_delta = np.multiply(self.O - T,vfunc(self.forward_mem[-1]))
        self.backward_mem[-1] = next_delta

        # Iterate backwards over layers
        for layer in range(self.n_layers)[:-1] :
            mem_result = np.append(self.forward_mem[-2-layer],[np.ones(np.shape(self.forward_mem[-2-layer])[1])],axis=0)
            delta = np.multiply(self.weight_matrices[-1-layer].T.dot(next_delta),vfunc(mem_result))[:-1,:] 
            self.backward_mem[-2-layer] = delta  # Store result

            # Store as next result for prev iteration
            next_delta = delta
    
    def weights_update(self,X):
        # Vectorize activation function
        vfunc = np.vectorize(self.activation_fn)

        # Account for bias term
        X = np.append(X,[np.ones(np.shape(X)[1])],axis=0)

        # Initial layer update
        if not self.momentum :
            delta_weights = self.lr*self.backward_mem[0].dot(X.T)
        else : 
            self.momentum_matrices[0] = self.alpha*self.momentum_matrices[0] - (1-self.alpha)*self.backward_mem[0].dot(X.T)
            delta_weights = self.lr*self.momentum_matrices[0]
        # Update weights
        self.weight_matrices[0] = self.weight_matrices[0] + delta_weights

        # Iterate frontwards over rest of layers
        for layer in range(self.n_layers)[1:]:
            mem_result = vfunc(np.append(self.forward_mem[layer-1],[np.ones(np.shape(X)[1])],axis=0)).T
            if not self.momentum :
                delta_weights = self.lr*self.backward_mem[layer].dot(mem_result)
            else :
                self.momentum_matrices[layer] = self.alpha*self.momentum_matrices[layer] - (1-self.alpha)*self.backward_mem[layer].dot(mem_result)
                delta_weights = self.lr*self.momentum_matrices[layer]
            # Update weights
            self.weight_matrices[layer] = self.weight_matrices[layer] + delta_weights

    def classify(self,X,T,show=True) :
        T_guessed = 2*(self.forward_pass(X)>0) - 1

        # Index -1 class
        T_neg = T[T < 0]
        T_guessed_neg = T_guessed[:,T < 0]
        hits_neg = (T_guessed_neg == T_neg).sum()
        fails_neg = np.shape(T_neg)[0] - hits_neg
        accuracy_neg = round(hits_neg*100/np.shape(T_neg)[0],3)

        # Index +1 class
        T_pos = T[T > 0]
        T_guessed_pos = T_guessed[:,T > 0]
        hits_pos = (T_guessed_pos == T_pos).sum()
        fails_pos = np.shape(T_pos)[0] - hits_pos
        accuracy_pos = round(hits_pos*100/np.shape(T_pos)[0],3)

        # Overall accuracy
        hits = (T_guessed == T).sum()
        fails = np.shape(T)[0] - hits
        accuracy = round(hits*100/np.shape(T)[0],3)

        if show :
            print('Class A. Hits = {}, Fails = {}, Accuracy = {}%'.format(hits_neg,fails_neg,accuracy_neg))
            print('Class B. Hits = {}, Fails = {}, Accuracy = {}%'.format(hits_pos,fails_pos,accuracy_pos))
            print('Hits = {}, Fails = {}, Accuracy = {}%'.format(hits,fails,accuracy))

        return T_guessed, accuracy_pos, accuracy_neg, accuracy

    def train(self,X,T,show=False):
        mses = []
        errors = []
        for i in range(self.n_epochs) :
            # Perform weights update
            self.forward_pass(X)
            self.backward_pass(T)
            self.weights_update(X)

            # Compute missclassification
            T_guessed,accuracy_pos, accuracy_neg, accuracy = self.classify(X,T,show)
            errors.append(100-accuracy)
            mses.append(np.sum(np.square(self.O-T))/np.shape(T))
        
        return T_guessed, errors, mses

    def decision_boundary(self,X,K,L) :
        min1, max1 = X[0, :].min() - L, X[0, :].max() + L #1st feature
        min2, max2 = X[1, :].min() - L, X[1, :].max() + L #2nd feature

        # Input patterns to be sampled
        x1, x2 = min1, min2
        sampling_pattern = []
        for i1 in range(K+1) :
            x1 = min1 + (max1-min1)*(i1/K)
            for i2 in range(K+1) :
                x2 = min2 + (max2-min2)*(i2/K)
                sampling_pattern.append(np.array([x1,x2]))
        
        # Classify input pattern
        sampling_pattern = np.array(sampling_pattern).T
        boundary_samples = (2*(self.forward_pass(sampling_pattern)>0) - 1)[0,:]
        return sampling_pattern, boundary_samples


        
# HELPER FUNCTIONS
def activation_fn (x):
    return 2/(1 + np.exp(-x)) - 1

def activation_fn_der (x) :
    return (1/2)*(1 + activation_fn(x))*(1 - activation_fn(x))

def plot_data(X,T) :
    fig, ax = plt.subplots()
    ax.scatter(X[0,T>0],X[1,T>0], c=colors[0], label='Class B')
    ax.scatter(X[0,T<0],X[1,T<0], c=colors[1], label='Class A')
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Patterns and Labels')
    plt.show()

def plot_data_boundary(X,T,sampling_pattern,boundary_samples,L) :
    min1, max1 = X[0, :].min() - L, X[0, :].max() + L #1st feature
    min2, max2 = X[1, :].min() - L, X[1, :].max() + L #2nd feature
    fig, ax = plt.subplots()
    ax.scatter(sampling_pattern[0,boundary_samples<0],sampling_pattern[1,boundary_samples<0], c=colors[0], label='Classified as A', alpha = 0.1)
    ax.scatter(sampling_pattern[0,boundary_samples>0],sampling_pattern[1,boundary_samples>0], c=colors[1], label='Classified as B', alpha = 0.1)
    ax.scatter(X[0,T<0],X[1,T<0], c=colors[0], label='Class A')
    ax.scatter(X[0,T>0],X[1,T>0], c=colors[1], label='Class B')
    ax.set_xlim([min1,max1])
    ax.set_ylim([min2,max2])
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Sampled Decision Boundary')
    plt.show()

def plot_error(errors,title) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title(title+' over epochs')
    ax.plot(errors)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    plt.show()

def gen_data_clusters(N, mean_A1, mean_A2, cov_A, mean_B, cov_B) :
    # Class A
    X_A1 = np.random.multivariate_normal(mean_A1, cov_A, int(N/2)).T
    X_A2 = np.random.multivariate_normal(mean_A2, cov_A, int(N/2)).T
    X_A = np.append(X_A1, X_A2, axis=1)
    X_A = np.append(X_A,[1*np.ones(2*int(N/2))],axis=0) # Class label

    # Class B
    X_B = np.random.multivariate_normal(mean_B, cov_B, N).T
    X_B = np.append(X_B,[-np.ones(2*int(N/2))],axis=0) # Class label

    return X_A,X_B

def subsample_mix_classes(X_A,X_B,f_A,fB) :
    # Subsample classes
    random_subs_indices = np.random.choice(N, size=int(N*0.75), replace=False)
    X_A_subs = X_A[:,random_subs_indices]
    X_B_subs = X_B[:,random_subs_indices]
    X_A_disc = X_A[:,[i for i in range(N) if i not in random_subs_indices]]
    X_B_disc = X_B[:,[i for i in range(N) if i not in random_subs_indices]]

    # Mix classes
    random_col_indices = np.random.choice(int(1.5*N), size=int(1.5*N), replace=False)
    X_subs = np.append(X_A_subs,X_B_subs,axis=1)[:,random_col_indices]
    X_disc = np.append(X_A_disc,X_B_disc,axis=1)

    # Define labels vector
    T = X_subs[-1,:]
    X_subs = X_subs[:-1,:]
    T_disc = X_disc[-1,:]
    X_disc = X_disc[:-1,:]
