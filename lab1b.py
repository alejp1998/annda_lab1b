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
    
    def classify_matrix(self,X,T,show=True) :
        T_guessed = (self.forward_pass(X) > 0 )* 2 - 1

        # Overall accuracy
        hits = 0
        for col in range(np.shape(T_guessed)[1]) :
            hits += (T_guessed[:,col] == T[:,col]).sum() == np.shape(T)[0]
        fails = np.shape(T)[0] - hits
        accuracy = round(hits*100/np.shape(T)[0],3)

        if show :
            print('Hits = {}, Fails = {}, Accuracy = {}%'.format(hits,fails,accuracy))

        return T_guessed, accuracy

    def train(self,X,T,X_valid,T_valid,show=False,matrix=False):
        mses, mses_valid = [], []
        errors, errors_valid = [], []
        for i in range(self.n_epochs) :
            # Perform weights update
            self.forward_pass(X)
            self.backward_pass(T)
            self.weights_update(X)

            # Compute missclassification
            if not matrix :
                T_guessed,accuracy_pos, accuracy_neg, accuracy = self.classify(X,T,show)
                T_guessed_valid,accuracy_pos_valid, accuracy_neg_valid, accuracy_valid = self.classify(X_valid,T_valid,show)
            else :
                T_guessed, accuracy = self.classify_matrix(X,T,show)
                T_guessed_valid, accuracy_valid = self.classify_matrix(X_valid,T_valid,show)

            errors.append(100-accuracy)
            mses.append(np.sum(np.square(self.forward_pass(X)-T))/np.shape(T))

            errors_valid.append(100-accuracy_valid)
            O_valid = self.forward_pass(X_valid)
            mses_valid.append(np.sum(np.square(O_valid-T_valid))/np.shape(T_valid))
        
        return errors, mses, errors_valid, mses_valid

    def validate(self,X,T,show=False) :
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

def plot_error(errors,errors_valid,title) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title(title+' over epochs')
    ax.plot(errors,label='Training')
    ax.plot(errors_valid,label='Validation')
    ax.legend()
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

def subsample_mix_classes(X_A,X_B,f_A,f_B) :
    # Subsample classes
    N_A, N_B = np.shape(X_A)[1], np.shape(X_B)[1]
    random_subs_indices_A = np.random.choice(N_A, size=int(N_A*f_A), replace=False)
    random_subs_indices_B = np.random.choice(N_B, size=int(N_B*f_B), replace=False)
    X_A_train = X_A[:,random_subs_indices_A]
    X_B_train = X_B[:,random_subs_indices_B]
    X_A_valid = X_A[:,[i for i in range(N_A) if i not in random_subs_indices_A]]
    X_B_valid = X_B[:,[i for i in range(N_B) if i not in random_subs_indices_B]]
    
    # Mix classes
    N_train = np.shape(X_A_train)[1] + np.shape(X_B_train)[1]
    N_valid = np.shape(X_A_valid)[1] + np.shape(X_B_valid)[1]
    random_col_indices_train = np.random.choice(N_train, size=N_train, replace=False)
    random_col_indices_valid = np.random.choice(N_valid, size=N_valid, replace=False)
    X_train = np.append(X_A_train,X_B_train,axis=1)[:,random_col_indices_train]
    X_valid = np.append(X_A_valid,X_B_valid,axis=1)[:,random_col_indices_valid]

    # Define labels vector
    T_train = X_train[-1,:]
    X_train = X_train[:-1,:]
    T_valid = X_valid[-1,:]
    X_valid = X_valid[:-1,:]

    return T_train, X_train, T_valid, X_valid

def subsample_mix_classes_complex(X_A,X_B,f_A,f_B,f_posneg) :
    # Separate between positive and negative
    X_A_pos = X_A[:,X_A[0,:] > 0]
    X_A_neg = X_A[:,X_A[0,:] < 0]
    
    # Subsample classes
    N_A_pos, N_A_neg, N_B = np.shape(X_A_pos)[1], np.shape(X_A_pos)[1], np.shape(X_B)[1]
    random_subs_indices_A_pos = np.random.choice(N_A_pos, size=int(N_A_pos*f_posneg), replace=False)
    random_subs_indices_A_neg = np.random.choice(N_A_neg, size=int(N_A_neg*(1-f_posneg)), replace=False)
    random_subs_indices_B = np.random.choice(N_B, size=int(N_B*f_B), replace=False)
    X_A_train = np.append(X_A_pos[:,random_subs_indices_A_pos],X_A_neg[:,random_subs_indices_A_neg],axis=1)
    X_B_train = X_B[:,random_subs_indices_B]
    X_A_valid = np.append(X_A_pos[:,[i for i in range(N_A_pos) if i not in random_subs_indices_A_pos]],X_A_neg[:,[i for i in range(N_A_neg) if i not in random_subs_indices_A_neg]],axis=1)
    X_B_valid = X_B[:,[i for i in range(N_B) if i not in random_subs_indices_B]]

    # Mix classes
    N_train = np.shape(X_A_train)[1] + np.shape(X_B_train)[1]
    N_valid = np.shape(X_A_valid)[1] + np.shape(X_B_valid)[1]
    random_col_indices_train = np.random.choice(N_train, size=N_train, replace=False)
    random_col_indices_valid = np.random.choice(N_valid, size=N_valid, replace=False)
    X_train = np.append(X_A_train,X_B_train,axis=1)[:,random_col_indices_train]
    X_valid = np.append(X_A_valid,X_B_valid,axis=1)[:,random_col_indices_valid]

    # Define labels vector
    T_train = X_train[-1,:]
    X_train = X_train[:-1,:]
    T_valid = X_valid[-1,:]
    X_valid = X_valid[:-1,:]

    return T_train, X_train, T_valid, X_valid

def generate_func_approx_data (N,feat_range,variance,bias,noise_variance=0):
    def gaussian_2d(x,y,variance,bias,noise_variance) :
        return np.exp(-(x**2 + y**2)/variance) + bias + np.random.normal(0, noise_variance)
    x = np.linspace(feat_range[0], feat_range[1],int(np.sqrt(N)))
    y = np.linspace(feat_range[0], feat_range[1], int(np.sqrt(N)))
    X, T = [], []
    for x_i in x :
        for y_i in y :
            X.append(np.array([x_i,y_i]))
            T.append(gaussian_2d(x_i,y_i,variance,bias,noise_variance))
    X = np.array(X).T
    T = np.array(T)

    X_grid, Y_grid = np.meshgrid(x, y)
    Z_grid = np.reshape(T,(int(np.sqrt(N)),int(np.sqrt(N))))

    return X,T,X_grid,Y_grid,Z_grid

def subsample_function_data(X,T,f) :
    N = np.shape(X)[1]
    random_subs_indices = np.random.choice(N, size=int(N*f), replace=False)
    X_train = X[:,random_subs_indices]
    X_valid = X[:,[i for i in range(N) if i not in random_subs_indices]]
    T_train = T[random_subs_indices]
    T_valid = T[[i for i in range(N) if i not in random_subs_indices]]
    return X_train, T_train, X_valid, T_valid

def gen_mackey_glass_series(beta, tau, gamma, n, N) :
    x = [1.5]
    for t in range(1,N+1) :
        x_tau = x[t-tau] if t >= tau else 0
        x.append((1-gamma)*x[t-1] + (beta*x_tau)/(1 + x_tau**n))
        if (t >= 25) and (t <= N) :
            X_col = np.array([[x[t-25],x[t-20],x[t-15],x[t-10],x[t-5]]])
            T_col = np.array([[x[t]]])
            X = X_col if t == 25 else np.append(X,X_col,axis=0)
            T = T_col if t == 25 else np.append(T,T_col,axis=0)

    return x, X.T, T.T



