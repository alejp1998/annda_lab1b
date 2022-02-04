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
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import trange

# Auxiliary variables
colors = ['#1E90FF','#FF69B4']

# NEURAL NETWORK CLASS
class NeuralNetwork(nn.Module) :
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size) :
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Sigmoid(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Sigmoid(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y
        
    def initialize_weights(self):
        for layer in self.Sequential.children():
            # Initiliazing with Xavier Normal
            nn.init.xavier_normal_(layer.weight)

    def train(self,lr,n_epochs,loss_fn,X_train_ten,T_train_ten,X_valid_ten,T_valid_ten) :
        mses, mses_valid = [], []
        optim = torch.optim.Adam(self.parameters(),lr=lr)
        for epoch in range(n_epochs):
            #forward feed
            T_pred_train_ten = self.forward(X_train_ten.requires_grad_())
            #calculate the loss
            loss = loss_fn(T_pred_train_ten, T_train_ten)
            mses.append(loss.item())

            #forward validation
            T_pred_valid_ten = self.forward(X_valid_ten)
            #calculate the validation loss
            loss_valid = loss_fn(T_pred_valid_ten, T_valid_ten)
            mses_valid.append(loss_valid.item())
            
            #backward propagation: calculate gradients
            loss.backward()
            #update the weights
            optim.step()
            #clear out the gradients from the last step loss.backward()
            optim.zero_grad()
        return mses, mses_valid, T_pred_train_ten, T_pred_valid_ten

    def show(self) :
        print('--------------------------------------------------\n')
        print('Neural Network Description')
        print('Input Layer Neurons = ',self.input_size)
        print('Hidden Layer 1 Neurons = ',self.hidden_size_1)
        print('Hidden Layer 2 Neurons = ',self.hidden_size_2)
        print('Output Layer Neurons = ',self.output_size,'\n')
        print(self)
        print('\n--------------------------------------------------\n')


# HELPER FUNCTIONS
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

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)

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

def plot_series(series,series_pred,title) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title(title)
    ax.plot(series,label='True')
    ax.plot(series_pred,label='Predicted')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    plt.show()

def plot_archs_comparison(combs,combs_mean_mses,combs_mean_mses_valid,combs_gen_error) :
    plotdata = pd.DataFrame({
        'Training':combs_mean_mses,
        'Validation':combs_mean_mses_valid,
        'Gen. Error':combs_gen_error
        }, 
        index=combs
    )
    plotdata.plot(kind='bar', stacked=True)

def compare_archs(lr,n_epochs,loss_fn,X_train_ten,T_train_ten,X_valid_ten,T_valid_ten,input_size,output_size,hidden_sizes_1,hidden_sizes_2) :
    combs, combs_mean_mses, combs_mean_mses_valid = [], [], []
    # For each of the combinations
    for hidden_size_1 in hidden_sizes_1 :
        for hidden_size_2 in hidden_sizes_2 :
            # Train each combination 10 times and store mean MSE
            comb_mses = np.zeros(10)
            comb_mses_valid = np.zeros(10)
            for i in range(10) :
                model = NeuralNetwork(input_size, hidden_size_1, hidden_size_2, output_size)
                model.apply(init_weights)
                mses, mses_valid, T_pred_train_ten, T_pred_valid_ten = model.train(lr,n_epochs,loss_fn,X_train_ten,T_train_ten,X_valid_ten,T_valid_ten)
                comb_mses[i] = mses[-1]
                comb_mses_valid[i] = mses_valid[-1]

            # Compute mean MSE
            combs.append(str(hidden_size_1)+'+'+str(hidden_size_2))
            combs_mean_mses.append(comb_mses.sum()/10)
            combs_mean_mses_valid.append(comb_mses_valid.sum()/10)

    return combs, combs_mean_mses, combs_mean_mses_valid