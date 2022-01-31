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

# PERCEPTRON RULES
def perceptron_rule_sequential (lr, x, t, W) :
    delta_W = (t-(2*(W.dot(x) > 0) - 1))*lr*x
    return delta_W

# DELTA RULES
def delta_rule_sequential (lr, x, t, W) :
    delta_W = lr*(t - W.dot(x))*x.T
    return delta_W

def delta_rule_batch (lr, X, T, W) :
    e = (T - W.dot(X))
    delta_W = lr*e.dot(X.T)
    return delta_W

# DECISION BOUNDARY
def decision_boundary (W) :
    x2_1 = 100
    x2_2 = -100
    try :
        x1_1 = -(W[0,1]*x2_1 + W[0,2])/W[0,0]
        x1_2 = -(W[0,1]*x2_2 + W[0,2])/W[0,0]
    except : 
        x1_1 = -(W[0,1]*x2_1)/W[0,0]
        x1_2 = -(W[0,1]*x2_2)/W[0,0]
    return (x1_1,x1_2) , (x2_1,x2_2)

# CLASSIFIER 
def classify(W,X,T,show=False) :
    T_guessed = 2*(W.dot(X) > 0) - 1

    # Index -1 class
    columns_bool = T < 0
    T_neg = T[columns_bool]
    T_guessed_neg = T_guessed[:,columns_bool]

    hits_neg = (T_guessed_neg == T_neg).sum()
    fails_neg = np.shape(T_neg)[0] - hits_neg
    accuracy_neg = round(hits_neg*100/np.shape(T_neg)[0],3)

    # Index +1 class
    columns_bool = T > 0
    T_pos = T[columns_bool]
    T_guessed_pos = T_guessed[:,columns_bool]

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

    return T_guessed


def accuracy(W,X,T) :
    T_guessed = 2*(W.dot(X) > 0) - 1
    hits = (T_guessed == T).sum()
    accuracy = hits*100/np.shape(T)[0]
    return accuracy