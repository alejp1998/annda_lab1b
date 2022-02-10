import numpy as np

def RBF (x, mu, sigma): #Kind of redundant, if input is defined correctly RBF_multdim can be used for 1D as well
    try: 
        phi = np.zeros((len(x),len(mu))) #For some reason len of 1x1 vector throws an error
    except:
        phi = np.zeros((1,len(mu))) #If single data points are fed

    for i in range(len(mu)):
        phi[:,i] = np.exp(-((x-mu[i])**2)/(2*sigma[i]**2))
    return phi

def RBF_multdim (x,mu,sigma):
    phi = np.zeros((x.shape[-1],mu.shape[-1])) #pre-allocate
    for i in range(mu.shape[-1]): #for all RBF's
        mu_col = np.array([mu[:,i]]).T #Needed because even though mu[:,i] is a column vector, it is automatically transformed to row vector by numpy
        phi[:,i] = np.exp(-(np.linalg.norm(x-mu_col)**2)/(2*sigma[:,i]**2))
    return phi

def delta_rule (x,y_train,mu,sigma,lr,epochs):
    w = np.random.normal(0,1,mu.shape[-1])
    for i in range(epochs):
        ind = np.random.permutation(x.shape[-1]) #shuffle data order
        x =  x[ind]
        y_train = y_train[ind]
        for k in range(x.shape[-1]):
            phi = RBF(x[k],mu,sigma)
            phi = phi[0,:]
            w = w + lr*(y_train[k] - np.sum(w*phi))*phi
    return w

def least_squares (x,y_train,mu,sigma):
    phi = RBF(x,mu,sigma)
    w = np.transpose(phi).dot(y_train).dot(np.linalg.inv(np.dot(np.transpose(phi),phi)))
    return w

def forward (x,w,mu,sigma):
    y = np.sum(w*RBF(x,mu,sigma),-1)
    return y

def CL_initialisation (x,dim,N,s,epochs,n_winners,lr):
    # x has dim rows and N_datapoints columns 
    # mu has dim rows and N_RBF (N) columns
    # sigma has 1 row with N_RBF (N) columns (std equal in all directions)
    if dim == 1:
        mu = np.random.uniform(x.min(-1),x.max(-1),N) #Arrays act weird in python
        sigma = s*np.ones(N)
    else:
        #mu = np.random.uniform(x.min(-1),x.max(-1),size=(dim,N)) #initialise RBF's randomly uniform
        np.random.uniform(np.array([x.min(-1)]).T,np.array([x.max(-1)]).T,(2,30))
        sigma = s*np.ones((1,N))

    for k in range(epochs):
        x = x[np.random.permutation(x.shape[-1])] #shuffle data order
        for i in range(x.shape[-1]):
            phi = RBF(x[i],mu,sigma)
            for j in range(1,n_winners+1):
                biggest = max(phi)
                biggest_ind = np.where(phi == biggest)
                if dim == 1:
                    mu[biggest_ind[0]] = mu[biggest_ind[0]] + lr/j*(x[i] - mu[biggest_ind[0]])
                else:
                    mu[:,biggest_ind][0] = mu[:,biggest_ind[0]] + lr/j*(x[:,i] - mu[:,biggest_ind[0]]) #move RBF closer to data point
                phi[biggest_ind[0]] = -1
    return mu, sigma