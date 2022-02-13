import numpy as np
import matplotlib.pyplot as plt

### Functions ###

## Initialise connectivity matrix
def initialiseOutputDistance (M,N,circular): 
    #MxN grid connected by direct horizontal and vertical neighbours.
    NumberOfNodes = M*N
    connectivity = np.zeros((NumberOfNodes, NumberOfNodes)) #connectivity[i,j] = 1 if i,j are neighbours, 0 otherwise
    for i in range(NumberOfNodes):
        if i > M-1: #then there is a connection with the element above
            connectivity[i][i-M] = 1
        if i + M < M*N: #then there is a connection with the element below
            connectivity[i][i+M] = 1
        if i%M != 0: #then there is a connection with the element to the left
            connectivity[i][i-1] = 1
        if (i+1)%M != 0: #then there is a connection with the element to the right
            connectivity[i][i+1] = 1
    if circular:
        connectivity[0][-1] = 1
        connectivity[-1][0] = 1
    
    #Use connectivity matrix to find distance
    outputDistance = np.zeros((NumberOfNodes,NumberOfNodes))
    for i in range(NumberOfNodes,0,-1): 
        nonZero = np.nonzero(np.linalg.matrix_power(connectivity, i)) #if element nonzero, destination can be reached in i steps so distance is at max i
        outputDistance[nonZero] = i
    np.fill_diagonal(outputDistance,0) #Distance between one and the same node is 0
    return outputDistance

## Initialise weight matrix (a.k.a. positions of the RBF's)
def initialiseWeights (M,N, inputDim):
    numberOfNodes = M*N
    weights = np.random.uniform(0,1,(numberOfNodes,inputDim)) #specified in assignment to initialise with random numbers between 0 and 1.
    return weights

## Train the network
# Neighbourhood helper function
def neighbourhood(closestIndex, outputDistance, epoch,sigma0,tau):
    sigma = sigma0*np.exp(-epoch**2/tau)
    outputNeighbourhoodFunction = np.exp(-outputDistance[closestIndex][:]**2/(2*sigma**2))
    return outputNeighbourhoodFunction

# Main training function
def train (inputData, weights, outputDistance, numberOfEpochs, learningRate, sigma0, tau):
    numberOfDataPoints = len(inputData[:,0])
    for i in range(numberOfEpochs):
        for j in range(numberOfDataPoints):
            distance = np.zeros((len(weights[:,0])))
            for k in range(len(weights[:,0])):
                distance[k] = np.dot((weights[k,:] - inputData[j,:]).T, (weights[k,:] - inputData[j,:])) #actual distances dont matter, so sqrt is skipped
            closestIndex = np.argmin(distance) #find which node is closest
            outputNeighbourhoodFunction = neighbourhood(closestIndex,outputDistance,i,sigma0,tau) #use neighbourhoodfunction
            weights = weights - learningRate*np.array([outputNeighbourhoodFunction]).T*(weights - inputData[j,:]) #update weights dw = eta*h*(x-w)
    return weights

def plot (inputData, weights, M, N, labels):
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title('1D mapping of animals')
    ax.set_xlim([0, M])
    ax.set_ylim([0, N])

    numberOfDataPoints = len(inputData[:,0])
    for j in range(numberOfDataPoints):
        distance = np.zeros((len(weights[:,0])))
        for k in range(len(weights[:,0])):
            distance[k] = np.dot((weights[k,:] - inputData[j,:]).T, (weights[k,:] - inputData[j,:])) #actual distances dont matter, so sqrt is skipped
        closestIndex = np.argmin(distance) #find which node is closest
        yInd = np.floor(closestIndex/M)
        xInd = closestIndex%M
        if N > 1:
            ax.text(xInd + 0.5, yInd + 0.5, labels[j].replace("'","")) #Use this for 2D
        else:
            ax.text(xInd + 0.5, j/(numberOfDataPoints), labels[j].replace("'","")) #Use this for 1D (for readability)

    plt.show()

def plotCities (inputData, weights):
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title('Travelling salesman')

    numberOfDataPoints = len(inputData[:,0])
    distance = np.zeros((len(weights[:,0]),numberOfDataPoints))
    for j in range(numberOfDataPoints):
        for k in range(len(weights[:,0])):
            distance[k,j] = np.dot((weights[k,:] - inputData[j,:]).T, (weights[k,:] - inputData[j,:])) #actual distances dont matter, so sqrt is skipped
    
    closestIndices = np.argmin(distance,axis = 1) #find which node is closest
    for k in range(len(weights[:,0])-1):
        beginPoint = inputData[closestIndices[k]]
        print(beginPoint)
        endPoint = inputData[closestIndices[k+1]]
        ax.plot([beginPoint[0],endPoint[0]],[beginPoint[1],endPoint[1]])

    ax.scatter(inputData[:,0],inputData[:,1])
    plt.show()

### 4.1 Animal task ###
## Process input
animalData = np.genfromtxt('animals.dat', delimiter=',')
inputData = np.reshape(animalData,(32,84)) #32 rows of animals with 84 columns of attributes
animalLabelsFile = open('animalnames.txt','r')
animalLabels = animalLabelsFile.read().splitlines()

## Run
numberOfEpochs = 20
learningRate = 0.2
sigma0 = 20
tau = 3

outputDistance = initialiseOutputDistance(100,1, False) #100x1 grid
inputDim = len(inputData[:][0])
weights = initialiseWeights(100,1,inputDim)
weights = train(inputData,weights,outputDistance,numberOfEpochs,learningRate,sigma0,tau)

## Plot
plot(inputData,weights,100,1,animalLabels)



### 4.2 Travelling Salesman ###
# to lazy how to extract this from the file
inputData = np.array([[0.4000, 0.4439],[0.2439, 0.1463],[0.1707, 0.2293],[0.2293, 0.7610],[0.5171, 0.9414],[0.8732, 0.6536],[0.6878, 0.5219],[0.8488, 0.3609],[0.6683, 0.2536],[0.6195, 0.2634]])

## Run
numberOfEpochs = 50
learningRate = 0.2
sigma0 = 4
tau = 1

outputDistance = initialiseOutputDistance(10,1, True) #10x1 circular grid
inputDim = len(inputData[:][0])
weights = initialiseWeights(10,1,inputDim)
weights = train(inputData,weights,outputDistance,numberOfEpochs,learningRate,sigma0,tau)

## Plot
plotCities(inputData,weights)