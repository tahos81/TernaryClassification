import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x     #input from the dataset
        self.weights1   = np.random.rand(self.input.shape[1],6)     #initially randomize the weights
        self.weights2   = np.random.rand(6,3)                 
        self.y          = y     #expected output
        self.output     = np.zeros(self.y.shape)    #actual output
        self.bias1      = np.random.randn(6)    #initially randomize the biases
        self.bias2      = np.random.randn(3)
        self.lr         = 0.01      #learning rate
    
    #forward stage of neural network
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
        self.output = softmax(np.dot(self.layer1, self.weights2) + self.bias2)

    #backpropogation
    def backprop(self):
        delta2 = softmax_derivative(self.output, self.y)
        delta1 = np.dot(delta2, self.weights2.T) * self.layer1 * (1 - self.layer1) 
        
        d_weights1 = np.dot(self.input.T, delta1)
        d_weights2 = np.dot(self.layer1.T, delta2)

        #update weights and biases
        self.weights1 -= self.lr * d_weights1
        self.bias1    -= self.lr * (delta1).sum(axis=0)

        self.weights2 -= self.lr * d_weights2
        self.bias2    -= self.lr * (delta2).sum(axis=0)

def main():
    np.random.seed(4)
    inputs, labels = load_data() #loading data from data set
    train = NeuralNetwork(inputs, labels) #creating a neural network instance
    for epoch in range(10000):
       train.feedforward()
       train.backprop()
       print(cross_entropy(train.output, train.y.argmax(axis=1))) #loss
    print(train.output.argmax(axis=1)) #predictions of the ai
    
    '''
    this part is for testing against the data in the test_set. I couldnt find any other idea than printing the output and checking with my eyes against the objects.png in the test set.

    test_layer1 = sigmoid(np.dot(load_test('case_study1_data/Test/1/data.txt'), train.weights1) + train.bias1)
    test_output = softmax(np.dot(test_layer1, train.weights2) + train.bias2)
    print(test_output.argmax(axis=1))
    '''


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1).reshape(x.shape[0],1)

def softmax_derivative(x, y):
    return (x-y)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
        return sigmoid(x)*(1-sigmoid(x))

def cross_entropy(X,y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def load_data():
    '''
    function for loading data from train data set
    '''
    array1 = np.loadtxt('case_study1_data/Train/1/data.txt', delimiter=',')   
    array2 = np.loadtxt('case_study1_data/Train/2/data.txt', delimiter=',')
    array3 = np.loadtxt('case_study1_data/Train/3/data.txt', delimiter=',')
    array4 = np.loadtxt('case_study1_data/Train/4/data.txt', delimiter=',')
    array5 = np.loadtxt('case_study1_data/Train/5/data.txt', delimiter=',')
    array6 = np.loadtxt('case_study1_data/Train/6/data.txt', delimiter=',')
    array7 = np.loadtxt('case_study1_data/Train/7/data.txt', delimiter=',')
    array8 = np.loadtxt('case_study1_data/Train/8/data.txt', delimiter=',')
    array9 = np.loadtxt('case_study1_data/Train/9/data.txt', delimiter=',')

    input_array = np.concatenate((array1,array2,array3,array4,array5,array6,array7,array8,array9))
    
    one = [1, 0, 0]
    two = [0, 1, 0]
    three = [0, 0, 1]

    labels = []

    for _ in range(42):
        labels.append(one)
    for _ in range(27):
        labels.append(two)
    for _ in range(30):
        labels.append(three)
    
    labels = np.array(labels)
    
    labels.reshape(99,3)
    
    return input_array, labels

def load_test(path):
    '''
    function for loading data from test dataset
    '''
    input_array = np.loadtxt(path, delimiter=',')
    return input_array

    


main()