my neural network has one hidden layer with 6 nodes, its input layer has 6 nodes(number of variables in dataset) and its output layer has 3 nodes(number of classes).
I used cross entropy loss for my loss function, my first activation function is sigmoid and second activation function is softmax(since its a multiclassed problem).
after 10000 epochs it successfully identifies all objects in the train dataset with a cross-entropy loss of 0.555

python version 3.8.5
numpy version 1.17.4