##
# Author: Austin Irvine
# License: WTFPL License
# Purpose: Create a basic neural network to see how it works
##

# Tutorial Followed: https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

from numpy import exp, array, random, dot

class NeuralNet():
    def __init__(self):
        # Seed the random number gen, so it generates the same nubers
        # every time the program runs
        random.seed(1)

        #Modeling a neuron, 3 input connections, 1 output
        # Assign weights to 3 x 1 matrix, with values from -1 to 1
        # and mean 0
        self.weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations) :
        for iteration in range(iterations):
            # pass traingin set through neural net (one neuron)
            predicted_output = self.think(inputs)

            # calculate error (difference btwn desired output and
            # and the predicted output)
            error = outputs - predicted_output

            # Multiply the error by the input and again by the gradient
            # of the sigmoid curve.
            # Meaning: less confident weights are adjusted more
            # Meaning: mean inputs, which are zero, do not cause weight changes
            adjustment = dot(inputs.T, error * self.__sigmoid_derivative(predicted_output))

            # Adjust Weights
            self.weights += adjustment
    def think(self, inputs):
        #pass input through a single neuron
        return self.__sigmoid(dot(inputs, self.weights))

if __name__ == "__main__":

    # start neural network
    neural_net = NeuralNet()

    print("Assign Random Weights to Synapses")
    print(neural_net.weights)

    # create training sets as matrices
    # and 1 as the expected output
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    # run training algorithm with 10,000 trials
    neural_net.train(training_set_inputs, training_set_outputs, 100000)

    print("New synaptic weights after training: ")
    print(neural_net.weights)

    # Test the neural network with a new situation
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_net.think(array([1,0,0])))
