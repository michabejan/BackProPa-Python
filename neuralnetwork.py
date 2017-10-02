import numpy as np

class NeuralNetwork:

    def __init__(self, num_layers, num_bias_weights):
        self.hidden_layers = []
        self.inputs = []
        self.output = []
        self.targets = []

        self.hidden_layers[0] = Layer(True,num_bias_weights,self.inputs)
        for i in range(1, num_layers-1):
            self.hidden_layers[i] = Layer(self,num_bias_weights,self.hidden_layers[i-1].output_weights )

    def sigmoid(net):
        return 1 / 1 + np.exp(net);

    def forward(self):
        for i in range(len(self.hidden_layers - 1)):
            for j in range(len(self.hidden_layers[i].values - 1)):
                for k in range(len(self.hidden_layers[i]).input_weights - 1):
                    self.hiddenlayers[i].value[j] += self.hidden_layers[k].input_weights * self.hidden_layers[i-1].value
                self.hiddenlayers[i].value[j] += self.hiddenlayers[i].biases[j]
                self.hiddenlayers[i].value[j] = self.sigmoid(self.hiddenlayers[i].value[j])

    def outputError(self):
        self.errors = []
        temp = len(self.hidden_layers) - 1
        for i in range(len(self.outputs)):
            for j in range(len(self.hidden_layers[temp])):
                self.output.values[i] += self.hidden_layers[temp].output_weights[j] + self.hidden_layers[temp].value[j]
            self.output.values[i] += self.output.biases[i]
            self.output.values[i] = self.sigmoid(self.output.values[i])

        temp = 0
        for i in range(len(self.output.values)):
            self.output.errors[i] = 1/2 * (self.output.target[i] - self.output.values[i]).exp(2)
        for i in range(len(self.output.errors)):
            temp += self.output.errors[i]
        return temp

    def backpropagation(self, learning_rate):
        index = len(self.hidden_layers)
        delta_errors = len(self.hidden_layers)
        delta_net = len(self.hidden_layers)

        for i in range(len(self.outputs)):
            delta_errors = - (self.output.target[i] - self.output.value[i])
        for i in range(len(self.outputs)):
            delta_net = self.output.value[i] * (1 - self.output.value[i])
            self.hidden_layers[index-1].output_weights[i] = self.hidden_layers[index- 1].output_weights[i] - learning_rate * delta_errors[i] * delta_net * self.output.value[i]

        for k in range(len(self.hidden_layers)):
            for i in range(len(self.self.hidden_layers[index-k-1])):
                delta_errors = - (self.hidden_layers[index-k-i-1].output_weights[i] - self.hidden_layers[index-k-i-1].values[i])
            for i in range(len(self.outputs)):
                delta_net = self.output.value[i] * (1 - self.output.value[i])
                self.hidden_layers[index-k-1].output_weights[i] = self.hidden_layers[index-k-1].output_weights[i] - learning_rate * delta_errors[i] * delta_net * self.output.value[i]


class Layer:

    def _init_(self, num_bias_weights,input_weights):
            self.values = []
            self.biases = []
            for i in range(num_bias_weights):
                self.biases.append(np.random(0,1))
            for i in range(num_bias_weights):
                self.output_weights.append(np.random(0,1))
            self.input_weights = input_weights

class Output:

    def __init__(self):
        self.values = []
        self.biases = []
        self.errors = []
        self.target = []
