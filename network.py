import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        Initialize the neural network with the given parameters.

        Args:
            input_size (int): Number of input neurons.
            hidden_size (int): Number of hidden neurons.
            output_size (int): Number of output neurons.
            learning_rate (float): Learning rate for weight updates.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        """
        Apply the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying sigmoid derivative.
        """
        return x * (1 - x)

    def predict(self, input_vector):
        """
        Make a prediction for the given input vector.

        Args:
            input_vector (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Predicted output.
        """
        hidden_layer_input = np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self._sigmoid(output_layer_input)

        return output_layer_output

    def _compute_gradients(self, input_vector, target):
        """
        Compute the gradients for the weights and biases.

        Args:
            input_vector (numpy.ndarray): Input vector.
            target (numpy.ndarray): Target vector.

        Returns:
            tuple: Gradients for weights and biases.
        """
        hidden_layer_input = np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self._sigmoid(output_layer_input)

        output_error = target - output_layer_output
        output_delta = output_error * self._sigmoid_derivative(output_layer_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(hidden_layer_output)

        derror_dweights_hidden_output = hidden_layer_output.T.dot(output_delta)
        derror_dbias_output = np.sum(output_delta, axis=0, keepdims=True)

        derror_dweights_input_hidden = input_vector.T.dot(hidden_delta)
        derror_dbias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)

        return derror_dweights_input_hidden, derror_dbias_hidden, derror_dweights_hidden_output, derror_dbias_output

    def _update_parameters(self, derror_dweights_input_hidden, derror_dbias_hidden, derror_dweights_hidden_output, derror_dbias_output):
        """
        Update the weights and biases using the computed gradients.

        Args:
            derror_dweights_input_hidden (numpy.ndarray): Gradients for input-hidden weights.
            derror_dbias_hidden (numpy.ndarray): Gradients for hidden biases.
            derror_dweights_hidden_output (numpy.ndarray): Gradients for hidden-output weights.
            derror_dbias_output (numpy.ndarray): Gradients for output biases.
        """
        self.weights_input_hidden += self.learning_rate * derror_dweights_input_hidden
        self.bias_hidden += self.learning_rate * derror_dbias_hidden
        self.weights_hidden_output += self.learning_rate * derror_dweights_hidden_output
        self.bias_output += self.learning_rate * derror_dbias_output

    def _compute_error(self, prediction, target):
        """
        Compute the error between the prediction and the target.

        Args:
            prediction (numpy.ndarray): Predicted output.
            target (numpy.ndarray): Target output.

        Returns:
            float: Computed error.
        """
        return np.mean(np.square(target - prediction))

    def train(self, input_vectors, targets, iterations):
        """
        Train the neural network using the provided input vectors and targets.

        Args:
            input_vectors (numpy.ndarray): Input vectors.
            targets (numpy.ndarray): Target vectors.
            iterations (int): Number of iterations to train the network.

        Returns:
            list: List of cumulative errors for each iteration.
        """
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dweights_input_hidden, derror_dbias_hidden, derror_dweights_hidden_output, derror_dbias_output = self._compute_gradients(input_vector, target)
            self._update_parameters(derror_dweights_input_hidden, derror_dbias_hidden, derror_dweights_hidden_output, derror_dbias_output)

            # Measure the cumulative error for all the instances every 100 iterations
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for i in range(len(input_vectors)):
                    input_vector = input_vectors[i]
                    target = targets[i]
                    prediction = self.predict(input_vector)
                    error = self._compute_error(prediction, target)
                    cumulative_error += error

                cumulative_errors.append(cumulative_error)

        return cumulative_errors