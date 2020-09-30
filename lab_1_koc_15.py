import numpy as np
import matplotlib.pyplot as plt
# from numba import jit, cuda
import os

# Create a class to build perceptron layers
class SLP:
    def __init__(self, **kwargs):
        # get parameters
        self.output_size = kwargs.get('output_size')
        self.input_size = kwargs.get('input_size')
        self.activation = kwargs.get('activation')
        # Create empty input vector
        self.input_vec = None
        # Initialize weights
        self.weights = None
        self.learning_rate = None
        self.last_layer = True
        self.weight_delta_ = None
        self.momentum_alpha = kwargs.get('momentum')
        self.weight_delta = None
        self.loss = None

    # Update weights using expected input and output
    def calculate_weight_delta(self, delta):
        self.weight_delta = np.dot(delta, self.input_vec.T)

    def differentiation_of_activation(self):
        if self.activation == 'sigmoid':
            return (self._output * (1 - self._output))

    def update_weights(self, desired=None, delta_mtx=None):
        if self.last_layer:
            # I am just adding for sigmoid layer for now, maybe in the following assignments I will add more.
            if self.activation == 'sigmoid':
                error = desired - self._output
                delta = error * self.differentiation_of_activation()
                self.calculate_weight_delta(delta)
                delta = np.dot(self.weights.T, delta)
                self.weights = self.weights + self.learning_rate * self.weight_delta
                if self.momentum_alpha:
                    self.weights += self.momentum_alpha * self.weight_delta_
                self.weight_delta_ = self.weight_delta.copy()
        else:
            if self.activation == 'sigmoid':
                # Just note that the error in the last layer and delta_mtx is subject to the same calculations for
                # backpropagation, that is the idea of backpropagation.
                delta = delta_mtx[0:-1,:] * self.differentiation_of_activation()
                self.calculate_weight_delta(delta)
                delta = np.dot(self.weights.T, delta)
                self.weights = self.weights + self.learning_rate * self.weight_delta
                if self.momentum_alpha:
                    self.weights += self.momentum_alpha * self.weight_delta_
                self.weight_delta_ = self.weight_delta.copy()

        return delta

    def reset(self, learning_rate):
        self.input_vec = np.zeros((self.input_size, 1))
        # Initialize weights
        np.random.seed(35) # Easy one to converge
        # np.random.seed(27) # Hard one to converge
        # np.random.seed(15) # Also another hard one
        self.weights = np.random.uniform(-1, 1, (self.output_size, self.input_size))
        self.weight_delta = np.zeros((self.output_size, self.input_size))
        self.loss = None
        self.weight_delta_ = np.zeros((self.output_size, self.input_size))
        self.learning_rate = learning_rate


    @property
    def output(self):
        value = np.dot(self.weights, self.input_vec)
        if self.activation == 'sigmoid':
            self._output = self.sigmoid(value)
            return self._output.copy()

        elif self.activation == 'linear':
            self._output = value
            return self._output.copy()

    @staticmethod
    def sigmoid(value, a=1):
        value *= a
        return 1 / (1 + np.exp(-value))


class MLP:
    def __init__(self, **kwargs):
        self.network = []
        self.error_tolerance = kwargs.get('error_tolerance')
        self.number_of_epoch = None
        self.learning_rate = kwargs.get('learning_rate')
        self.max_epoch = 1e6
        self.name = kwargs.get('name')

    def add(self, single_layer):
        if not self.network:
            self.network.append(single_layer)
            self.network_size = len(self.network)

        else:
            self.network.append(single_layer)
            # Set the previous last layer to False
            self.network[-2].last_layer = False
            self.network_size = len(self.network)

        single_layer.input_size += 1 # To add the biases

    def backpropagation(self, desired_output):
        i = self.network_size - 1
        while i >= 0:
            if i == self.network_size - 1:
                delta = self.network[i].update_weights(desired=desired_output)
            else:
                delta = self.network[i].update_weights(delta_mtx=delta)

            i -= 1

    def output(self, input_vec):
        for count, slp in enumerate(self.network):
            if count == 0:
                slp.input_vec = np.append(input_vec.flatten(), [1], axis=0).reshape(-1,1) # Add the bias terms
            else:
                slp.input_vec = np.append(output.flatten(), [1], axis=0).reshape(-1,1) # Add the bias terms
            output = slp.output
        return output.copy()

    def sample_data(self, dataset):
        index = np.random.choice(len(dataset), len(dataset), replace=False)
        return index

    def train(self, dataset):
        # Initial call
        data_trained = np.zeros(len(dataset), dtype=np.bool)
        self.number_of_epoch = 0
        for slp in self.network:
            slp.reset(self.learning_rate)
        failed = False
        loss_vs_epoch = []

        while np.sum(data_trained) < 16:
            if self.number_of_epoch > self.max_epoch:
                failed = True
                break
            index = self.sample_data(dataset)
            # Make back propagation individually
            error_avg = np.zeros(len(dataset))
            total_loss = 0
            for idx in index:
                data = dataset[idx].copy()
                desired_output = dataset_output(data)
                output = self.output(data)
                error = desired_output - output
                abs_error = abs(error)
                total_loss += 0.5 * np.dot(error.T, error).item()
                error_avg[idx] = abs_error
                if abs_error < self.error_tolerance:
                    data_trained[idx] = True
                else:
                    data_trained[idx] = False
                # Update the weights whenever the desired state is not reached
                self.backpropagation(desired_output=desired_output)
            average_error = np.mean(error_avg)
            loss_vs_epoch.append(total_loss)
            self.number_of_epoch += 1
            # if self.number_of_epoch % 1000 == 0:
            #     print(
            #         # f"Selected input: {data.reshape((-1,))}, Output: {output.item()}, "
            #         # f"Desired Output: {desired_output}, Error: {error.item()}, "
            #         f"Number of epochs {self.number_of_epoch}, "
            #         f"trained number elements {np.sum(data_trained)}, "
            #         f"learning rate: {self.learning_rate}, "
            #         f"Total loss: {total_loss}, "
            #         f"Average Error: {average_error}\n"
            #         f"Inivididual errors: {error_avg}\n"
            #         f"{data_trained}"
            #     )

            # print(data_trained)
            # self.number_of_epoch += 1
        for i in range(self.network_size):
            with open(f"saved_weights/{self.name}_learning_rate_{self.learning_rate:.2f}_weights{i:d}.npy", 'wb') as f:
                np.save(f, self.network[i].weights)
        reason_string = "Failed" * failed + "Completed" * (1 - failed)
        print(f"Training Done for learning rate {self.learning_rate} | reason: {reason_string}")
        return self.number_of_epoch, loss_vs_epoch, reason_string, data_trained


def dataset_output(data):
    return 1 if np.sum(data) % 2 == 1 else 0

def main():
    # Save figures and weights
    if not os.path.exists('saved_weights'):
        os.makedirs('saved_weights')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    # Create the network
    neural_network = MLP(error_tolerance=0.05, learning_rate=0.05, name='network1')
    neural_network.add(SLP(input_size=4, output_size=4, activation='sigmoid'))
    neural_network.add(SLP(input_size=4, output_size=1, activation='sigmoid'))

    # Create a dictionary for input-output
    dataset = []
    # Create the dataset
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for iterator in range(2):
                    dataset.append(np.array([i, j, k, iterator]).reshape(4, 1))  # Create bits

    # Keep a list to visualize the training results
    list_epochs = []
    loss_vs_epoch_list = []

    for i in range(10):
        if i != 0:
            neural_network.learning_rate = 0.05 + i * 0.05
        epoch, loss_vs_epoch, reason_string, data_train_result = neural_network.train(dataset)
        loss_vs_epoch_list.append(loss_vs_epoch)
        list_epochs.append(epoch)
        plt.plot(loss_vs_epoch, color='b')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title(f"Number of Epochs against Loss, Number of Learned Inputs: {np.sum(data_train_result)}")
        # plt.show()
        plt.savefig(
            f"figures/{neural_network.name}_loss_learning_rate{neural_network.learning_rate:.2f}.png",
            bbox_inches='tight')
        plt.close()

    plt.plot([x * 0.05 for x in range(1, 10 + 1)], list_epochs, color='b')
    plt.xlabel('Learning rate')
    plt.ylabel('Epoch Time')
    plt.title(f"Number of Epochs against learning rates")
    plt.savefig(
        f"figures/{neural_network.name}_learning_rate_vs_epoch_time.png",
        bbox_inches='tight')
    plt.close()

    for count, loss in enumerate(loss_vs_epoch_list):
        learning_rate = 0.05 + count * 0.05
        plt.plot(loss,label=f"Learning_rate{learning_rate:.2f}")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title(f"Number of Epochs against Loss, Number of Learned Inputs: {np.sum(data_train_result)}")
    plt.savefig(
        f"figures/{neural_network.name}_loss_combined.png",
        bbox_inches='tight')
    plt.close()

    # Keep lists to visualize the training results
    list_epochs = []
    loss_vs_epoch_list = []

    # Create the second network momentum now
    neural_network = MLP(error_tolerance=0.05, learning_rate=0.05, name='network2')
    neural_network.add(SLP(input_size=4, output_size=4, activation='sigmoid', momentum=0.9))
    neural_network.add(SLP(input_size=4, output_size=1, activation='sigmoid', momentum=0.9))
    for i in range(10):
        if i != 0:
            neural_network.learning_rate = 0.05 + i * 0.05
        epoch, loss_vs_epoch, reason_string, data_train_result = neural_network.train(dataset)
        list_epochs.append(epoch)
        loss_vs_epoch_list.append(loss_vs_epoch)
        plt.plot(loss_vs_epoch, color='b')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title(f"Number of Epochs against Loss, Number of Learned Inputs: {np.sum(data_train_result)}")
        plt.savefig(
            f"figures/{neural_network.name}_loss_learning_rate{neural_network.learning_rate:.2f}.png",
            bbox_inches='tight')
        plt.close()

    for count, loss in enumerate(loss_vs_epoch_list):
        learning_rate = 0.05 + count * 0.05
        plt.plot(loss,label=f"Learning_rate{learning_rate:.2f}")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title(f"Number of Epochs against Loss, Number of Learned Inputs: {np.sum(data_train_result)}")
    plt.savefig(
        f"figures/{neural_network.name}_loss_combined.png",
        bbox_inches='tight')
    plt.close()

    plt.plot([x * 0.05 for x in range(1, 10 + 1)], list_epochs, color='b')
    plt.xlabel('Learning rate')
    plt.ylabel('Epoch Time')
    plt.title(f"Number of Epochs against learning rates")
    plt.savefig(
        f"figures/{neural_network.name}_learning_rate_vs_epoch_time.png",
        bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    main()

    # Later, if you want to use weights to do something else, use the saved weights
    # with open('test.npy', 'rb') as f:
    #
    #     a = np.load(f)
    #
    #     b = np.load(f)
