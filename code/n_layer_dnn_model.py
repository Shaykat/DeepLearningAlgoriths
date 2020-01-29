from numpy import np
from .utility import *
import matplotlib.pyplot as plt
from .neural_network_helper_methods import *


def L_layer_model(X, Y, layers_dims, learning_rate=0.0049, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# Get Data
train_x, test_x, train_y, test_y = get_pre_processed_data()
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("train_y's shape: " + str(train_y.shape))
print("test_y's shape: " + str(test_y.shape))

# CONSTANTS DEFINING THE MODEL ####
n_x = train_x.shape[0]
n_y = 1
layers_dims = (n_x, 20, 7, 5, n_y)
parameters = L_layer_model(train_x, train_y, layers_dims=layers_dims, num_iterations=2500, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
