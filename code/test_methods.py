from .neural_network_helper_methods import *
from .test_cases import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")


parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))


A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))


X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))


dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))


dAL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print("sigmoid:")
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print("relu:")
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))


AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)


grads = L_model_backward_test_case
print("dW1 = "+ str(grads["dW1"]))
print("db1 = "+ str(grads["db1"]))
print("dA0 = "+ str(grads["dA1"]))


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print("W1 = "+ str(parameters["W1"]))
print("b1 = "+ str(parameters["b1"]))
print("W2 = "+ str(parameters["W2"]))
print("b2 = "+ str(parameters["b2"]))
