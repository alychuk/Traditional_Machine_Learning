import numpy as np
import math
from scipy.special import softmax
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt


# Helper function to evaluate the total loss on the dataset
# model is the current version of the model{’W1’:W1,’b1’:b1,’W2 ’:W2,’ b2’: b2’} It’s a dictionary
# X is all the training data
# y is the training labels
def calculate_loss(model, X, y):
    loss = 0
    for (sample,label) in zip(X,y):
        yhat = predict(model, sample)
        if label != yhat:
            loss += label * math.log(yhat)
    return (-(loss / y.shape[0]))


def predict(model, x):
    prediction = {1,1}
    a = np.dot(x, model['W1']) + model['b1']
    h = np.tanh(a)
    z = np.dot(h,model['W2']) + model['b2']
    y_hat = softmax(z)
    prediction = np.argmax(y_hat, axis=1)
    prediction = np.array([0,1]) if prediction else np.array([1,0])
    print(prediction)
    return prediction


def build_model(X, y, nn_hdim, num_passes=2000, print_loss=False):
    model = {'W1': np.random.rand(2, nn_hdim), 'b1': np.random.rand(1,nn_hdim),
                'W2': np.random.rand(nn_hdim, 2), 'b2': np.random.rand(1,2)}
    learning_rate = .1
    for i in range(0, num_passes):
        for (sample,label) in zip(X,y):
            a = np.dot(sample, model['W1']) + model['b1']
            L_yhat = predict(model, sample) - (np.array([0,1]) if label else np.array([1,0]))
            L_a = np.dot((1-(np.square( np.tanh(a) ) )), np.dot(L_yhat, model['W2'].T))
            L_W2 = np.tanh(a).T * L_yhat
            L_b2 = L_yhat
            L_W1 = sample.reshape(1,2).T * L_a
            L_b1 = L_a
            model['W1'] = model['W1'] - learning_rate * L_W1
            model['W2'] = model['W2'] - learning_rate * L_W2
            model['b1'] = model['b1'] - learning_rate * L_b1
            model['b2'] = model['b2'] - learning_rate * L_b2
        #if (i % 1000 == 0) and print_loss:
        #    print(calculate_loss(model, X, y))


def plot_decision_boundary(pred_func, X, y):
    # set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# Generate a dataset and plot it
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
# Generate ouputs with the following code
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
