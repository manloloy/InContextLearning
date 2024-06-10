import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from FunctionDataset import FunctionDataset

class LSGDBaseline:
    def __init__(self, input_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = np.zeros(input_dim)
    
    def fit(self, X, y):
        """
        Fit the model by performing gradient descent updates for each in-context example.
        X: Input data matrix (num_samples, input_dim)
        y: Target values (num_samples,)
        """
        self.weights = np.zeros(self.input_dim)  # Reset weights
        num_examples = X.shape[0]  # Calculate the number of in-context examples
        for i in range(num_examples):
            self.update_weights(X[i], y[i])
        
    def update_weights(self, X, y):
        """
        Update weights using a single gradient descent step.
        X: Single input data vector (input_dim,)
        y: Single target value
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        prediction = np.dot(X, self.weights)
        error = prediction - y
        gradient = 2 * X * error
        self.weights -= self.learning_rate * gradient
    
    def predict(self, X):
        """
        Predict using the learned weights.
        X: Input data matrix (num_samples, input_dim)
        Returns: Predictions (num_samples,)
        """
        return np.dot(X, self.weights)
    
    def evaluate(self, X, y):
        """
        Evaluate the model using mean squared error.
        X: Input data matrix (num_samples, input_dim)
        y: Target values (num_samples,)
        Returns: Mean squared error
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

def generate_linear_prompts(num_samples, input_dim, noise_std=0.1):
    dataset = FunctionDataset(num_samples=num_samples, input_dim=input_dim, function_class='linear', noise_std=noise_std)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    for data in dataloader:
        inputs, targets = data
        return inputs.numpy(), targets.numpy()

if __name__ == "__main__":
    # Generate data
    num_samples = 40
    input_dim = 5
    X, y = generate_linear_prompts(num_samples, input_dim)
    
    # Train and evaluate the model with different numbers of in-context examples
    learning_rate = 0.05
    mse_list = []

    for in_context_examples in range(1, num_samples + 1):
        model = LSGDBaseline(input_dim,learning_rate)
        model.fit(X[:in_context_examples], y[:in_context_examples])
        mse = model.evaluate(X, y)
        mse_list.append(mse)

    # Plotting squared error vs in-context examples
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_samples + 1), mse_list, marker=None, linestyle='-', color='b')
    plt.xlabel('Number of In-Context Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error vs. Number of In-Context Examples')
    plt.grid(True)
    plt.show()
