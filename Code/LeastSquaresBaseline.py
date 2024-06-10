import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from FunctionDataset import FunctionDataset

class LeastSquaresBaseline:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
    
    def fit(self, X, y):
        """
        Fit the model using the Moore-Penrose pseudoinverse for least squares regression.
        X: Input data matrix (num_samples, input_dim)
        y: Target values (num_samples,)
        """
        # Compute the Moore-Penrose pseudoinverse of X
        X_pseudo_inverse = np.linalg.pinv(X)
        
        # Compute the weights
        self.weights = X_pseudo_inverse @ y
    
    def predict(self, X):
        """
        Predict using the learned weights.
        X: Input data matrix (num_samples, input_dim)
        Returns: Predictions (num_samples,)
        """
        return X @ self.weights
    
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
    input_dim = 20
    X, y = generate_linear_prompts(num_samples, input_dim)
    
    # Train and evaluate the model with different numbers of in-context examples
    mse_list = []

    for in_context_examples in range(1, num_samples + 1):
        model = LeastSquaresBaseline(input_dim)
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
