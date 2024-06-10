import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from FunctionDataset import FunctionDataset

class AveragingBaseline:
    def __init__(self):
        self.mean = 0.0
        
    def fit(self, X, y):
        """
        Fit the model by computing the mean of the observed values.
        X: Input data matrix (num_samples, input_dim) - Dummy parameter
        y: Target values (num_samples,)
        """
        self.mean = np.mean(y)
    
    def predict(self, X):
        """
        Predict using the mean of the observed values.
        X: Input data matrix (num_samples, input_dim)
        Returns: Predictions (num_samples,)
        """
        return np.full(X.shape[0], self.mean)
    
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
    num_samples = 50
    input_dim = 5
    X, y = generate_linear_prompts(num_samples, input_dim)

    # Train and evaluate the model with different numbers of in-context examples
    mse_list = []

    for in_context_examples in range(1, num_samples + 1):
        model = AveragingBaseline()
        model.fit(X, y[:in_context_examples])
        mse = model.evaluate(X, y)
        mse_list.append(mse)

    # Plotting squared error vs in-context examples
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_samples + 1), mse_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of In-Context Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error vs. Number of In-Context Examples (Averaging Baseline)')
    plt.grid(True)
    plt.show()
