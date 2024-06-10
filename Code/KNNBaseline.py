import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from FunctionDataset import FunctionDataset

class KNNBaseline:
    def __init__(self, k=3):
        self.k = k
        self.in_context_X = None
        self.in_context_y = None
    
    def fit(self, X, y):
        """
        Fit the model by storing the in-context examples.
        X: Input data matrix (num_samples, input_dim)
        y: Target values (num_samples,)
        """
        self.in_context_X = X
        self.in_context_y = y
    
    def predict(self, X):
        """
        Predict using the mean of the k-nearest in-context examples.
        X: Input data matrix (num_samples, input_dim)
        Returns: Predictions (num_samples,)
        """
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.in_context_X - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_values = self.in_context_y[nearest_indices]
            predictions.append(np.mean(nearest_values))
        return np.array(predictions)
    
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
    k = 3
    X, y = generate_linear_prompts(num_samples, input_dim)

    # Train and evaluate the model with different numbers of in-context examples
    mse_list = []

    for in_context_examples in range(1, num_samples + 1):
        model = KNNBaseline(k=k)
        model.fit(X[:in_context_examples], y[:in_context_examples])
        mse = model.evaluate(X, y)
        mse_list.append(mse)

    # Plotting squared error vs in-context examples
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_samples + 1), mse_list, marker=None, linestyle='-', color='b')
    plt.xlabel('Number of In-Context Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error vs. Number of In-Context Examples (K-Nearest Neighbors Baseline)')
    plt.grid(True)
    plt.show()
