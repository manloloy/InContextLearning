import torch
from torch.utils.data import Dataset
from FunctionDataset import FunctionDataset

class TransformerTrainDataset(Dataset):
    def __init__(self, num_samples, input_dim, max_context_size, function_class='linear', noise_std=0.1):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.max_context_size = max_context_size
        self.noise_std = noise_std
        self.dimension = input_dim
        self.function_class = function_class

    def generate_new_function(self):
        return FunctionDataset(num_samples=self.max_context_size + 1, input_dim=self.input_dim, function_class=self.function_class, noise_std=self.noise_std)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        function_dataset = self.generate_new_function()

        inputs = []
        targets = []

        # Collect inputs and targets
        for i in range(self.max_context_size):
            x, y = function_dataset[i]
            y_vector = torch.cat((torch.tensor([y]), torch.zeros(self.dimension - 1)))
            inputs.extend([x.tolist(), y_vector.tolist()])

        # Add the next input (x_{n+1}) with zero as the function value
        x_next, _ = function_dataset[self.max_context_size]
        inputs.append(x_next.tolist())
        inputs.append([0.0] * self.dimension)

        # Convert lists to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, self.dimension)

        # Create targets with zero padding
        targets = torch.zeros((self.max_context_size * 2 + 1, self.dimension), dtype=torch.float32)

        # Place the actual target in the last f(x_i) position
        _, y_next = function_dataset[self.max_context_size]
        y_vector = torch.cat((torch.tensor([y_next]), torch.zeros(self.dimension - 1)))
        targets[-1] = y_vector

        return inputs, targets


class TransformerEvalDataset(Dataset):
    def __init__(self, function_dataset, max_context_size):
        self.function_dataset = function_dataset
        self.max_context_size = max_context_size
        self.dimension = function_dataset[0][0].shape[0]

    def __len__(self):
        return self.max_context_size - 1

    def __getitem__(self, idx):
        context_size = idx + 1
        inputs = []

        # Collect inputs and targets
        for i in range(context_size):
            x, y = self.function_dataset[i]
            y_vector = torch.cat((torch.tensor([y]), torch.zeros(self.dimension - 1)))
            inputs.extend([x.tolist(), y_vector.tolist()])

        # Add the next input (x_{n+1}) with zero as the function value
        x_next, _ = self.function_dataset[context_size]
        inputs.append(x_next.tolist())
        inputs.append([0.0] * self.dimension)

        # Pad inputs to the max context size
        while len(inputs) < (self.max_context_size * 2):
            inputs.append([0.0] * self.dimension)

        # Convert lists to tensors
        inputs = torch.tensor(inputs[:self.max_context_size * 2], dtype=torch.float32).view(-1, self.dimension)

        # Create targets with zero padding
        targets = torch.zeros((self.max_context_size * 2, self.dimension), dtype=torch.float32)

        # Place the actual target in the correct f(x_i) position
        if context_size < len(self.function_dataset):
            _, y_next = self.function_dataset[context_size]
            y_vector = torch.cat((torch.tensor([y_next]), torch.zeros(self.dimension - 1)))
            targets[(context_size+1) * 2 - 1] = y_vector

        return inputs, targets


if __name__ == "__main__":
    # Test the TransformerTrainDataset and TransformerEvalDataset with more samples
    num_samples = 10000
    input_dim = 2
    max_context_size = 5
    noise_std = 0.0
    function_class = 'linear'

    transformer_train_dataset = TransformerTrainDataset(num_samples=num_samples, input_dim=input_dim, max_context_size=max_context_size, function_class=function_class, noise_std=noise_std)

    # Debug TransformerTrainDataset
    print(f"TransformerTrainDataset samples (count = {len(transformer_train_dataset)}):")
    for i in range(5):
        inputs, targets = transformer_train_dataset[i]
        print(f"Sample {i} - Inputs: {inputs.numpy()}, Targets: {targets.numpy()}")

    # Create a static function dataset for evaluation
    function_dataset = FunctionDataset(num_samples=num_samples, input_dim=input_dim, function_class='linear', noise_std=noise_std)
    transformer_eval_dataset = TransformerEvalDataset(function_dataset, max_context_size=max_context_size)

    # Debug TransformerEvalDataset
    print("\n\nTransformerEvalDataset samples:")
    for i in range(4):
        inputs, targets = transformer_eval_dataset[i]
        print(f"Sample {i} - Inputs: {inputs.numpy()}, Targets: {targets.numpy()}")

