import torch
import torch.nn.functional as F
from FunctionDataset import FunctionDataset
from TransformerDataset import TransformerEvalDataset
from model import GPT
from utils import set_seed
import json
import matplotlib.pyplot as plt

def evaluate_transformer(model, trainer_device, input_dim, max_context_size):
    model.eval()
    squared_errors = []

    # Ensure enough samples for all context sizes
    function_dataset = FunctionDataset(num_samples=max_context_size + 1, input_dim=input_dim, function_class='linear', noise_std=0.0)
    transformer_dataset = TransformerEvalDataset(function_dataset, max_context_size=max_context_size)

    with torch.no_grad():
        for i in range(len(transformer_dataset)):
            inputs, targets = transformer_dataset[i]
            inputs, targets = inputs.unsqueeze(0).to(trainer_device), targets.unsqueeze(0).to(trainer_device)

            # Provide the model with context examples including the new input, and predict the new target
            predictions, _ = model(inputs, targets)

            # Identify all non-zero targets
            non_zero_indices = (targets != 0).nonzero(as_tuple=True)
            batch_indices = non_zero_indices[0]
            target_indices = non_zero_indices[1]

            # Collect all non-zero target values and corresponding predictions
            target_values = targets[batch_indices, target_indices, 0]
            prediction_values = predictions.squeeze(-1)[batch_indices, target_indices]

            # Calculate the squared error for each non-zero target
            for pred, target in zip(prediction_values, target_values):
                squared_error = F.mse_loss(pred, target, reduction='sum').item()
                squared_errors.append(squared_error)

    mean_squared_error = sum(squared_errors) / len(squared_errors)
    return squared_errors, mean_squared_error

configFile = 'config.json' # 'config.json' or 'curr_config.json'
modelFile = 'trained_model.pth' # 'trained_model.pth' | 'curr_trained_mode.pth'

print(f'Evaluating Model: {modelFile}, {configFile}')

# Load the configurations
with open('config.json', 'r') as f:
    config = json.load(f)

# Set seed and create the model
set_seed(42)

model_config = GPT.get_default_config()
model_config.model_type = config['model_type']
model_config.vocab_size = config['input_dim'] + 1  # Adding 1 for the target dimension
model_config.block_size = (2 * config['max_context_size']) * config['input_dim']  # Updated block_size based on max context size
model_config.input_dim = config['input_dim']  # Set the input dimension

model = GPT(model_config)
print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6,))

# Load the trained model
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
print("Model loaded from 'trained_model.pth'")

# Evaluate the model
trainer_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(trainer_device)
squared_errors, mean_squared_error = evaluate_transformer(model, trainer_device, config['input_dim'], config['max_context_size'])

# Plotting squared error vs in-context examples
plt.plot(range(1, len(squared_errors) + 1), squared_errors, marker='o')
plt.xlabel('Number of In-Context Examples')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Number of In-Context Examples')
plt.grid(True)
plt.show()

