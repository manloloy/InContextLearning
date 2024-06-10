import torch
import matplotlib.pyplot as plt
from model import GPT
from trainer import Trainer
from utils import set_seed
import json
from TransformerDataset import TransformerTrainDataset

DEBUG = False  # Set to True to enable debugging

print('starting ....')

# Configuration
config = {
    'num_samples': 1000000,
    'input_dim': 10,
    'function_class': 'linear',
    'noise_std': 0,
    'model_type': 'standard',
    'batch_size': 64,
    'min_context_size': 1,
    'max_context_size': 40,
    'max_iters': 100000,
    'learning_rate': 10e-4,
    'num_workers': 0
}

# Create the TransformerTrainDataset
transformer_train_dataset = TransformerTrainDataset(num_samples=config['num_samples'], input_dim=config['input_dim'], max_context_size=config['max_context_size'], function_class=config['function_class'], noise_std=config['noise_std'])

# Debug TransformerTrainDataset
if DEBUG:
    print("TransformerTrainDataset samples:")
    for i in range(3):
        inputs, targets = transformer_train_dataset[i]
        print(f"Sample {i} - Inputs: {inputs.numpy()}, Targets: {targets.numpy()}")

# Model configuration
set_seed(42)

model_config = GPT.get_default_config()
model_config.model_type = config['model_type']
model_config.vocab_size = config['input_dim'] + 1  # Adding 1 for the target dimension
model_config.block_size = (2 * config['max_context_size']) * config['input_dim']  # Updated block_size based on max context size
model_config.input_dim = config['input_dim']  # Set the input dimension

# Create the model
model = GPT(model_config)
print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6,))

# Training configuration
train_config = Trainer.get_default_config()
train_config.learning_rate = config['learning_rate']
train_config.max_iters = config['max_iters']
train_config.num_workers = config['num_workers']
train_config.batch_size = config['batch_size']

# Create the trainer
trainer = Trainer(train_config, model, transformer_train_dataset)
print("running on device", trainer.device)

# Initialize weights for tracking
initial_scalar_head_weight = model.scalar_head.weight.clone().detach()
initial_wte_weight = model.transformer.wte.weight.clone().detach()

# Lists to store weights and losses
trainer.scalar_head_weights = []
trainer.wte_weights = []
trainer.train_losses = []

# Define a callback for batch end to print training status and debug weights
def batch_end_callback(trainer):
    current_scalar_head_weight = model.scalar_head.weight.clone().detach()
    current_wte_weight = model.transformer.wte.weight.clone().detach()

    scalar_head_weight_changed = not torch.equal(initial_scalar_head_weight, current_scalar_head_weight)
    wte_weight_changed = not torch.equal(initial_wte_weight, current_wte_weight)
    
    # Store weights for plotting
    trainer.scalar_head_weights.append(current_scalar_head_weight.cpu().numpy())
    trainer.wte_weights.append(current_wte_weight.cpu().numpy())
    trainer.train_losses.append(trainer.loss.item())
    
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; Scalar Head Weights Changed: {scalar_head_weight_changed}; WTE Weights Changed: {wte_weight_changed}")

    # Update the initial weights
    initial_scalar_head_weight.copy_(current_scalar_head_weight)
    initial_wte_weight.copy_(current_wte_weight)

trainer.set_callback('on_batch_end', batch_end_callback)

# Run the training
trainer.run()

# Save the model and configurations
torch.save(model.state_dict(), 'trained_model.pth')
with open('config.json', 'w') as f:
    json.dump(config, f)
print("Model and configurations saved.")

# Plot the stored weight values
plt.figure(figsize=(12, 6))
for i in range(min(len(trainer.scalar_head_weights[0].flatten()), 5)):  # Plot up to 5 scalar head weights
    plt.plot([w.flatten()[i] for w in trainer.scalar_head_weights], label=f'Scalar Head Weight {i}')
plt.xlabel('Iterations')
plt.ylabel('Weight Value')
plt.title('Scalar Head Weights Over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for i in range(min(len(trainer.wte_weights[0].flatten()), 5)):  # Plot up to 5 wte weights
    plt.plot([w.flatten()[i] for w in trainer.wte_weights], label=f'WTE Weight {i}')
plt.xlabel('Iterations')
plt.ylabel('Weight Value')
plt.title('Word Token Embedding Weights Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the training loss
plt.figure(figsize=(12, 6))
plt.plot(trainer.train_losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()
