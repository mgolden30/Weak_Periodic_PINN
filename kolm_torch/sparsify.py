import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from lib.model import HydroNetwork  # Import your model definition

# Load the pre-trained model
pretrained_model_path = 'trained_model.pth'
pretrained_model = torch.load(pretrained_model_path)

# Instantiate a new model for sparsification (with the same architecture)
sparsified_model = HydroNetwork(streamnetwork=None)  # Instantiate the model (you may need to modify this based on your model architecture)

# Print sparsity before pruning
total_params_before = sum(p.numel() for p in sparsified_model.parameters())
total_sparse_params_before = sum(1 for p, _ in sparsified_model.named_parameters() if "weight" in p and hasattr(p, "orig"))
print(f"Total parameters before pruning: {total_params_before}")
print(f"Total sparse parameters before pruning: {total_sparse_params_before}")
print(f"Sparsity before pruning: {total_sparse_params_before / total_params_before * 100:.2f}%")

# Apply weight pruning to the new model based on the pre-trained model
for name, module in pretrained_model.named_modules():
    if isinstance(module.weight, nn.Parameter):
        prune.l1_unstructured(module, name='weight', amount=0.2)  # You can adjust the pruning amount

# Remove the pruned parameters
for name, module in pretrained_model.named_modules():
    if isinstance(module.weight, nn.Parameter):
        prune.remove(module, name='weight')

# Save the sparsified model with pruning information
sparsified_model_path = 'sparsified_model.pth'
torch.save({
    'model_state_dict': pretrained_model.state_dict(),
    'pruning_masks': {name: getattr(module, 'weight_mask') for name, module in pretrained_model.named_modules() if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter)},
}, sparsified_model_path)

# Print sparsity after pruning
total_params_after = sum(p.numel() for p in pretrained_model.parameters())
total_sparse_params_after = sum(1 for p, _ in pretrained_model.named_parameters() if "weight" in p and hasattr(p, "orig"))
print(f"\nTotal parameters after pruning: {total_params_after}")
print(f"Total sparse parameters after pruning: {total_sparse_params_after}")
print(f"Sparsity after pruning: {total_sparse_params_after / total_params_after * 100:.2f}%")

