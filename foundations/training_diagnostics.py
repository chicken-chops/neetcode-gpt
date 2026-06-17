import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        
        
        stats = []

        # Hook function
        def hook_fn(module, input, output):

            mean_val = torch.mean(output).item()
            std_val = torch.std(output).item()

            dead_neurons_mask = torch.all(output <= 0, dim=0)

            dead_fraction = torch.mean(dead_neurons_mask.float()).item()

            stats.append({
                'mean': round(mean_val, 4),
                'std': round(std_val, 4),
                'dead_fraction': round(dead_fraction, 4)
            })

        hooks = []

        # Attach hooks
        for layer in model.modules():

            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        # Run forward pass
        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return stats
            

        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
    

        stats = []

        # Remove old gradients
        model.zero_grad()

        # Forward pass
        predictions = model(x)

        # Loss function
        loss_fn = nn.MSELoss()

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backpropagation
        loss.backward()

        # Inspect gradients
        for layer in model.modules():

            if isinstance(layer, nn.Linear):

                grad = layer.weight.grad

                mean_val = torch.mean(grad).item()

                std_val = torch.std(grad).item()

                norm_val = torch.norm(grad).item()

                stats.append({
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4),
                    'norm': round(norm_val, 4)
                })

        return stats
           

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        
        # Check dead neurons
        for stat in activation_stats:

            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'

        # Check exploding gradients
        for stat in gradient_stats:

            if stat['norm'] > 100:
                return 'exploding_gradients'

        # Check vanishing gradients
        for stat in gradient_stats:

            if stat['norm'] < 1e-5:
                return 'vanishing_gradients'

        # Otherwise healthy
        return 'healthy'

