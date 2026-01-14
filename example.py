"""
SWGL: Stability-Weighted Gradient Learning - Professional Example

Comprehensive demonstration of SWGL integration in a continual learning scenario.
Includes synthetic task generation, forgetting measurement, and visualization.

Author: Walid Moqassim
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import numpy as np

from swgl import SWGL


class ContinualEvaluator:
    """
    Tracks model performance across sequential tasks and measures catastrophic forgetting.
    """
    
    def __init__(self):
        self.task_history: Dict[int, List[float]] = {}
    
    def evaluate_all_tasks(self, model: nn.Module, tasks: List[Tuple[torch.Tensor, torch.Tensor]], 
                          device: torch.device) -> Dict[str, float]:
        """
        Evaluates model on all seen tasks and computes forgetting metrics.
        """
        model.eval()
        accuracies = []
        
        with torch.no_grad():
            for task_id, (data, labels) in enumerate(tasks):
                data, labels = data.to(device), labels.to(device)
                preds = torch.argmax(model(data), dim=1)
                acc = (preds == labels).float().mean().item() * 100
                accuracies.append(acc)
                
                # Track history
                if task_id not in self.task_history:
                    self.task_history[task_id] = []
                self.task_history[task_id].append(acc)
        
        # Compute forgetting (max accuracy drop for any task)
        forgetting = 0.0
        if len(accuracies) > 1:
            historical_peaks = [max(self.task_history[i]) for i in range(len(accuracies) - 1)]
            if historical_peaks:
                forgetting = max(max(historical_peaks) - min(accuracies[:-1]), 0.0)
        
        model.train()
        return {
            "per_task": accuracies,
            "average": np.mean(accuracies),
            "forgetting": forgetting
        }


def create_model(input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10) -> nn.Module:
    """Creates a simple MLP for demonstration."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, num_classes)
    )


def generate_task(task_id: int, num_samples: int = 2000, input_dim: int = 784) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic classification data with progressive distribution shift.
    Higher task_id → larger shift in mean and variance.
    """
    shift = task_id * 0.3
    scale = 1.0 + (task_id * 0.15)
    
    data = torch.randn(num_samples, input_dim) * scale + shift
    labels = torch.randint(0, 10, (num_samples,))
    
    return data, labels


def train_continual(
    model: nn.Module,
    num_tasks: int = 5,
    epochs_per_task: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    swgl_config: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Main training loop for sequential task learning with SWGL.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.train()
    
    # SWGL configuration
    swgl_config = swgl_config or {"alpha": 0.99, "lambda_reg": 0.01, "max_boost": 10.0}
    
    # Initialize optimizer and SWGL wrapper
    base_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    swgl = SWGL(model.parameters(), **swgl_config)
    
    criterion = nn.CrossEntropyLoss()
    evaluator = ContinualEvaluator()
    
    print(f"\nDevice: {device}")
    print(f"SWGL config: {swgl_config}\n")
    
    all_tasks = []
    
    # Sequential training loop
    for task_id in range(num_tasks):
        print(f"--- Task {task_id + 1}/{num_tasks} ---")
        
        # Generate and prepare task data
        task_data, task_labels = generate_task(task_id, num_samples=2000)
        task_data, task_labels = task_data.to(device), task_labels.to(device)
        all_tasks.append((task_data, task_labels))
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(task_data, task_labels),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Train for specified epochs
        for epoch in range(epochs_per_task):
            epoch_losses = []
            
            for batch_x, batch_y in dataloader:
                # Forward + backward
                base_opt.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                
                # Apply SWGL modulation before optimizer step
                swgl.step()
                base_opt.step()
                
                epoch_losses.append(loss.item())
            
            # Periodic evaluation
            if (epoch + 1) % max(1, epochs_per_task // 3) == 0:
                avg_loss = np.mean(epoch_losses)
                metrics = evaluator.evaluate_all_tasks(model, all_tasks, device)
                
                print(
                    f"  Epoch {epoch + 1:2d}/{epochs_per_task} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Avg Acc: {metrics['average']:6.2f}% | "
                    f"Forget: {metrics['forgetting']:5.2f}%"
                )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    
    # Final results
    final = evaluator.evaluate_all_tasks(model, all_tasks, device)
    print(f"\nFinal Average Accuracy: {final['average']:6.2f}%")
    print(f"Final Forgetting:       {final['forgetting']:6.2f}%")
    
    return {"evaluator": evaluator, "final_metrics": final, "model": model}


def plot_results(evaluator: ContinualEvaluator, save_path: str = "continual_results.png"):
    """Generates publication-quality visualization of continual learning performance."""
    if not evaluator.task_history:
        print("No data to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for task_id in sorted(evaluator.task_history.keys()):
        steps = range(len(evaluator.task_history[task_id]))
        ax.plot(steps, evaluator.task_history[task_id], 
                marker='o', linewidth=2, label=f'Task {task_id + 1}')
    
    ax.set_xlabel('Training Progress', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Continual Learning: Task Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")


def main():
    """Entry point for running the complete SWGL demonstration."""
    print("\n" + "=" * 50)
    print("SWGL CONTINUAL LEARNING DEMO")
    print("=" * 50)
    
    # Configuration
    config = {
        "num_tasks": 5,
        "epochs_per_task": 15,
        "batch_size": 64,
        "learning_rate": 0.001,
        "swgl_config": {"alpha": 0.99, "lambda_reg": 0.01, "max_boost": 10.0}
    }
    
    # Create model and train
    model = create_model()
    results = train_continual(model=model, **config)
    
    # Visualize
    plot_results(results["evaluator"])
    
    print("\n✓ Demo completed successfully.")
    print("✓ Check the generated plot for performance visualization.")


if __name__ == "__main__":
    main()
                 
