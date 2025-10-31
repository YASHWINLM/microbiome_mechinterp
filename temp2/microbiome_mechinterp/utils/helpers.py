import torch
import matplotlib.pyplot as plt

def get_device():
    """Get the appropriate device for computation"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def add_noise(data, noise_factor=0.05):
    """Add noise to input data for regularization"""
    noise = torch.randn_like(data) * noise_factor * data.std()
    return data + noise

def plot_training_curves(train_losses, val_losses=None, title="Training Curves", save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.show()
