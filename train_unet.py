#!/usr/bin/env python3
"""
Train Multi-Channel IQUNet Model
Task: Identity mapping (autoencoder)
Training data: FP32, uniform distribution [-1.5, 1.5]
Dataset: 10,000 fixed samples
Input: 32 channels (16 I + 16 Q)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import MultiChannelIQUNet


def generate_dataset(num_samples=10000, num_channels=32, seed=42):
    """
    Generate fixed dataset of multi-channel IQ samples
    
    Args:
        num_samples: Number of samples to generate
        num_channels: Number of channels (32 = 16 I + 16 Q)
        seed: Random seed for reproducibility
    
    Returns:
        torch.Tensor of shape (num_samples, 32, 5120, 1)
    """
    print(f"Generating dataset with {num_samples} samples...")
    print(f"Channels: {num_channels} (16 I + 16 Q)")
    np.random.seed(seed)
    data = np.random.uniform(-1.5, 1.5, (num_samples, num_channels, 5120, 1)).astype(np.float32)
    dataset = torch.from_numpy(data)
    print(f"✓ Dataset generated: {dataset.shape}")
    print(f"  Range: [{dataset.min():.3f}, {dataset.max():.3f}]")
    return dataset


def train_model(model, dataset, num_epochs=100, batch_size=16, learning_rate=0.001, early_stopping_patience=5):
    """
    Train the Multi-Channel IQUNet model with early stopping
    
    Args:
        model: MultiChannelIQUNet model instance
        dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training (reduced for larger model)
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait before early stopping
    
    Returns:
        train_losses: List of average losses per epoch
        train_correlations: List of average correlations per epoch
    """
    print("="*70)
    print("Training Multi-Channel IQUNet Model")
    print("="*70)
    print(f"Architecture: U-Net with 1D convolutions")
    print(f"Task: Identity mapping (autoencoder)")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Input channels: 32 (16 I + 16 Q)")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Training on: FP32 data, range [-1.5, 1.5]")
    print()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_losses = []
    train_correlations = []
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    num_batches = len(dataset) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_corr = 0.0
        
        # Shuffle dataset
        indices = torch.randperm(len(dataset))
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            inputs = dataset[batch_indices]
            targets = inputs.clone()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate correlation
            with torch.no_grad():
                inputs_flat = inputs.numpy().flatten()
                outputs_flat = outputs.numpy().flatten()
                corr = np.corrcoef(inputs_flat, outputs_flat)[0, 1]
                epoch_corr += corr
        
        avg_loss = epoch_loss / num_batches
        avg_corr = epoch_corr / num_batches
        
        train_losses.append(avg_loss)
        train_correlations.append(avg_corr)
        
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.6f} | Correlation: {avg_corr:.4f} | ✓ Best")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.6f} | Correlation: {avg_corr:.4f} | Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print()
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best loss: {best_loss:.6f}")
                model.load_state_dict(best_model_state)
                break
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Final Loss: {train_losses[-1]:.6f}")
    print(f"Final Correlation: {train_correlations[-1]:.4f}")
    print()
    
    return train_losses, train_correlations


def test_model(model, num_samples=5):
    """Test the trained model on new samples"""
    print("Testing model on new samples...")
    model.eval()
    
    # Generate test samples
    test_inputs = generate_dataset(num_samples, num_channels=32, seed=99999)
    
    with torch.no_grad():
        test_outputs = model(test_inputs)
    
    test_inputs_np = test_inputs.numpy()
    test_outputs_np = test_outputs.numpy()
    
    print("Test Results:")
    print("-" * 70)
    for i in range(num_samples):
        inp = test_inputs_np[i].flatten()
        out = test_outputs_np[i].flatten()
        corr = np.corrcoef(inp, out)[0, 1]
        mse = np.mean((inp - out) ** 2)
        print(f"Sample {i+1}:")
        print(f"  Correlation: {corr:.4f}, MSE: {mse:.6f}")
        print(f"  Input range:  [{inp.min():.3f}, {inp.max():.3f}]")
        print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
        print()


def plot_training_history(losses, correlations, save_path='results/training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(losses, linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot correlation
    ax2.plot(correlations, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Training Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.close()


def main():
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Create model with 32 input channels and dropout
    model = MultiChannelIQUNet(input_channels=32, num_filters=64, dropout_rate=0.1)
    
    print()
    print("Model Architecture:")
    print(model)
    print()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print()
    
    # Generate dataset with 32 channels
    dataset = generate_dataset(num_samples=10000, num_channels=32, seed=42)
    print()
    
    # Train with early stopping (patience=5, smaller batch size)
    losses, correlations = train_model(
        model, dataset, 
        num_epochs=100, 
        batch_size=16,
        learning_rate=0.001,
        early_stopping_patience=5
    )
    
    # Test
    test_model(model, num_samples=5)
    
    # Save PyTorch model
    model_path = 'models/multichannel_unet_trained.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ PyTorch model saved to: {model_path}")
    print()
    
    # Plot training history
    plot_training_history(losses, correlations)
    
    print("="*70)
    print("✓ Training pipeline complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Export to ONNX:")
    print("   python3 export_onnx.py")
    print()
    print("2. Compile for Axelera NPU:")
    print("   bash compile_for_npu.sh")
    print()
    print("3. Run live visualization:")
    print("   python3 demo_live_multichannel.py")
    print()


if __name__ == "__main__":
    main()
